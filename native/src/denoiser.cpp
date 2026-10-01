#include "denoiser.h"

#include "shaders/denoise.spv.h"

#include <vector>

namespace mcrt {

namespace {

struct PassParams {
    uint32_t stepSize;
    uint32_t readFromA;
    uint32_t historyIndex;
    uint32_t pad;
};

constexpr VkImageUsageFlags kTargetUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

void computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
}

} // namespace

Denoiser::Denoiser(VkContext& ctx, DeletionQueue& deletion) : ctx_(ctx), deletion_(deletion) {
    createPipelines();
    exposureState_ = ctx_.createBuffer(4 * sizeof(float),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       MemoryKind::DeviceLocal);
}

Denoiser::~Denoiser() {
    destroyTargets();
    ctx_.destroyBuffer(exposureState_);
    VkDevice device = ctx_.device();
    for (VkPipeline pipeline : {temporal_, atrous_, compose_, exposure_})
        vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout_, nullptr);
}

void Denoiser::createPipelines() {
    VkDevice device = ctx_.device();
    const VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // noisy illumination
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // positions
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, cs, nullptr},  // normals[2]
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, cs, nullptr},  // history[2]
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, cs, nullptr},  // moments[2]
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // filterA
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // filterB
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // albedo modulation
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // foreground
        {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // output
        {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, // exposure state
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, // luminance partials
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(std::size(bindings));
    layoutInfo.pBindings = bindings;
    MCRT_VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &setLayout_));

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 13 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kFramesInFlight},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kFramesInFlight;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;
    MCRT_VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool_));
    for (VkDescriptorSet& set : sets_) {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = descriptorPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &setLayout_;
        MCRT_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));
    }

    VkPushConstantRange push{cs, 0, sizeof(PassParams)};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push;
    MCRT_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(kSpirv_denoise);
    moduleInfo.pCode = kSpirv_denoise;
    VkShaderModule module;
    MCRT_VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));
    auto create = [&](const char* entry, VkPipeline& out) {
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = entry;
        info.layout = pipelineLayout_;
        return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &out);
    };
    VkResult results[] = {create("temporalAccumulate", temporal_), create("atrous", atrous_),
                          create("compose", compose_), create("exposure", exposure_)};
    vkDestroyShaderModule(device, module, nullptr);
    for (VkResult result : results)
        MCRT_VK_CHECK(result);
}

void Denoiser::destroyTargets() {
    for (Image* image : {&noisy_, &albedo_, &foreground_, &positions_, &normals_[0], &normals_[1], &history_[0],
                         &history_[1], &moments_[0], &moments_[1], &filterA_, &filterB_, &output_})
        ctx_.destroyImage(*image);
    ctx_.destroyBuffer(luminancePartials_);
}

bool Denoiser::ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue) {
    if (output_.image && width == width_ && height == height_)
        return false;

    // Retire the old set as a whole; frames in flight may still read it.
    struct Old {
        std::array<Image, 13> images;
        Buffer partials;
    } old{{noisy_, albedo_, foreground_, positions_, normals_[0], normals_[1], history_[0], history_[1], moments_[0],
           moments_[1], filterA_, filterB_, output_},
          luminancePartials_};
    deletion_.push(retireValue, [this, old]() mutable {
        for (Image& image : old.images)
            ctx_.destroyImage(image);
        ctx_.destroyBuffer(old.partials);
    });

    auto rgba16 = [&]() { return ctx_.createStorageImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, kTargetUsage); };
    noisy_ = rgba16();
    albedo_ = rgba16();
    foreground_ = rgba16();
    positions_ = ctx_.createStorageImage(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, kTargetUsage);
    for (int i = 0; i < 2; ++i) {
        normals_[i] = rgba16();
        history_[i] = rgba16();
        moments_[i] = rgba16();
    }
    filterA_ = rgba16();
    filterB_ = rgba16();
    output_ = ctx_.createStorageImage(width, height, VK_FORMAT_R8G8B8A8_UNORM,
                                      kTargetUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const uint32_t groups = ((width + 15) / 16) * ((height + 15) / 16);
    luminancePartials_ = ctx_.createBuffer(groups * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           MemoryKind::DeviceLocal);
    width_ = width;
    height_ = height;
    targetsNeedInit_ = true;
    return true;
}

void Denoiser::recordTargetInit(VkCommandBuffer cmd) {
    if (!targetsNeedInit_)
        return;
    std::vector<VkImageMemoryBarrier> barriers;
    const Image* images[] = {&noisy_, &albedo_, &foreground_, &positions_, &normals_[0], &normals_[1], &history_[0],
                             &history_[1], &moments_[0], &moments_[1], &filterA_, &filterB_, &output_};
    for (const Image* image : images) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image->image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(barrier);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
    // Empty history (length 0) everywhere, so the first frame starts fresh.
    VkClearColorValue zero{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (const Image* image : images)
        vkCmdClearColorImage(cmd, image->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    vkCmdFillBuffer(cmd, exposureState_.buffer, 0, VK_WHOLE_SIZE, 0);

    VkMemoryBarrier toShaders{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toShaders.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShaders.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &toShaders, 0, nullptr, 0, nullptr);
    targetsNeedInit_ = false;
}

void Denoiser::record(VkCommandBuffer cmd, uint32_t slot, const Buffer& frameUniforms, VkDeviceSize uniformSize,
                      uint32_t historyIndex) {
    VkDescriptorSet set = sets_[slot];
    auto storage = [](const Image& image) { return VkDescriptorImageInfo{VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL}; };
    VkDescriptorImageInfo noisy = storage(noisy_), positions = storage(positions_);
    VkDescriptorImageInfo normals[2] = {storage(normals_[0]), storage(normals_[1])};
    VkDescriptorImageInfo history[2] = {storage(history_[0]), storage(history_[1])};
    VkDescriptorImageInfo moments[2] = {storage(moments_[0]), storage(moments_[1])};
    VkDescriptorImageInfo filterA = storage(filterA_), filterB = storage(filterB_);
    VkDescriptorImageInfo albedo = storage(albedo_), foreground = storage(foreground_), output = storage(output_);
    VkDescriptorBufferInfo uniforms{frameUniforms.buffer, 0, uniformSize};
    VkDescriptorBufferInfo exposure{exposureState_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo partials{luminancePartials_.buffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[13]{};
    auto write = [&](uint32_t binding, VkDescriptorType type, uint32_t count) -> VkWriteDescriptorSet& {
        VkWriteDescriptorSet& w = writes[binding];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = count;
        w.descriptorType = type;
        return w;
    };
    const VkDescriptorType image = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write(0, image, 1).pImageInfo = &noisy;
    write(1, image, 1).pImageInfo = &positions;
    write(2, image, 2).pImageInfo = normals;
    write(3, image, 2).pImageInfo = history;
    write(4, image, 2).pImageInfo = moments;
    write(5, image, 1).pImageInfo = &filterA;
    write(6, image, 1).pImageInfo = &filterB;
    write(7, image, 1).pImageInfo = &albedo;
    write(8, image, 1).pImageInfo = &foreground;
    write(9, image, 1).pImageInfo = &output;
    write(10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1).pBufferInfo = &uniforms;
    write(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1).pBufferInfo = &exposure;
    write(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1).pBufferInfo = &partials;
    vkUpdateDescriptorSets(ctx_.device(), 13, writes, 0, nullptr);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0, nullptr);
    auto push = [&](uint32_t stepSize, uint32_t readFromA) {
        PassParams params{stepSize, readFromA, historyIndex, 0};
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    };
    const uint32_t groups8x = (width_ + 7) / 8, groups8y = (height_ + 7) / 8;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_);
    push(0, 0);
    vkCmdDispatch(cmd, groups8x, groups8y, 1);
    computeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atrous_);
    uint32_t readFromA = 1; // temporal pass writes filterA
    for (uint32_t i = 0; i < kAtrousIterations; ++i) {
        push(1u << i, readFromA);
        vkCmdDispatch(cmd, groups8x, groups8y, 1);
        computeBarrier(cmd);
        readFromA ^= 1;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compose_);
    push(0, readFromA);
    vkCmdDispatch(cmd, (width_ + 15) / 16, (height_ + 15) / 16, 1);
    computeBarrier(cmd);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_);
    push(0, 0);
    vkCmdDispatch(cmd, 1, 1, 1);
}

} // namespace mcrt
