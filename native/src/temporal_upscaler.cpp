#include "temporal_upscaler.h"

#include "shaders/taa.spv.h"

#include <vector>

namespace mcrt {

namespace {

struct TaaParams {
    uint32_t outputSize[2];
    uint32_t historyIndex;
    uint32_t resetHistory;
};

} // namespace

TemporalUpscaler::TemporalUpscaler(VkContext& ctx, DeletionQueue& deletion) : ctx_(ctx), deletion_(deletion) {
    VkDevice device = ctx_.device();
    const VkShaderStageFlags cs = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, cs, nullptr},  // current HDR (internal)
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // positions (internal)
        {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2, cs, nullptr},  // history, read
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2, cs, nullptr},  // history, write
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, cs, nullptr},  // output (rgba8)
        {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, cs, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, // depth in
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, cs, nullptr}, // depth out
        {8, VK_DESCRIPTOR_TYPE_SAMPLER, 1, cs, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(std::size(bindings));
    layoutInfo.pBindings = bindings;
    MCRT_VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &setLayout_));
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kFramesInFlight},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kFramesInFlight;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;
    MCRT_VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool_));
    for (VkDescriptorSet& set : sets_) {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = pool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &setLayout_;
        MCRT_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));
    }
    VkPushConstantRange push{cs, 0, sizeof(TaaParams)};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push;
    MCRT_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(kSpirv_taa);
    moduleInfo.pCode = kSpirv_taa;
    VkShaderModule module;
    MCRT_VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "temporalUpscale";
    info.layout = pipelineLayout_;
    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_);
    vkDestroyShaderModule(device, module, nullptr);
    MCRT_VK_CHECK(result);

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    MCRT_VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &sampler_));
}

TemporalUpscaler::~TemporalUpscaler() {
    VkDevice device = ctx_.device();
    for (Image& image : history_)
        ctx_.destroyImage(image);
    ctx_.destroyImage(output_);
    ctx_.destroyBuffer(depth_);
    vkDestroySampler(device, sampler_, nullptr);
    vkDestroyPipeline(device, pipeline_, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(device, pool_, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout_, nullptr);
}

bool TemporalUpscaler::ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue) {
    if (output_.image && width == width_ && height == height_)
        return false;
    std::array<Image, 3> oldImages{history_[0], history_[1], output_};
    Buffer oldDepth = depth_;
    deletion_.push(retireValue, [this, oldImages, oldDepth]() mutable {
        for (Image& image : oldImages)
            ctx_.destroyImage(image);
        ctx_.destroyBuffer(oldDepth);
    });
    for (Image& image : history_)
        image = ctx_.createStorageImage(width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    output_ = ctx_.createStorageImage(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    depth_ = ctx_.createBuffer(VkDeviceSize(width) * height * sizeof(float),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               MemoryKind::DeviceLocal);
    width_ = width;
    height_ = height;
    needsInit_ = true;
    return true;
}

void TemporalUpscaler::recordTargetInit(VkCommandBuffer cmd) {
    if (!needsInit_)
        return;
    std::vector<VkImageMemoryBarrier> barriers;
    for (const Image* image : {&history_[0], &history_[1], &output_}) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image->image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barriers.push_back(barrier);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
    needsInit_ = false;
}

void TemporalUpscaler::record(VkCommandBuffer cmd, uint32_t slot, const Buffer& frameUniforms,
                              VkDeviceSize uniformSize, VkImageView currentHdr, VkImageView positions,
                              const Buffer& depthIn, bool resetHistory) {
    historyIndex_ ^= 1;
    VkDescriptorSet set = sets_[slot];
    VkDescriptorImageInfo current{VK_NULL_HANDLE, currentHdr, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo positionInfo{VK_NULL_HANDLE, positions, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo history[2] = {{VK_NULL_HANDLE, history_[0].view, VK_IMAGE_LAYOUT_GENERAL},
                                        {VK_NULL_HANDLE, history_[1].view, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorImageInfo output{VK_NULL_HANDLE, output_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo uniforms{frameUniforms.buffer, 0, uniformSize};
    VkDescriptorBufferInfo depthInInfo{depthIn.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo depthOutInfo{depth_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo samplerInfo{sampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

    VkWriteDescriptorSet writes[9]{};
    auto write = [&](uint32_t binding, VkDescriptorType type, uint32_t count) -> VkWriteDescriptorSet& {
        VkWriteDescriptorSet& w = writes[binding];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = count;
        w.descriptorType = type;
        return w;
    };
    write(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1).pImageInfo = &current;
    write(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1).pImageInfo = &positionInfo;
    write(2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2).pImageInfo = history;
    write(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2).pImageInfo = history;
    write(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1).pImageInfo = &output;
    write(5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1).pBufferInfo = &uniforms;
    write(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1).pBufferInfo = &depthInInfo;
    write(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1).pBufferInfo = &depthOutInfo;
    write(8, VK_DESCRIPTOR_TYPE_SAMPLER, 1).pImageInfo = &samplerInfo;
    vkUpdateDescriptorSets(ctx_.device(), 9, writes, 0, nullptr);

    TaaParams params{{width_, height_}, historyIndex_, resetHistory ? 1u : 0u};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    vkCmdDispatch(cmd, (width_ + 7) / 8, (height_ + 7) / 8, 1);
}

} // namespace mcrt
