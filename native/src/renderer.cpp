#include "renderer.h"

#include "shaders/pathtrace.spv.h"
#include "shaders/sky.spv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace mcrt {

namespace {

constexpr uint64_t kWaitTimeoutNs = 5'000'000'000ull;

// Shader stages, in pipeline order.
enum Stage : uint32_t {
    kStageRayGen,
    kStagePrimaryMiss,
    kStageShadowMiss,
    kStageClosestHitSolid,
    kStageClosestHitTranslucent,
    kStageAnyHitCutoutPrimary,
    kStageAnyHitCutoutShadow,
    kStageCount
};

// Hit records follow (instance offset + ray type + 2 * geometry index); see pathtrace.slang.
constexpr uint32_t kMissGroups = 2;
constexpr uint32_t kHitGroups = 6;
constexpr uint32_t kGroupCount = 1 + kMissGroups + kHitGroups;

struct FrameUniforms {
    float viewProj[16];
    float invViewProj[16];
    float cameraOffset[4];
    float sunDir[4];
    float moonDir[4];
    float skyColor[4];
    float params[4];
    uint32_t frameInfo[4];
    int32_t cameraBlock[4];
    float prevViewProj[16];
    float prevCameraShift[4];
    float atmosphere[4];
    float taa[4];
};
static_assert(sizeof(FrameUniforms) == 352, "must match FrameUniforms in frame.slang");

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void fullBarrier(VkCommandBuffer cmd) {
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

void destroyAccelerationStructure(VkContext& ctx, AccelerationStructure& as) {
    if (as.handle)
        vkDestroyAccelerationStructureKHR(ctx.device(), as.handle, nullptr);
    ctx.destroyBuffer(as.storage);
    as = {};
}

} // namespace

Renderer::Renderer(std::unique_ptr<VkContext> context) : ctx_(std::move(context)) {
    VkDevice device = ctx_->device();
    sections_ = std::make_unique<SectionManager>(*ctx_, deletion_);
    denoiser_ = std::make_unique<Denoiser>(*ctx_, deletion_);
    upscaler_ = std::make_unique<TemporalUpscaler>(*ctx_, deletion_);
    entities_ = std::make_unique<EntityLayer>(*ctx_, deletion_, *sections_);
    materials_ = std::make_unique<MaterialStore>(*ctx_, deletion_);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx_->queueFamilyIndex();
    MCRT_VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool_));

    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    for (FrameSlot& slot : slots_) {
        MCRT_VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &slot.commandBuffer));
        slot.uniforms = ctx_->createBuffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           MemoryKind::HostUpload);
    }

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &timelineInfo;
    MCRT_VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &timeline_));

    VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 2 * kFramesInFlight;
    MCRT_VK_CHECK(vkCreateQueryPool(device, &queryInfo, nullptr, &timestamps_));
    vkResetQueryPool(device, timestamps_, 0, queryInfo.queryCount);

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    MCRT_VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &atlasSampler_));

    createDescriptors();
    createPipeline();
    createShaderBindingTable();
    createSkyPass();
}

void Renderer::createSkyPass() {
    VkDevice device = ctx_->device();
    skyView_ = ctx_->createStorageImage(kSkyWidth, kSkyHeight, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT);
    cloudView_ = ctx_->createStorageImage(kCloudWidth, kCloudHeight, VK_FORMAT_R16G16B16A16_SFLOAT,
                                          VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT; // azimuth wraps
    samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    MCRT_VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &skySampler_));

    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // sky view
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // cloud view
        {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // sky view, sampled
        {4, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(std::size(bindings));
    layoutInfo.pBindings = bindings;
    MCRT_VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &skySetLayout_));
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kFramesInFlight},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kFramesInFlight;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;
    MCRT_VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &skyPool_));
    for (VkDescriptorSet& set : skySets_) {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = skyPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &skySetLayout_;
        MCRT_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &set));
    }
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &skySetLayout_;
    MCRT_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &skyPipelineLayout_));

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(kSpirv_sky);
    moduleInfo.pCode = kSpirv_sky;
    VkShaderModule module;
    MCRT_VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));
    auto create = [&](const char* entry, VkPipeline& out) {
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = entry;
        info.layout = skyPipelineLayout_;
        return vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &out);
    };
    VkResult skyResult = create("skyViewLut", skyPipeline_);
    VkResult cloudResult = create("cloudViewLut", cloudPipeline_);
    vkDestroyShaderModule(device, module, nullptr);
    MCRT_VK_CHECK(skyResult);
    MCRT_VK_CHECK(cloudResult);
}

void Renderer::recordSkyPass(VkCommandBuffer cmd, FrameSlot& slot, uint32_t slotIndex) {
    if (skyViewNeedsInit_) {
        VkImageMemoryBarrier barriers[2]{};
        VkImage images[2] = {skyView_.image, cloudView_.image};
        for (int i = 0; i < 2; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = images[i];
            barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 2, barriers);
        skyViewNeedsInit_ = false;
    }
    VkDescriptorImageInfo skyStorage{VK_NULL_HANDLE, skyView_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo uniformInfo{slot.uniforms.buffer, 0, sizeof(FrameUniforms)};
    VkDescriptorImageInfo cloudStorage{VK_NULL_HANDLE, cloudView_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo skySampled{VK_NULL_HANDLE, skyView_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo samplerInfo{skySampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet writes[5]{};
    const VkDescriptorType types[5] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                       VK_DESCRIPTOR_TYPE_SAMPLER};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = skySets_[slotIndex];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = types[i];
    }
    writes[0].pImageInfo = &skyStorage;
    writes[1].pBufferInfo = &uniformInfo;
    writes[2].pImageInfo = &cloudStorage;
    writes[3].pImageInfo = &skySampled;
    writes[4].pImageInfo = &samplerInfo;
    vkUpdateDescriptorSets(ctx_->device(), 5, writes, 0, nullptr);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skyPipelineLayout_, 0, 1, &skySets_[slotIndex], 0,
                            nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skyPipeline_);
    vkCmdDispatch(cmd, (kSkyWidth + 7) / 8, (kSkyHeight + 7) / 8, 1);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cloudPipeline_);
    vkCmdDispatch(cmd, (kCloudWidth + 7) / 8, (kCloudHeight + 7) / 8, 1);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_READ_BIT);
}

Renderer::~Renderer() {
    try {
        waitForValue(lastSignalValue_);
    } catch (const VulkanError&) {
        // The device is going away regardless; release what we can.
    }
    VkDevice device = ctx_->device();
    deletion_.flushAll();
    sections_.reset();
    destroyAccelerationStructure(*ctx_, tlas_);
    ctx_->destroyBuffer(tlasScratch_);
    denoiser_.reset();
    upscaler_.reset();
    entities_.reset();
    materials_.reset();
    ctx_->destroyImage(skyView_);
    ctx_->destroyImage(cloudView_);
    vkDestroyPipeline(device, cloudPipeline_, nullptr);
    vkDestroySampler(device, skySampler_, nullptr);
    vkDestroyPipeline(device, skyPipeline_, nullptr);
    vkDestroyPipelineLayout(device, skyPipelineLayout_, nullptr);
    vkDestroyDescriptorPool(device, skyPool_, nullptr);
    vkDestroyDescriptorSetLayout(device, skySetLayout_, nullptr);
    ctx_->destroyBuffer(depth_);
    if (atlasView_)
        vkDestroyImageView(device, atlasView_, nullptr);
    for (FrameSlot& slot : slots_) {
        ctx_->destroyBuffer(slot.uniforms);
        ctx_->destroyBuffer(slot.instances);
    }
    ctx_->destroyBuffer(sbt_);
    vkDestroySampler(device, atlasSampler_, nullptr);
    vkDestroyPipeline(device, pipeline_, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool_, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout_, nullptr);
    vkDestroySemaphore(device, timeline_, nullptr);
    vkDestroyQueryPool(device, timestamps_, nullptr);
    vkDestroyCommandPool(device, commandPool_, nullptr);
}

void Renderer::createDescriptors() {
    VkDevice device = ctx_->device();
    const VkShaderStageFlags rgen = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    const VkShaderStageFlags hits = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, rgen, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr},  // noisy illumination
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr},  // albedo modulation
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr},
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, rgen | hits | VK_SHADER_STAGE_MISS_BIT_KHR, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, hits, nullptr},
        {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, hits, nullptr},
        {7, VK_DESCRIPTOR_TYPE_SAMPLER, 1, hits, nullptr},
        {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr},
        {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr}, // foreground
        {11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr}, // positions
        {12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr}, // normals (current)
        {13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, hits, nullptr},  // material albedo array
        {14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, hits, nullptr},  // material normal array
        {15, VK_DESCRIPTOR_TYPE_SAMPLER, 1, hits, nullptr},
        {16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, hits, nullptr}, // material params
        {17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, rgen | hits, nullptr}, // sky-view LUT
        {18, VK_DESCRIPTOR_TYPE_SAMPLER, 1, rgen | hits, nullptr},
        {19, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, rgen | hits, nullptr}, // cloud LUT
        {20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, hits, nullptr},         // material roughness/ao array
        {21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr},         // reservoirs (previous frame)
        {22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr},         // reservoirs (this frame)
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(std::size(bindings));
    layoutInfo.pBindings = bindings;
    MCRT_VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout_));

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 6 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 3 * kFramesInFlight},
    };
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kFramesInFlight;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;
    MCRT_VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool_));

    for (FrameSlot& slot : slots_) {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = descriptorPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout_;
        MCRT_VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &slot.descriptorSet));
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    MCRT_VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout_));
}

void Renderer::createPipeline() {
    VkDevice device = ctx_->device();
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(kSpirv_pathtrace);
    moduleInfo.pCode = kSpirv_pathtrace;
    VkShaderModule module;
    MCRT_VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));

    auto stage = [module](VkShaderStageFlagBits flag, const char* entry) {
        VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage = flag;
        info.module = module;
        info.pName = entry;
        return info;
    };
    VkPipelineShaderStageCreateInfo stages[kStageCount] = {
        stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "rayGen"),
        stage(VK_SHADER_STAGE_MISS_BIT_KHR, "primaryMiss"),
        stage(VK_SHADER_STAGE_MISS_BIT_KHR, "shadowMiss"),
        stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "closestHitSolid"),
        stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "closestHitTranslucent"),
        stage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "anyHitCutoutPrimary"),
        stage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "anyHitCutoutShadow"),
    };

    auto general = [](uint32_t shader) {
        VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader = shader;
        group.closestHitShader = group.anyHitShader = group.intersectionShader = VK_SHADER_UNUSED_KHR;
        return group;
    };
    auto hitGroup = [](uint32_t closestHit, uint32_t anyHit) {
        VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        group.generalShader = group.intersectionShader = VK_SHADER_UNUSED_KHR;
        group.closestHitShader = closestHit;
        group.anyHitShader = anyHit;
        return group;
    };
    const uint32_t none = VK_SHADER_UNUSED_KHR;
    VkRayTracingShaderGroupCreateInfoKHR groups[kGroupCount] = {
        general(kStageRayGen),
        general(kStagePrimaryMiss),
        general(kStageShadowMiss),
        hitGroup(kStageClosestHitSolid, none),                              // solid, primary
        hitGroup(none, none),                                               // solid, shadow
        hitGroup(kStageClosestHitSolid, kStageAnyHitCutoutPrimary),         // cutout, primary
        hitGroup(none, kStageAnyHitCutoutShadow),                           // cutout, shadow
        hitGroup(kStageClosestHitTranslucent, none),                        // translucent, primary
        hitGroup(none, none),                                               // translucent, shadow (masked out)
    };

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineInfo.stageCount = kStageCount;
    pipelineInfo.pStages = stages;
    pipelineInfo.groupCount = kGroupCount;
    pipelineInfo.pGroups = groups;
    pipelineInfo.maxPipelineRayRecursionDepth = 1; // every TraceRay is issued from ray generation
    pipelineInfo.layout = pipelineLayout_;
    VkResult result = vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                                     &pipeline_);
    vkDestroyShaderModule(device, module, nullptr);
    MCRT_VK_CHECK(result);
}

void Renderer::createShaderBindingTable() {
    const auto& props = ctx_->rtProperties();
    const VkDeviceSize handleSize = props.shaderGroupHandleSize;
    const VkDeviceSize handleStride = alignUp(handleSize, props.shaderGroupHandleAlignment);
    const VkDeviceSize baseAlign = props.shaderGroupBaseAlignment;

    raygenRegion_.stride = alignUp(handleStride, baseAlign);
    raygenRegion_.size = raygenRegion_.stride;
    missRegion_.stride = handleStride;
    missRegion_.size = alignUp(kMissGroups * handleStride, baseAlign);
    hitRegion_.stride = handleStride;
    hitRegion_.size = alignUp(kHitGroups * handleStride, baseAlign);

    std::vector<uint8_t> handles(kGroupCount * handleSize);
    MCRT_VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(ctx_->device(), pipeline_, 0, kGroupCount, handles.size(),
                                                        handles.data()));

    const VkDeviceSize missOffset = raygenRegion_.size;
    const VkDeviceSize hitOffset = missOffset + missRegion_.size;
    const VkDeviceSize total = hitOffset + hitRegion_.size;
    std::vector<uint8_t> table(total, 0);
    auto put = [&](VkDeviceSize offset, uint32_t group) {
        std::memcpy(table.data() + offset, handles.data() + group * handleSize, handleSize);
    };
    put(0, 0);
    for (uint32_t i = 0; i < kMissGroups; ++i)
        put(missOffset + i * handleStride, 1 + i);
    for (uint32_t i = 0; i < kHitGroups; ++i)
        put(hitOffset + i * handleStride, 1 + kMissGroups + i);

    sbt_ = ctx_->createBuffer(total, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, MemoryKind::HostUpload, baseAlign);
    ctx_->writeBuffer(sbt_, table.data(), total);
    raygenRegion_.deviceAddress = sbt_.address;
    missRegion_.deviceAddress = sbt_.address + missOffset;
    hitRegion_.deviceAddress = sbt_.address + hitOffset;
}

void Renderer::ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue) {
    if (denoiser_->ensureTargets(width, height, retireValue))
        hasPrevious_ = false; // fresh history
    if (depth_.buffer && depthWidth_ == width && depthHeight_ == height)
        return;
    Buffer oldDepth = depth_;
    deletion_.push(retireValue, [this, oldDepth]() mutable { ctx_->destroyBuffer(oldDepth); });
    for (Buffer& reservoirs : reservoirs_) {
        Buffer old = reservoirs;
        deletion_.push(retireValue, [this, old]() mutable { ctx_->destroyBuffer(old); });
        reservoirs = ctx_->createBuffer(VkDeviceSize(width) * height * 48,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        MemoryKind::DeviceLocal);
    }
    reservoirsNeedClear_ = true;
    depth_ = ctx_->createBuffer(VkDeviceSize(width) * height * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                MemoryKind::DeviceLocal);
    depthWidth_ = width;
    depthHeight_ = height;
}

void Renderer::ensureAtlasView(const McrtFrameInput& input, uint64_t retireValue) {
    const auto image = reinterpret_cast<VkImage>(input.atlas_image);
    if (image == atlasImage_ && atlasView_)
        return;
    if (atlasView_) {
        VkImageView old = atlasView_;
        deletion_.push(retireValue, [this, old]() { vkDestroyImageView(ctx_->device(), old, nullptr); });
    }
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = static_cast<VkFormat>(input.atlas_format);
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, std::max(input.atlas_mip_levels, 1u), 0, 1};
    MCRT_VK_CHECK(vkCreateImageView(ctx_->device(), &viewInfo, nullptr, &atlasView_));
    atlasImage_ = image;
}

void Renderer::ensureTlas(uint32_t instanceCount, uint64_t retireValue) {
    if (tlas_.handle && instanceCount <= tlasCapacity_)
        return;
    const uint32_t capacity = std::max({instanceCount, tlasCapacity_ * 2, 1024u});

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(ctx_->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                                            &capacity, &sizes);

    AccelerationStructure oldTlas = tlas_;
    Buffer oldScratch = tlasScratch_;
    deletion_.push(retireValue, [this, oldTlas, oldScratch]() mutable {
        destroyAccelerationStructure(*ctx_, oldTlas);
        ctx_->destroyBuffer(oldScratch);
    });

    tlas_ = {};
    tlas_.storage = ctx_->createBuffer(sizes.accelerationStructureSize,
                                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, MemoryKind::DeviceLocal);
    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = tlas_.storage.buffer;
    createInfo.size = sizes.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    MCRT_VK_CHECK(vkCreateAccelerationStructureKHR(ctx_->device(), &createInfo, nullptr, &tlas_.handle));
    tlasScratch_ = ctx_->createBuffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      MemoryKind::DeviceLocal,
                                      ctx_->asProperties().minAccelerationStructureScratchOffsetAlignment);
    tlasCapacity_ = capacity;
}

void Renderer::recordTlasBuild(VkCommandBuffer cmd, FrameSlot& slot, const McrtFrameInput& input,
                               uint64_t retireValue) {
    instanceScratch_.clear();
    sections_->appendInstances(instanceScratch_, input.camera_block_pos);
    if (entitiesThisFrame_)
        instanceScratch_.push_back(entities_->instance(currentSlot_));
    const uint32_t count = static_cast<uint32_t>(instanceScratch_.size());

    if (slot.instanceCapacity < std::max(count, 1u)) {
        Buffer old = slot.instances;
        deletion_.push(retireValue, [this, old]() mutable { ctx_->destroyBuffer(old); });
        slot.instanceCapacity = std::max({count, slot.instanceCapacity * 2, 1024u});
        slot.instances = ctx_->createBuffer(VkDeviceSize(slot.instanceCapacity) * sizeof(VkAccelerationStructureInstanceKHR),
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                            MemoryKind::HostUpload, 16);
    }
    if (count > 0)
        ctx_->writeBuffer(slot.instances, instanceScratch_.data(), count * sizeof(VkAccelerationStructureInstanceKHR));
    ensureTlas(count, retireValue);

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.data.deviceAddress = slot.instances.address;
    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    build.dstAccelerationStructure = tlas_.handle;
    build.scratchData.deviceAddress = tlasScratch_.address;
    VkAccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &ranges);
    tlasInstanceCount_ = count;
}

void Renderer::updateDescriptors(FrameSlot& slot) {
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas_.handle;
    auto storage = [](const Image& image) { return VkDescriptorImageInfo{VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL}; };
    VkDescriptorImageInfo noisyInfo = storage(denoiser_->noisyIllumination());
    VkDescriptorImageInfo albedoInfo = storage(denoiser_->albedoModulation());
    VkDescriptorImageInfo foregroundInfo = storage(denoiser_->foreground());
    VkDescriptorImageInfo positionInfo = storage(denoiser_->positions());
    VkDescriptorImageInfo normalInfo = storage(denoiser_->normals(historyIndex_));
    VkDescriptorBufferInfo depthInfo{depth_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo uniformInfo{slot.uniforms.buffer, 0, sizeof(FrameUniforms)};
    VkDescriptorBufferInfo sectionInfo{sections_->sectionInfoBuffer().buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo atlasInfo{VK_NULL_HANDLE, atlasView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo samplerInfo{atlasSampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorBufferInfo lightRangeInfo{sections_->lightRanges().buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo lightListInfo{sections_->lightList().buffer, 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo materialAlbedoInfo{VK_NULL_HANDLE, materials_->albedoView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo materialDataInfo{VK_NULL_HANDLE, materials_->normalView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo materialSurfaceInfo{VK_NULL_HANDLE, materials_->surfaceView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo materialSamplerInfo{materials_->sampler(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkDescriptorBufferInfo materialParamsInfo{materials_->params().buffer, 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo skyViewInfo{VK_NULL_HANDLE, skyView_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo skySamplerInfo{skySampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

    VkDescriptorImageInfo cloudViewInfo{VK_NULL_HANDLE, cloudView_.view, VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorBufferInfo reservoirPreviousInfo{reservoirs_[reservoirIndex_ ^ 1].buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo reservoirOutputInfo{reservoirs_[reservoirIndex_].buffer, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[23]{};
    auto write = [&](uint32_t binding, VkDescriptorType type) -> VkWriteDescriptorSet& {
        VkWriteDescriptorSet& w = writes[binding];
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = slot.descriptorSet;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = type;
        return w;
    };
    write(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).pNext = &asInfo;
    write(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &noisyInfo;
    write(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &albedoInfo;
    write(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &depthInfo;
    write(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).pBufferInfo = &uniformInfo;
    write(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &sectionInfo;
    write(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &atlasInfo;
    write(7, VK_DESCRIPTOR_TYPE_SAMPLER).pImageInfo = &samplerInfo;
    write(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &lightRangeInfo;
    write(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &lightListInfo;
    write(10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &foregroundInfo;
    write(11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &positionInfo;
    write(12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &normalInfo;
    write(13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &materialAlbedoInfo;
    write(14, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &materialDataInfo;
    write(15, VK_DESCRIPTOR_TYPE_SAMPLER).pImageInfo = &materialSamplerInfo;
    write(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &materialParamsInfo;
    write(17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &skyViewInfo;
    write(18, VK_DESCRIPTOR_TYPE_SAMPLER).pImageInfo = &skySamplerInfo;
    write(19, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &cloudViewInfo;
    write(20, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &materialSurfaceInfo;
    write(21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &reservoirPreviousInfo;
    write(22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &reservoirOutputInfo;
    vkUpdateDescriptorSets(ctx_->device(), 23, writes, 0, nullptr);
}

McrtStats Renderer::stats() const {
    McrtStats s{};
    s.resident_sections = static_cast<uint32_t>(sections_->residentCount());
    s.pending_sections = static_cast<uint32_t>(sections_->pendingCount());
    s.tlas_instances = tlasInstanceCount_;
    s.gpu_frame_ms = gpuFrameMs_;
    s.lights = static_cast<uint32_t>(sections_->lightCount());
    s.cpu_frame_ms = cpuFrameMs_;
    s.cpu_lights_ms = sections_->lastLightRebuildMs();
    return s;
}

void Renderer::waitForValue(uint64_t value) {
    if (value == 0)
        return;
    VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &timeline_;
    waitInfo.pValues = &value;
    VkResult result = vkWaitSemaphores(ctx_->device(), &waitInfo, kWaitTimeoutNs);
    if (result == VK_TIMEOUT)
        throw VulkanError("Timed out waiting for GPU frame " + std::to_string(value));
    MCRT_VK_CHECK(result);
}

bool Renderer::renderFrame(const McrtFrameInput& input, McrtFrameOutput& output) {
    const auto started = std::chrono::steady_clock::now();
    if (input.width == 0 || input.height == 0 || !input.color_image || !input.depth_image || !input.atlas_image)
        return false;
    for (float v : input.inv_view_proj)
        if (!std::isfinite(v))
            return false;
    if (input.color_format != VK_FORMAT_R8G8B8A8_UNORM)
        throw VulkanError("Unsupported main color format " + std::to_string(input.color_format));
    if (input.depth_format != VK_FORMAT_D32_SFLOAT)
        throw VulkanError("Unsupported main depth format " + std::to_string(input.depth_format));

    FrameSlot& slot = slots_[lastSignalValue_ % kFramesInFlight];
    waitForValue(slot.signalValue);
    uint64_t completed = 0;
    MCRT_VK_CHECK(vkGetSemaphoreCounterValue(ctx_->device(), timeline_, &completed));
    deletion_.collect(completed);

    const uint32_t slotIndex = static_cast<uint32_t>(&slot - slots_.data());
    if (slot.signalValue != 0) {
        uint64_t ticks[2];
        if (vkGetQueryPoolResults(ctx_->device(), timestamps_, slotIndex * 2, 2, sizeof(ticks), ticks, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            gpuFrameMs_ = float(double(ticks[1] - ticks[0]) * ctx_->timestampPeriodNs() * 1e-6);
    }
    vkResetQueryPool(ctx_->device(), timestamps_, slotIndex * 2, 2);

    const uint64_t retireValue = lastSignalValue_ + 1; // the value this frame signals
    // Path tracing and denoising run at the internal resolution; TAA upscales to the window.
    const float scale = std::clamp(input.render_scale > 0.0f ? input.render_scale : 1.0f, 0.5f, 1.0f);
    const uint32_t internalWidth = std::max(1u, uint32_t(std::lround(input.width * scale)));
    const uint32_t internalHeight = std::max(1u, uint32_t(std::lround(input.height * scale)));
    ensureTargets(internalWidth, internalHeight, retireValue);
    const bool upscalerReset = upscaler_->ensureTargets(input.width, input.height, retireValue);
    ensureAtlasView(input, retireValue);

    VkCommandBuffer cmd = slot.commandBuffer;
    MCRT_VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    MCRT_VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Orders us after everything already submitted or recorded on the queue (including our
    // previous frames), which is what makes the shared scratch and TLAS buffers safe to reuse.
    fullBarrier(cmd);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps_, slotIndex * 2);
    denoiser_->recordTargetInit(cmd);
    upscaler_->recordTargetInit(cmd);
    if (reservoirsNeedClear_) {
        for (Buffer& reservoirs : reservoirs_)
            vkCmdFillBuffer(cmd, reservoirs.buffer, 0, VK_WHOLE_SIZE, 0);
        memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        reservoirsNeedClear_ = false;
    }
    materials_->recordUploads(cmd, retireValue);

    sections_->recordUpdates(cmd, slotIndex, retireValue, input.camera_block_pos);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    currentSlot_ = slotIndex;
    entitiesThisFrame_ = entities_->record(cmd, slotIndex, retireValue);
    if (entitiesThisFrame_)
        memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    recordTlasBuild(cmd, slot, input, retireValue);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    historyIndex_ ^= 1;
    reservoirIndex_ ^= 1;
    FrameUniforms uniforms{};
    std::memcpy(uniforms.viewProj, input.view_proj, sizeof(uniforms.viewProj));
    std::memcpy(uniforms.invViewProj, input.inv_view_proj, sizeof(uniforms.invViewProj));
    std::memcpy(uniforms.cameraOffset, input.camera_offset, sizeof(uniforms.cameraOffset));
    std::memcpy(uniforms.sunDir, input.sun_dir, sizeof(uniforms.sunDir));
    std::memcpy(uniforms.moonDir, input.moon_dir, sizeof(uniforms.moonDir));
    std::memcpy(uniforms.skyColor, input.sky_color, sizeof(uniforms.skyColor));
    uniforms.params[0] = input.time_seconds;
    uniforms.params[1] = input.pixel_spread / scale; // per internal pixel
    uniforms.atmosphere[0] = input.cloud_height;
    uniforms.atmosphere[1] = input.render_distance > 0.0f ? input.render_distance : 1e6f;
    uniforms.atmosphere[2] = float(input.atlas_width);
    uniforms.atmosphere[3] = float(input.atlas_height);
    uniforms.params[2] = static_cast<float>(input.debug_mode);
    uniforms.params[3] = input.rain;
    uniforms.frameInfo[0] = input.frame_index;
    uniforms.frameInfo[1] = internalWidth;
    uniforms.frameInfo[2] = internalHeight;
    // Halton(2, 3) sub-pixel jitter, 8-frame cycle.
    auto halton = [](uint32_t index, uint32_t base) {
        float f = 1.0f, r = 0.0f;
        for (; index > 0; index /= base) {
            f /= float(base);
            r += f * float(index % base);
        }
        return r;
    };
    uniforms.taa[0] = halton(input.frame_index % 8 + 1, 2) - 0.5f;
    uniforms.taa[1] = halton(input.frame_index % 8 + 1, 3) - 0.5f;
    uniforms.taa[2] = scale;
    uniforms.taa[3] = hasPrevious_ ? float(sections_->lightGeneration()) : -1.0f; // ReSTIR reuse
    uniforms.frameInfo[3] = input.flags | (sections_->hasFarTerrain() ? 32u : 0u); // FLAG_FAR_TERRAIN
    for (int i = 0; i < 3; ++i)
        uniforms.cameraBlock[i] = input.camera_block_pos[i];
    if (!hasPrevious_) {
        std::memcpy(prevViewProj_, input.view_proj, sizeof(prevViewProj_));
        std::memcpy(prevCameraBlock_, input.camera_block_pos, sizeof(prevCameraBlock_));
        std::memcpy(prevCameraOffset_, input.camera_offset, sizeof(prevCameraOffset_));
        hasPrevious_ = true;
    }
    std::memcpy(uniforms.prevViewProj, prevViewProj_, sizeof(uniforms.prevViewProj));
    for (int i = 0; i < 3; ++i)
        uniforms.prevCameraShift[i] =
            float(input.camera_block_pos[i] - prevCameraBlock_[i]) - prevCameraOffset_[i];
    std::memcpy(prevViewProj_, input.view_proj, sizeof(prevViewProj_));
    std::memcpy(prevCameraBlock_, input.camera_block_pos, sizeof(prevCameraBlock_));
    std::memcpy(prevCameraOffset_, input.camera_offset, sizeof(prevCameraOffset_));
    ctx_->writeBuffer(slot.uniforms, &uniforms, sizeof(uniforms));
    updateDescriptors(slot);
    recordSkyPass(cmd, slot, slotIndex);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout_, 0, 1, &slot.descriptorSet,
                            0, nullptr);
    vkCmdTraceRaysKHR(cmd, &raygenRegion_, &missRegion_, &hitRegion_, &callableRegion_, internalWidth, internalHeight, 1);

    memoryBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    denoiser_->record(cmd, slotIndex, slot.uniforms, sizeof(FrameUniforms), historyIndex_, skyView_.view, skySampler_);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    upscaler_->record(cmd, slotIndex, slot.uniforms, sizeof(FrameUniforms), denoiser_->output().view,
                      denoiser_->positions().view, depth_, upscalerReset);

    memoryBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageCopy colorCopy{};
    colorCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    colorCopy.dstSubresource = colorCopy.srcSubresource;
    colorCopy.extent = {input.width, input.height, 1};
    vkCmdCopyImage(cmd, upscaler_->output().image, VK_IMAGE_LAYOUT_GENERAL, reinterpret_cast<VkImage>(input.color_image),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &colorCopy);
    VkBufferImageCopy depthCopy{};
    depthCopy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    depthCopy.imageExtent = {input.width, input.height, 1};
    vkCmdCopyBufferToImage(cmd, upscaler_->depth().buffer, reinterpret_cast<VkImage>(input.depth_image), VK_IMAGE_LAYOUT_GENERAL,
                           1, &depthCopy);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps_, slotIndex * 2 + 1);
    fullBarrier(cmd); // make our writes visible to whatever Minecraft records next
    MCRT_VK_CHECK(vkEndCommandBuffer(cmd));

    slot.signalValue = ++lastSignalValue_;
    cpuFrameMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();
    output.command_buffer = reinterpret_cast<uint64_t>(cmd);
    output.semaphore = reinterpret_cast<uint64_t>(timeline_);
    output.signal_value = slot.signalValue;
    return true;
}

} // namespace mcrt
