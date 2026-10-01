#include "renderer.h"

#include "shaders/pathtrace.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mcrt {

namespace {

constexpr uint64_t kWaitTimeoutNs = 5'000'000'000ull;
constexpr uint32_t kMaxAccumulatedFrames = 1024;

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
};
static_assert(sizeof(FrameUniforms) == 224, "must match FrameUniforms in pathtrace.slang");

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

void toGeneralLayout(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
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
    ctx_->destroyImage(output_);
    ctx_->destroyImage(accumulation_);
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
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rgen, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rgen, nullptr},
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, rgen | hits | VK_SHADER_STAGE_MISS_BIT_KHR, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, hits, nullptr},
        {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, hits, nullptr},
        {7, VK_DESCRIPTOR_TYPE_SAMPLER, 1, hits, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(std::size(bindings));
    layoutInfo.pBindings = bindings;
    MCRT_VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout_));

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kFramesInFlight},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kFramesInFlight},
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
    if (output_.image && output_.width == width && output_.height == height)
        return;
    Image oldOutput = output_, oldAccumulation = accumulation_;
    Buffer oldDepth = depth_;
    deletion_.push(retireValue, [this, oldOutput, oldAccumulation, oldDepth]() mutable {
        ctx_->destroyImage(oldOutput);
        ctx_->destroyImage(oldAccumulation);
        ctx_->destroyBuffer(oldDepth);
    });
    output_ = ctx_->createStorageImage(width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    accumulation_ = ctx_->createStorageImage(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, 0);
    depth_ = ctx_->createBuffer(VkDeviceSize(width) * height * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                MemoryKind::DeviceLocal);
    targetsNeedInit_ = true;
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
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
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
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
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
    VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, output_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo accumulationInfo{VK_NULL_HANDLE, accumulation_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo depthInfo{depth_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo uniformInfo{slot.uniforms.buffer, 0, sizeof(FrameUniforms)};
    VkDescriptorBufferInfo sectionInfo{sections_->sectionInfoBuffer().buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo atlasInfo{VK_NULL_HANDLE, atlasView_, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo samplerInfo{atlasSampler_, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

    VkWriteDescriptorSet writes[8]{};
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
    write(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &outputInfo;
    write(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo = &accumulationInfo;
    write(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &depthInfo;
    write(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER).pBufferInfo = &uniformInfo;
    write(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).pBufferInfo = &sectionInfo;
    write(6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).pImageInfo = &atlasInfo;
    write(7, VK_DESCRIPTOR_TYPE_SAMPLER).pImageInfo = &samplerInfo;
    vkUpdateDescriptorSets(ctx_->device(), 8, writes, 0, nullptr);
}

bool Renderer::updateAccumulation(const McrtFrameInput& input, bool geometryChanged, bool targetsChanged) {
    bool same = !geometryChanged && !targetsChanged && input.debug_mode == 0 &&
                std::memcmp(lastViewProj_, input.view_proj, sizeof(lastViewProj_)) == 0 &&
                std::memcmp(lastCameraBlock_, input.camera_block_pos, sizeof(lastCameraBlock_)) == 0 &&
                std::memcmp(lastCameraOffset_, input.camera_offset, sizeof(lastCameraOffset_)) == 0;
    for (int i = 0; i < 3 && same; ++i)
        same = std::abs(lastSunDir_[i] - input.sun_dir[i]) < 1e-4f;

    std::memcpy(lastViewProj_, input.view_proj, sizeof(lastViewProj_));
    std::memcpy(lastCameraBlock_, input.camera_block_pos, sizeof(lastCameraBlock_));
    std::memcpy(lastCameraOffset_, input.camera_offset, sizeof(lastCameraOffset_));
    std::memcpy(lastSunDir_, input.sun_dir, sizeof(lastSunDir_));
    accumulatedFrames_ = same ? std::min(accumulatedFrames_ + 1, kMaxAccumulatedFrames) : 0;
    return same;
}

McrtStats Renderer::stats() const {
    McrtStats s{};
    s.resident_sections = static_cast<uint32_t>(sections_->residentCount());
    s.pending_sections = static_cast<uint32_t>(sections_->pendingCount());
    s.tlas_instances = tlasInstanceCount_;
    s.accumulated_frames = accumulatedFrames_;
    s.gpu_frame_ms = gpuFrameMs_;
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
    const bool targetsChanged = !output_.image || output_.width != input.width || output_.height != input.height ||
                                reinterpret_cast<VkImage>(input.atlas_image) != atlasImage_;
    ensureTargets(input.width, input.height, retireValue);
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
    if (targetsNeedInit_) {
        toGeneralLayout(cmd, output_.image);
        toGeneralLayout(cmd, accumulation_.image);
        targetsNeedInit_ = false;
    }

    const bool geometryChanged = sections_->recordUpdates(cmd, slotIndex, retireValue, input.camera_block_pos);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    recordTlasBuild(cmd, slot, input, retireValue);
    memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    updateAccumulation(input, geometryChanged, targetsChanged);
    FrameUniforms uniforms{};
    std::memcpy(uniforms.viewProj, input.view_proj, sizeof(uniforms.viewProj));
    std::memcpy(uniforms.invViewProj, input.inv_view_proj, sizeof(uniforms.invViewProj));
    std::memcpy(uniforms.cameraOffset, input.camera_offset, sizeof(uniforms.cameraOffset));
    std::memcpy(uniforms.sunDir, input.sun_dir, sizeof(uniforms.sunDir));
    std::memcpy(uniforms.moonDir, input.moon_dir, sizeof(uniforms.moonDir));
    std::memcpy(uniforms.skyColor, input.sky_color, sizeof(uniforms.skyColor));
    uniforms.params[0] = input.time_seconds;
    uniforms.params[1] = static_cast<float>(accumulatedFrames_);
    uniforms.params[2] = static_cast<float>(input.debug_mode);
    uniforms.params[3] = input.rain;
    uniforms.frameInfo[0] = input.frame_index;
    uniforms.frameInfo[1] = input.width;
    uniforms.frameInfo[2] = input.height;
    ctx_->writeBuffer(slot.uniforms, &uniforms, sizeof(uniforms));
    updateDescriptors(slot);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout_, 0, 1, &slot.descriptorSet,
                            0, nullptr);
    vkCmdTraceRaysKHR(cmd, &raygenRegion_, &missRegion_, &hitRegion_, &callableRegion_, input.width, input.height, 1);

    memoryBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageCopy colorCopy{};
    colorCopy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    colorCopy.dstSubresource = colorCopy.srcSubresource;
    colorCopy.extent = {input.width, input.height, 1};
    vkCmdCopyImage(cmd, output_.image, VK_IMAGE_LAYOUT_GENERAL, reinterpret_cast<VkImage>(input.color_image),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &colorCopy);
    VkBufferImageCopy depthCopy{};
    depthCopy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    depthCopy.imageExtent = {input.width, input.height, 1};
    vkCmdCopyBufferToImage(cmd, depth_.buffer, reinterpret_cast<VkImage>(input.depth_image), VK_IMAGE_LAYOUT_GENERAL,
                           1, &depthCopy);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps_, slotIndex * 2 + 1);
    fullBarrier(cmd); // make our writes visible to whatever Minecraft records next
    MCRT_VK_CHECK(vkEndCommandBuffer(cmd));

    slot.signalValue = ++lastSignalValue_;
    output.command_buffer = reinterpret_cast<uint64_t>(cmd);
    output.semaphore = reinterpret_cast<uint64_t>(timeline_);
    output.signal_value = slot.signalValue;
    return true;
}

} // namespace mcrt
