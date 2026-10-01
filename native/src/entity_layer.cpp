#include "entity_layer.h"

#include "mcrt/api.h"
#include "section_manager.h"

#include <algorithm>
#include <cstring>

namespace mcrt {

namespace {

// Light gray: entities have many textures, none of which the ray tracer sees; in shadows and
// reflections only their shape matters.
constexpr uint32_t kEntityColor = 0xFF9A9A9Au;

} // namespace

EntityLayer::EntityLayer(VkContext& ctx, DeletionQueue& deletion, SectionManager& sections)
    : ctx_(ctx), deletion_(deletion), sections_(sections) {
    for (Slot& slot : slots_)
        slot.sectionSlot = sections_.reserveSlot();
}

EntityLayer::~EntityLayer() {
    for (Slot& slot : slots_) {
        if (slot.blas.handle)
            vkDestroyAccelerationStructureKHR(ctx_.device(), slot.blas.handle, nullptr);
        ctx_.destroyBuffer(slot.blas.storage);
        ctx_.destroyBuffer(slot.vertices);
        ctx_.destroyBuffer(slot.scratch);
    }
}

void EntityLayer::setQuads(const float* positions, uint32_t vertexCount) {
    vertexCount = std::min(vertexCount & ~3u, kMaxQuads * 4);
    std::lock_guard lock(mutex_);
    pending_.assign(positions, positions + size_t(vertexCount) * 3);
}

bool EntityLayer::record(VkCommandBuffer cmd, uint32_t frameSlot, uint64_t retireValue) {
    Slot& slot = slots_[frameSlot];
    std::vector<float> positions;
    {
        std::lock_guard lock(mutex_);
        positions = pending_;
    }
    const uint32_t quads = uint32_t(positions.size() / 12);
    slot.quads = quads;
    if (quads == 0)
        return false;

    // Vertices in the section vertex layout (position, color, uv, light word).
    if (slot.vertexCapacity < quads) {
        Buffer old = slot.vertices;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
        slot.vertexCapacity = std::max(quads, slot.vertexCapacity * 2);
        slot.vertices = ctx_.createBuffer(VkDeviceSize(slot.vertexCapacity) * 4 * MCRT_VERTEX_STRIDE,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                          MemoryKind::HostUpload);
    }
    auto* out = static_cast<uint8_t*>(slot.vertices.mapped);
    const float uv[2] = {0.0f, 0.0f};
    const uint32_t word = kEntityVertexFlag;
    for (size_t v = 0; v < size_t(quads) * 4; ++v) {
        uint8_t* dst = out + v * MCRT_VERTEX_STRIDE;
        std::memcpy(dst, &positions[v * 3], 12);
        std::memcpy(dst + 12, &kEntityColor, 4);
        std::memcpy(dst + 16, uv, 8);
        std::memcpy(dst + 24, &word, 4);
    }
    MCRT_VK_CHECK(vmaFlushAllocation(ctx_.allocator(), slot.vertices.allocation, 0, VK_WHOLE_SIZE));

    const VkDeviceAddress indices = sections_.quadIndexAddress(quads, retireValue);
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& triangles = geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = slot.vertices.address;
    triangles.vertexStride = MCRT_VERTEX_STRIDE;
    triangles.maxVertex = quads * 4 - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indices;

    VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    const uint32_t primitives = quads * 2;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                                            &primitives, &sizes);

    if (slot.blasSize < sizes.accelerationStructureSize) {
        AccelerationStructure old = slot.blas;
        deletion_.push(retireValue, [this, old]() mutable {
            if (old.handle)
                vkDestroyAccelerationStructureKHR(ctx_.device(), old.handle, nullptr);
            ctx_.destroyBuffer(old.storage);
        });
        slot.blasSize = sizes.accelerationStructureSize * 2;
        slot.blas = {};
        slot.blas.storage = ctx_.createBuffer(slot.blasSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                              MemoryKind::DeviceLocal);
        VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.buffer = slot.blas.storage.buffer;
        createInfo.size = slot.blasSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        MCRT_VK_CHECK(vkCreateAccelerationStructureKHR(ctx_.device(), &createInfo, nullptr, &slot.blas.handle));
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        addressInfo.accelerationStructure = slot.blas.handle;
        slot.blas.address = vkGetAccelerationStructureDeviceAddressKHR(ctx_.device(), &addressInfo);
    }
    const VkDeviceSize scratchAlign = ctx_.asProperties().minAccelerationStructureScratchOffsetAlignment;
    const VkDeviceSize scratchSize = alignUp(sizes.buildScratchSize, scratchAlign) + scratchAlign;
    if (slot.scratch.size < scratchSize) {
        Buffer old = slot.scratch;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
        slot.scratch = ctx_.createBuffer(scratchSize * 2,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         MemoryKind::DeviceLocal);
    }
    build.dstAccelerationStructure = slot.blas.handle;
    build.scratchData.deviceAddress = alignUp(slot.scratch.address, scratchAlign);
    VkAccelerationStructureBuildRangeInfoKHR range{primitives, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* ranges = &range;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build, &ranges);

    sections_.writeSlotInfo(slot.sectionSlot, slot.vertices.address);
    return true;
}

VkAccelerationStructureInstanceKHR EntityLayer::instance(uint32_t frameSlot) const {
    const Slot& slot = slots_[frameSlot];
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = slot.sectionSlot;
    instance.mask = 0x04; // MASK_ENTITY: not seen by primary rays (Minecraft draws entities itself)
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = slot.blas.address;
    return instance;
}

} // namespace mcrt
