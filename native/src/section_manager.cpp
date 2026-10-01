#include "section_manager.h"

#include "mcrt/api.h"

#include <algorithm>
#include <cstring>

namespace mcrt {

namespace {

constexpr VkDeviceSize kMaxStagingBytesPerFrame = 32ull << 20;
constexpr size_t kMaxSectionsPerFrame = 512;
constexpr uint32_t kInitialSlotCapacity = 16384;
constexpr uint32_t kInitialQuadCapacity = 1u << 16;

VkAccelerationStructureGeometryKHR triangleGeometry(VkDeviceAddress vertices, uint32_t vertexCount,
                                                    VkDeviceAddress indices, bool opaque) {
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    auto& triangles = geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertices;
    triangles.vertexStride = MCRT_VERTEX_STRIDE;
    triangles.maxVertex = vertexCount > 0 ? vertexCount - 1 : 0;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indices;
    return geometry;
}

void destroyAccelerationStructure(VkContext& ctx, AccelerationStructure& as) {
    if (as.handle)
        vkDestroyAccelerationStructureKHR(ctx.device(), as.handle, nullptr);
    ctx.destroyBuffer(as.storage);
    as = {};
}

} // namespace

SectionManager::SectionManager(VkContext& ctx, DeletionQueue& deletion) : ctx_(ctx), deletion_(deletion) {
    infoCapacity_ = kInitialSlotCapacity;
    infoBuffer_ = ctx_.createBuffer(infoCapacity_ * sizeof(SectionInfoGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    MemoryKind::HostUpload);
    ensureQuadIndices(kInitialQuadCapacity, 0);
}

SectionManager::~SectionManager() {
    destroyAll();
}

void SectionManager::destroyAll() {
    for (auto& [k, section] : resident_) {
        destroyAccelerationStructure(ctx_, section.opaque);
        destroyAccelerationStructure(ctx_, section.translucent);
        ctx_.destroyBuffer(section.vertices);
    }
    resident_.clear();
    pending_.clear();
    for (Buffer& buffer : staging_)
        ctx_.destroyBuffer(buffer);
    ctx_.destroyBuffer(scratch_);
    ctx_.destroyBuffer(quadIndices_);
    ctx_.destroyBuffer(infoBuffer_);
}

uint64_t SectionManager::key(int32_t x, int32_t y, int32_t z) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x) & 0x3FFFFF) << 42) |
           (static_cast<uint64_t>(static_cast<uint32_t>(z) & 0x3FFFFF) << 20) |
           (static_cast<uint64_t>(static_cast<uint32_t>(y) & 0xFFFFF));
}

void SectionManager::enqueueUpdate(int32_t x, int32_t y, int32_t z, const void* solid, uint32_t solidVertices,
                                   const void* cutout, uint32_t cutoutVertices, const void* translucent,
                                   uint32_t translucentVertices) {
    // Only whole quads are meaningful.
    solidVertices &= ~3u;
    cutoutVertices &= ~3u;
    translucentVertices &= ~3u;

    Op op;
    op.kind = OpKind::Update;
    op.x = x;
    op.y = y;
    op.z = z;
    op.solidVertices = solid ? solidVertices : 0;
    op.cutoutVertices = cutout ? cutoutVertices : 0;
    op.translucentVertices = translucent ? translucentVertices : 0;
    op.data.resize(size_t(op.solidVertices + op.cutoutVertices + op.translucentVertices) * MCRT_VERTEX_STRIDE);
    uint8_t* dst = op.data.data();
    auto append = [&dst](const void* src, uint32_t vertices) {
        if (vertices == 0)
            return;
        std::memcpy(dst, src, size_t(vertices) * MCRT_VERTEX_STRIDE);
        dst += size_t(vertices) * MCRT_VERTEX_STRIDE;
    };
    append(solid, op.solidVertices);
    append(cutout, op.cutoutVertices);
    append(translucent, op.translucentVertices);

    std::lock_guard lock(incomingMutex_);
    incoming_.push_back(std::move(op));
}

void SectionManager::enqueueRemove(int32_t x, int32_t y, int32_t z) {
    Op op;
    op.kind = OpKind::Remove;
    op.x = x;
    op.y = y;
    op.z = z;
    std::lock_guard lock(incomingMutex_);
    incoming_.push_back(std::move(op));
}

void SectionManager::enqueueClear() {
    Op op;
    op.kind = OpKind::Clear;
    std::lock_guard lock(incomingMutex_);
    incoming_.push_back(std::move(op));
}

void SectionManager::retire(GpuSection& section, uint64_t retireValue) {
    GpuSection doomed = section;
    deletion_.push(retireValue, [this, doomed]() mutable {
        destroyAccelerationStructure(ctx_, doomed.opaque);
        destroyAccelerationStructure(ctx_, doomed.translucent);
        ctx_.destroyBuffer(doomed.vertices);
        freeSlots_.push_back(doomed.slot);
    });
}

void SectionManager::retireAll(uint64_t retireValue) {
    for (auto& [k, section] : resident_)
        retire(section, retireValue);
    resident_.clear();
}

uint32_t SectionManager::allocateSlot(uint64_t retireValue) {
    if (!freeSlots_.empty()) {
        uint32_t slot = freeSlots_.back();
        freeSlots_.pop_back();
        return slot;
    }
    if (nextSlot_ == infoCapacity_) {
        // Grow the table; in-flight frames keep reading the old buffer until it retires.
        uint32_t newCapacity = infoCapacity_ * 2;
        Buffer grown = ctx_.createBuffer(newCapacity * sizeof(SectionInfoGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         MemoryKind::HostUpload);
        ctx_.writeBuffer(grown, infoBuffer_.mapped, infoCapacity_ * sizeof(SectionInfoGpu));
        Buffer old = infoBuffer_;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
        infoBuffer_ = grown;
        infoCapacity_ = newCapacity;
    }
    return nextSlot_++;
}

void SectionManager::ensureQuadIndices(uint32_t quads, uint64_t retireValue) {
    if (quads <= quadIndexCapacity_)
        return;
    uint32_t capacity = std::max(quads, quadIndexCapacity_ * 2);
    std::vector<uint32_t> indices(size_t(capacity) * 6);
    for (uint32_t q = 0; q < capacity; ++q) {
        uint32_t* i = &indices[size_t(q) * 6];
        uint32_t base = q * 4;
        i[0] = base;
        i[1] = base + 1;
        i[2] = base + 2;
        i[3] = base + 2;
        i[4] = base + 3;
        i[5] = base;
    }
    Buffer buffer = ctx_.createBuffer(indices.size() * sizeof(uint32_t),
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                      MemoryKind::HostUpload);
    ctx_.writeBuffer(buffer, indices.data(), buffer.size);
    if (quadIndices_.buffer) {
        Buffer old = quadIndices_;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
    }
    quadIndices_ = buffer;
    quadIndexCapacity_ = capacity;
}

void SectionManager::ensureStaging(uint32_t slot, VkDeviceSize bytes, uint64_t retireValue) {
    Buffer& staging = staging_[slot];
    if (staging.size >= bytes)
        return;
    if (staging.buffer) {
        Buffer old = staging;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
    }
    staging = ctx_.createBuffer(std::max(bytes, VkDeviceSize(8) << 20), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                MemoryKind::HostUpload);
}

void SectionManager::ensureScratch(VkDeviceSize bytes, uint64_t retireValue) {
    if (scratch_.size >= bytes)
        return;
    if (scratch_.buffer) {
        Buffer old = scratch_;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
    }
    scratch_ = ctx_.createBuffer(std::max(bytes, scratch_.size * 2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 MemoryKind::DeviceLocal,
                                 ctx_.asProperties().minAccelerationStructureScratchOffsetAlignment);
}

bool SectionManager::recordUpdates(VkCommandBuffer cmd, uint32_t slot, uint64_t retireValue,
                                   const int32_t cameraBlock[3]) {
    std::vector<Op> incoming;
    {
        std::lock_guard lock(incomingMutex_);
        incoming.swap(incoming_);
    }

    bool changed = false;
    for (Op& op : incoming) {
        if (op.kind == OpKind::Clear) {
            pending_.clear();
            changed |= !resident_.empty();
            retireAll(retireValue);
            continue;
        }
        pending_[key(op.x, op.y, op.z)] = std::move(op);
    }
    if (pending_.empty())
        return changed;

    // Nearest sections first, so the area around the player fills in before the horizon.
    const int32_t cx = cameraBlock[0] >> 4, cy = cameraBlock[1] >> 4, cz = cameraBlock[2] >> 4;
    std::vector<std::pair<int64_t, uint64_t>> order;
    order.reserve(pending_.size());
    for (const auto& [k, op] : pending_) {
        int64_t dx = op.x - cx, dy = op.y - cy, dz = op.z - cz;
        order.emplace_back(dx * dx + dy * dy + dz * dz, k);
    }
    std::sort(order.begin(), order.end());

    struct Job {
        uint64_t key;
        const Op* op;
        VkDeviceSize stagingOffset;
        GpuSection section;
    };
    std::vector<Job> jobs;
    std::vector<uint64_t> processed;
    VkDeviceSize stagingTotal = 0;

    for (const auto& [distance, k] : order) {
        const Op& op = pending_[k];
        const VkDeviceSize bytes = op.data.size();
        if (op.kind == OpKind::Update && !jobs.empty() &&
            (jobs.size() >= kMaxSectionsPerFrame || stagingTotal + bytes > kMaxStagingBytesPerFrame))
            break;

        if (auto it = resident_.find(k); it != resident_.end()) {
            retire(it->second, retireValue);
            resident_.erase(it);
            changed = true;
        }
        processed.push_back(k);
        if (op.kind == OpKind::Remove || bytes == 0)
            continue;

        Job job{k, &op, stagingTotal, {}};
        job.section.x = op.x;
        job.section.y = op.y;
        job.section.z = op.z;
        jobs.push_back(job);
        stagingTotal += alignUp(bytes, 16);
    }

    if (!jobs.empty()) {
        ensureStaging(slot, stagingTotal, retireValue);
        uint32_t maxQuads = 0;
        for (const Job& job : jobs)
            maxQuads = std::max({maxQuads, job.op->solidVertices / 4, job.op->cutoutVertices / 4,
                                 job.op->translucentVertices / 4});
        ensureQuadIndices(maxQuads, retireValue);

        Buffer& staging = staging_[slot];
        for (const Job& job : jobs)
            std::memcpy(static_cast<uint8_t*>(staging.mapped) + job.stagingOffset, job.op->data.data(),
                        job.op->data.size());
        MCRT_VK_CHECK(vmaFlushAllocation(ctx_.allocator(), staging.allocation, 0, stagingTotal));

        for (Job& job : jobs) {
            job.section.vertices = ctx_.createBuffer(
                job.op->data.size(),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                MemoryKind::DeviceLocal);
            VkBufferCopy copy{job.stagingOffset, 0, job.op->data.size()};
            vkCmdCopyBuffer(cmd, staging.buffer, job.section.vertices.buffer, 1, &copy);
        }

        VkMemoryBarrier toBuild{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        toBuild.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toBuild.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &toBuild, 0, nullptr, 0, nullptr);

        struct Build {
            VkAccelerationStructureGeometryKHR geometries[2];
            VkAccelerationStructureBuildRangeInfoKHR ranges[2];
            VkAccelerationStructureBuildGeometryInfoKHR info;
            VkDeviceSize scratchOffset;
        };
        std::vector<Build> builds;
        builds.reserve(jobs.size() * 2); // pointers into builds must stay stable
        const VkDeviceSize scratchAlign = ctx_.asProperties().minAccelerationStructureScratchOffsetAlignment;
        VkDeviceSize scratchTotal = 0;

        auto addBuild = [&](AccelerationStructure& target, std::initializer_list<std::pair<VkAccelerationStructureGeometryKHR, uint32_t>> geometries) {
            Build& build = builds.emplace_back();
            uint32_t maxPrimitives[2]{};
            uint32_t count = 0;
            for (const auto& [geometry, primitives] : geometries) {
                build.geometries[count] = geometry;
                build.ranges[count] = {primitives, 0, 0, 0};
                maxPrimitives[count] = primitives;
                ++count;
            }
            build.info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            build.info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build.info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            build.info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            build.info.geometryCount = count;
            build.info.pGeometries = build.geometries;

            VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            vkGetAccelerationStructureBuildSizesKHR(ctx_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                    &build.info, maxPrimitives, &sizes);
            target.storage = ctx_.createBuffer(sizes.accelerationStructureSize,
                                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                               MemoryKind::DeviceLocal);
            VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            createInfo.buffer = target.storage.buffer;
            createInfo.size = sizes.accelerationStructureSize;
            createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            MCRT_VK_CHECK(vkCreateAccelerationStructureKHR(ctx_.device(), &createInfo, nullptr, &target.handle));
            VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            addressInfo.accelerationStructure = target.handle;
            target.address = vkGetAccelerationStructureDeviceAddressKHR(ctx_.device(), &addressInfo);
            build.info.dstAccelerationStructure = target.handle;
            build.scratchOffset = scratchTotal;
            scratchTotal += alignUp(sizes.buildScratchSize, scratchAlign);
        };

        for (Job& job : jobs) {
            const Op& op = *job.op;
            const VkDeviceAddress base = job.section.vertices.address;
            const VkDeviceAddress cutoutBase = base + VkDeviceAddress(op.solidVertices) * MCRT_VERTEX_STRIDE;
            const VkDeviceAddress translucentBase = cutoutBase + VkDeviceAddress(op.cutoutVertices) * MCRT_VERTEX_STRIDE;
            if (op.solidVertices + op.cutoutVertices > 0) {
                addBuild(job.section.opaque,
                         {{triangleGeometry(base, op.solidVertices, quadIndices_.address, true), op.solidVertices / 2},
                          {triangleGeometry(cutoutBase, op.cutoutVertices, quadIndices_.address, false),
                           op.cutoutVertices / 2}});
            }
            if (op.translucentVertices > 0) {
                addBuild(job.section.translucent,
                         {{triangleGeometry(translucentBase, op.translucentVertices, quadIndices_.address, true),
                           op.translucentVertices / 2}});
            }
        }

        ensureScratch(scratchTotal, retireValue);
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> ranges;
        infos.reserve(builds.size());
        ranges.reserve(builds.size());
        for (Build& build : builds) {
            build.info.scratchData.deviceAddress = scratch_.address + build.scratchOffset;
            infos.push_back(build.info);
            ranges.push_back(build.ranges);
        }
        if (!infos.empty())
            vkCmdBuildAccelerationStructuresKHR(cmd, static_cast<uint32_t>(infos.size()), infos.data(), ranges.data());

        for (Job& job : jobs) {
            const Op& op = *job.op;
            job.section.slot = allocateSlot(retireValue);
            SectionInfoGpu info{};
            info.vertexAddress = job.section.vertices.address;
            info.cutoutFirstVertex = op.solidVertices;
            info.translucentFirstVertex = op.solidVertices + op.cutoutVertices;
            info.origin[0] = op.x * 16;
            info.origin[1] = op.y * 16;
            info.origin[2] = op.z * 16;
            ctx_.writeBuffer(infoBuffer_, &info, sizeof(info), VkDeviceSize(job.section.slot) * sizeof(info));
            resident_[job.key] = job.section;
            changed = true;
        }
    }

    for (uint64_t k : processed)
        pending_.erase(k);
    return changed;
}

void SectionManager::appendInstances(std::vector<VkAccelerationStructureInstanceKHR>& out,
                                     const int32_t cameraBlock[3]) const {
    for (const auto& [k, section] : resident_) {
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
        instance.transform.matrix[0][3] = static_cast<float>(section.x * 16 - cameraBlock[0]);
        instance.transform.matrix[1][3] = static_cast<float>(section.y * 16 - cameraBlock[1]);
        instance.transform.matrix[2][3] = static_cast<float>(section.z * 16 - cameraBlock[2]);
        instance.instanceCustomIndex = section.slot;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        if (section.opaque.handle) {
            instance.mask = 0x01;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.accelerationStructureReference = section.opaque.address;
            out.push_back(instance);
        }
        if (section.translucent.handle) {
            instance.mask = 0x02;
            instance.instanceShaderBindingTableRecordOffset = 4;
            instance.accelerationStructureReference = section.translucent.address;
            out.push_back(instance);
        }
    }
}

} // namespace mcrt
