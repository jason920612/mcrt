#include "section_manager.h"

#include "mcrt/api.h"
#include "bevel.h"
#include "shapes.h"
#include "terrain_mesher.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

constexpr uint32_t kLightRebuildInterval = 8; // frames

uint32_t packLightColor(uint32_t emission, uint32_t rgb565) {
    uint32_t r = ((rgb565 >> 11) & 31) * 255 / 31;
    uint32_t g = ((rgb565 >> 5) & 63) * 255 / 63;
    uint32_t b = (rgb565 & 31) * 255 / 31;
    return (emission & 0xF) | (r << 8) | (g << 16) | (b << 24);
}

// Rewrites each vertex's light word (Minecraft's light map coordinates, which the path tracer has
// no use for) as: bits 12-15 emission level of the quad's block, bits 16-23 its PBR material id.
constexpr uint32_t kWaterMaterial = 255; // see SectionScanner.WATER_MATERIAL

void annotateQuads(std::vector<uint8_t>& data, const uint8_t* emission, const uint32_t* blockMaterials,
                   bool translucentLayer) {
    const size_t quads = data.size() / (size_t(MCRT_VERTEX_STRIDE) * 4);
    for (size_t q = 0; q < quads; ++q) {
        uint8_t* quad = data.data() + q * MCRT_VERTEX_STRIDE * 4;
        float normal[3];
        const int local = terrain::quadBlock(quad, normal);
        uint32_t word = 0;
        if (local >= 0) {
            const uint32_t level = emission ? emission[local] : 0;
            uint32_t material = 0;
            if (blockMaterials) {
                // Face role from the outward normal: top, bottom or side.
                const uint32_t shift = normal[1] > 0.7f ? 0 : (normal[1] < -0.7f ? 16 : 8);
                material = (blockMaterials[local] >> shift) & 0xFF;
                // Only the fluid surface itself is water; plants growing in it keep their texture.
                if (material == kWaterMaterial && !translucentLayer)
                    material = 0;
                // Atlas-textured blocks that were not smoothed (e.g. thin walls) keep their quads as is.
                if (material == terrain::kAtlasMaterial)
                    material = 0;
            }
            word = (level << 12) | (material << 16);
        }
        for (int v = 0; v < 4; ++v)
            std::memcpy(quad + v * MCRT_VERTEX_STRIDE + 24, &word, sizeof(word));
    }
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
    rebuildLights(0);
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
    ctx_.destroyBuffer(lightRanges_);
    ctx_.destroyBuffer(lightList_);
}

uint64_t SectionManager::key(int32_t x, int32_t y, int32_t z) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x) & 0x3FFFFF) << 42) |
           (static_cast<uint64_t>(static_cast<uint32_t>(z) & 0x3FFFFF) << 20) |
           (static_cast<uint64_t>(static_cast<uint32_t>(y) & 0xFFFFF));
}

void SectionManager::enqueueUpdate(int32_t x, int32_t y, int32_t z, const void* solid, uint32_t solidVertices,
                                   const void* cutout, uint32_t cutoutVertices, const void* translucent,
                                   uint32_t translucentVertices, const uint32_t* lights, uint32_t lightCount,
                                   const uint32_t* blockMaterials, const uint8_t* occupancy) {
    // Only whole quads are meaningful.
    solidVertices &= ~3u;
    cutoutVertices &= ~3u;
    translucentVertices &= ~3u;

    Op op;
    op.kind = OpKind::Update;
    op.x = x;
    op.y = y;
    op.z = z;
    auto copyLayer = [](const void* src, uint32_t vertices) {
        std::vector<uint8_t> layer;
        if (src && vertices > 0)
            layer.assign(static_cast<const uint8_t*>(src),
                         static_cast<const uint8_t*>(src) + size_t(vertices) * MCRT_VERTEX_STRIDE);
        return layer;
    };
    std::vector<uint8_t> layers[3] = {copyLayer(solid, solidVertices), copyLayer(cutout, cutoutVertices),
                                      copyLayer(translucent, translucentVertices)};

    uint8_t emission[4096] = {};
    if (lights && lightCount > 0) {
        op.lights.reserve(lightCount);
        for (uint32_t i = 0; i < lightCount; ++i) {
            const uint32_t packed = lights[i];
            const uint32_t local = packed & 0xFFF;
            const uint32_t level = (packed >> 12) & 0xF;
            emission[local] = static_cast<uint8_t>(level);
            op.lights.push_back({x * 16 + int32_t(local & 15), y * 16 + int32_t(local >> 8),
                                 z * 16 + int32_t((local >> 4) & 15), packLightColor(level, packed >> 16)});
        }
    }
    // Leaves and logs lose their cube quads; replacements are appended below.
    std::vector<shapes::BlockLook> looks;
    if (blockMaterials) {
        looks.assign(4096, {});
        for (auto& layer : layers)
            shapes::removeShapedQuads(layer, blockMaterials, looks.data());
    }
    std::vector<uint32_t> tints;
    std::vector<terrain::SpriteRect> sprites;
    if (occupancy) {
        tints.assign(4096, 0);
        sprites.assign(4096, {});
        for (auto& layer : layers)
            terrain::removeReplacedQuads(layer, occupancy, tints.data(), sprites.data());
    }
    // Remaining full cubes get beveled edges (before annotation, which reads each quad's block).
    bevel::bevelFullCubes(layers[0]);
    bevel::bevelFullCubes(layers[1]);
    for (int i = 0; i < 3; ++i)
        annotateQuads(layers[i], op.lights.empty() ? nullptr : emission, blockMaterials, i == 2);
    if (occupancy) {
        terrain::appendSmoothTerrain(layers[0], occupancy, blockMaterials, tints.data(), sprites.data());
        terrain::extendWaterUnderShore(layers[2], occupancy);
    }
    if (!looks.empty())
        shapes::appendShapes(layers[0], layers[1], x, y, z, blockMaterials, looks.data());

    op.solidVertices = static_cast<uint32_t>(layers[0].size() / MCRT_VERTEX_STRIDE);
    op.cutoutVertices = static_cast<uint32_t>(layers[1].size() / MCRT_VERTEX_STRIDE);
    op.translucentVertices = static_cast<uint32_t>(layers[2].size() / MCRT_VERTEX_STRIDE);
    op.data.reserve(layers[0].size() + layers[1].size() + layers[2].size());
    for (auto& layer : layers)
        op.data.insert(op.data.end(), layer.begin(), layer.end());

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
        job.section.lights = op.lights;
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

    lightsDirty_ |= changed;
    ++framesSinceLightRebuild_;
    // Residency changes must reach the light table, but rebuilding it every frame while chunks
    // stream in is wasted work; a grown slot table, however, must be covered immediately.
    if (lightsDirty_ && (framesSinceLightRebuild_ >= kLightRebuildInterval || lightRangesCapacity_ < infoCapacity_))
        rebuildLights(retireValue);
    return changed;
}

void SectionManager::rebuildLights(uint64_t retireValue) {
    const auto started = std::chrono::steady_clock::now();
    // Each section lists the lights of its 3x3x3 neighborhood (block light reaches 15 blocks).
    // Dense emitters (lava lakes) would make that list huge, so beyond kMaxLightsPerSection it is a
    // stratified sample whose entries each stand for count / kept lights.
    constexpr size_t kMaxLightsPerSection = 256;
    totalLights_ = 0;
    for (const auto& [k, owner] : resident_)
        totalLights_ += owner.lights.size();

    std::vector<uint32_t> ranges(size_t(infoCapacity_) * 4, 0);
    std::vector<GpuLight> list;
    if (totalLights_ > 0) {
        std::vector<const std::vector<GpuLight>*> sources;
        for (const auto& [k, section] : resident_) {
            sources.clear();
            size_t available = 0;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        auto it = resident_.find(key(section.x + dx, section.y + dy, section.z + dz));
                        if (it != resident_.end() && !it->second.lights.empty()) {
                            sources.push_back(&it->second.lights);
                            available += it->second.lights.size();
                        }
                    }
            if (available == 0)
                continue;
            const size_t kept = std::min(available, kMaxLightsPerSection);
            const size_t offset = list.size();
            // Stratified pick of `kept` lights across the concatenated neighborhood lists.
            size_t source = 0, base = 0;
            for (size_t i = 0; i < kept; ++i) {
                size_t index = (i * available) / kept + (available / kept) / 2;
                while (index >= base + sources[source]->size()) {
                    base += sources[source]->size();
                    ++source;
                }
                list.push_back((*sources[source])[index - base]);
            }
            const float weight = float(available) / float(kept);
            uint32_t weightBits;
            std::memcpy(&weightBits, &weight, sizeof(weightBits));
            uint32_t* range = &ranges[size_t(section.slot) * 4];
            range[0] = static_cast<uint32_t>(offset);
            range[1] = static_cast<uint32_t>(kept);
            range[2] = weightBits;
        }
    }
    if (list.empty())
        list.push_back({});

    Buffer newRanges = ctx_.createBuffer(ranges.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         MemoryKind::HostUpload);
    ctx_.writeBuffer(newRanges, ranges.data(), newRanges.size);
    Buffer newList = ctx_.createBuffer(list.size() * sizeof(GpuLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       MemoryKind::HostUpload);
    ctx_.writeBuffer(newList, list.data(), newList.size);

    Buffer oldRanges = lightRanges_, oldList = lightList_;
    deletion_.push(retireValue, [this, oldRanges, oldList]() mutable {
        ctx_.destroyBuffer(oldRanges);
        ctx_.destroyBuffer(oldList);
    });
    lightRanges_ = newRanges;
    lightList_ = newList;
    lightRangesCapacity_ = infoCapacity_;
    lightsDirty_ = false;
    framesSinceLightRebuild_ = 0;
    lastLightRebuildMs_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - started).count();
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
