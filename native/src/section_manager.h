#pragma once

#include "deletion_queue.h"
#include "vk_context.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mcrt {

struct AccelerationStructure {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    Buffer storage;
    VkDeviceAddress address = 0;
};

// Matches SectionInfo in pathtrace.slang (std430, 32 bytes).
struct SectionInfoGpu {
    VkDeviceAddress vertexAddress;
    uint32_t cutoutFirstVertex;
    uint32_t translucentFirstVertex;
    int32_t origin[3];
    uint32_t pad;
};
static_assert(sizeof(SectionInfoGpu) == 32);

// Matches GpuLight in pathtrace.slang: absolute block position, emission in bits 0-3,
// linear-ish RGB8 color in bits 8-31.
struct GpuLight {
    int32_t x, y, z;
    uint32_t packed;
};
static_assert(sizeof(GpuLight) == 16);

// Owns every chunk section's geometry on the GPU: vertex data copied from Minecraft's section
// meshes and one BLAS per section (plus one for translucent geometry).
// Updates arrive from Minecraft's compile threads and are applied on the render thread.
class SectionManager {
public:
    static constexpr uint32_t kFramesInFlight = 3;

    SectionManager(VkContext& ctx, DeletionQueue& deletion);
    ~SectionManager();
    SectionManager(const SectionManager&) = delete;
    SectionManager& operator=(const SectionManager&) = delete;

    // Thread-safe producers.
    void enqueueUpdate(int32_t x, int32_t y, int32_t z, const void* solid, uint32_t solidVertices, const void* cutout,
                       uint32_t cutoutVertices, const void* translucent, uint32_t translucentVertices,
                       const uint32_t* lights, uint32_t lightCount, const uint32_t* blockMaterials,
                       const uint8_t* occupancy);
    void enqueueRemove(int32_t x, int32_t y, int32_t z);
    void enqueueClear();
    // See mcrt_far_terrain. Stored as a pseudo-section above the build limit (kFarSectionY), as
    // alpha-tested geometry so the shader can drop the part inside render distance.
    void enqueueFarTerrain(int32_t originX, int32_t originZ, uint32_t size, uint32_t spacing, int32_t seaLevel,
                           const float* heights, const uint32_t* colors);
    bool hasFarTerrain() const;

    // Render thread. Records uploads and BLAS builds for a budgeted batch of pending sections.
    // Returns true when the visible geometry changed.
    bool recordUpdates(VkCommandBuffer cmd, uint32_t slot, uint64_t retireValue, const int32_t cameraBlock[3]);

    // Render thread. Appends TLAS instances for every resident section.
    void appendInstances(std::vector<VkAccelerationStructureInstanceKHR>& out, const int32_t cameraBlock[3]) const;

    const Buffer& sectionInfoBuffer() const { return infoBuffer_; }
    // Per section slot: uint4 (offset, count, weight as float bits, 0) into lightList(): a sample of
    // the lights of the section and its 26 neighbors; each listed light stands for `weight` lights.
    const Buffer& lightRanges() const { return lightRanges_; }
    const Buffer& lightList() const { return lightList_; }
    size_t residentCount() const { return resident_.size(); }
    size_t pendingCount() const { return pending_.size(); }
    size_t lightCount() const { return totalLights_; }
    float lastLightRebuildMs() const { return lastLightRebuildMs_; }

    // Destroys everything immediately; the caller guarantees the GPU is idle.
    void destroyAll();

private:
    enum class OpKind : uint8_t { Update, Remove, Clear };

    struct Op {
        OpKind kind = OpKind::Update;
        int32_t x = 0, y = 0, z = 0;
        uint32_t solidVertices = 0, cutoutVertices = 0, translucentVertices = 0;
        std::vector<uint8_t> data; // solid | cutout | translucent vertices
        std::vector<GpuLight> lights;
    };

    struct GpuSection {
        int32_t x = 0, y = 0, z = 0;
        std::vector<GpuLight> lights;
        Buffer vertices;
        AccelerationStructure opaque;      // geometry 0 = solid, 1 = cutout
        AccelerationStructure translucent;
        uint32_t slot = 0;
    };

    static uint64_t key(int32_t x, int32_t y, int32_t z);
    static constexpr int32_t kFarSectionY = 64; // above any build limit
    std::mutex farMutex_;
    bool farQueued_ = false;
    int32_t farX_ = 0, farZ_ = 0; // section coordinates of the current far landscape

    void retire(GpuSection& section, uint64_t retireValue);
    void retireAll(uint64_t retireValue);
    uint32_t allocateSlot(uint64_t retireValue);
    void ensureQuadIndices(uint32_t quads, uint64_t retireValue);
    void ensureStaging(uint32_t slot, VkDeviceSize bytes, uint64_t retireValue);
    void ensureScratch(VkDeviceSize bytes, uint64_t retireValue);
    void rebuildLights(uint64_t retireValue);

    VkContext& ctx_;
    DeletionQueue& deletion_;

    std::mutex incomingMutex_;
    std::vector<Op> incoming_;

    std::unordered_map<uint64_t, Op> pending_;
    std::unordered_map<uint64_t, GpuSection> resident_;

    Buffer infoBuffer_;
    uint32_t infoCapacity_ = 0;
    std::vector<uint32_t> freeSlots_;
    uint32_t nextSlot_ = 0;

    Buffer quadIndices_;
    uint32_t quadIndexCapacity_ = 0; // in quads

    std::array<Buffer, kFramesInFlight> staging_{};
    Buffer scratch_;

    // Rebuilt (into fresh buffers; old ones retire) whenever residency changes.
    Buffer lightRanges_;
    Buffer lightList_;
    bool lightsDirty_ = true;
    uint32_t framesSinceLightRebuild_ = 0;
    uint32_t lightRangesCapacity_ = 0;
    size_t totalLights_ = 0;
    float lastLightRebuildMs_ = 0.0f;
};

} // namespace mcrt
