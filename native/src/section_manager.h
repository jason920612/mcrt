#pragma once

#include "deletion_queue.h"
#include "vk_context.h"

#include <array>
#include <cstdint>
#include <future>
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
    // Thread-safe and expensive (terrain meshing etc.): builds a section update without queuing it,
    // so callers can do it outside their own locks. Hand the result to commitUpdate (queue it) or
    // discardUpdate.
    void* prepareUpdate(int32_t x, int32_t y, int32_t z, const void* solid, uint32_t solidVertices, const void* cutout,
                       uint32_t cutoutVertices, const void* translucent, uint32_t translucentVertices,
                       const uint32_t* lights, uint32_t lightCount, const uint32_t* blockMaterials,
                       const uint8_t* occupancy);
    void commitUpdate(void* prepared);
    static void discardUpdate(void* prepared);
    void enqueueRemove(int32_t x, int32_t y, int32_t z);
    void enqueueClear();
    // See mcrt_far_terrain. Stored as a pseudo-section above the build limit (kFarSectionY), as
    // alpha-tested geometry so the shader can drop the part inside render distance.
    void enqueueFarTerrain(uint32_t ring, int32_t originX, int32_t originZ, uint32_t size, uint32_t spacing,
                           int32_t seaLevel, uint32_t holeHalfExtent, const float* heights, const uint32_t* colors);
    bool hasFarTerrain() const;
    // Render thread, every frame: the far landscape is opaque geometry with a hole where the real
    // world is loaded; it is re-meshed (in the background) as the camera moves.
    void updateFarHole(int32_t cameraX, int32_t cameraZ, float renderDistance);

    // A section-info slot for geometry managed elsewhere (entities); never freed.
    uint32_t reserveSlot();
    void writeSlotInfo(uint32_t slot, VkDeviceAddress vertexAddress);
    // The shared quad index buffer (0,1,2, 2,3,0 per quad), grown to at least `quads`.
    VkDeviceAddress quadIndexAddress(uint32_t quads, uint64_t retireValue);
    // Changes whenever the light lists are rebuilt (light indices and contents change).
    uint32_t lightGeneration() const { return lightGeneration_; }

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
    static constexpr uint32_t kFarRings = 2;
    static constexpr uint32_t kFarTileQuads = 48; // far rings are meshed in tiles of 48x48 quads
    struct FarRing {
        int32_t originX = 0, originZ = 0;
        uint32_t size = 0, spacing = 0, holeHalfExtent = 0;
        int32_t seaLevel = 0;
        std::vector<float> heights;
        std::vector<uint32_t> colors;
        uint32_t tilesPerSide() const { return (size - 2) / kFarTileQuads + 1; }
    };
    struct FarHole {
        float x = 0.0f, z = 0.0f, radius = 0.0f;
    };
    std::shared_ptr<const FarRing> farRings_[kFarRings];
    std::vector<std::pair<int32_t, int32_t>> farTileKeys_[kFarRings]; // section x/z of every tile
    FarHole farHole_[kFarRings];
    struct FarMeshJob {
        std::shared_ptr<const FarRing> data;
        std::future<std::vector<Op>> ops;
    };
    FarMeshJob farMeshJob_[kFarRings];
    static Op buildFarTile(uint32_t ring, const FarRing& data, FarHole hole, uint32_t tileX, uint32_t tileZ);
    // How the round hole covers a tile: 0 not at all, 1 partly, 2 entirely.
    static int farTileCoverage(const FarRing& data, FarHole hole, uint32_t tileX, uint32_t tileZ);

    void retire(GpuSection& section, uint64_t retireValue);
    void retireAll(uint64_t retireValue);
    uint32_t allocateSlot(uint64_t retireValue);
    void ensureQuadIndices(uint32_t quads, uint64_t retireValue);
    void ensureStaging(uint32_t slot, VkDeviceSize bytes, uint64_t retireValue);
    void ensureScratch(VkDeviceSize bytes, uint64_t retireValue);
    void rebuildLights(uint64_t retireValue);

    // Light lists are computed on a background thread from a snapshot of the resident sections
    // (it takes milliseconds with thousands of sections) and swapped in when ready.
    struct LightSnapshotEntry {
        int32_t x, y, z;
        uint32_t slot;
        std::vector<GpuLight> lights;
    };
    struct LightBuild {
        std::vector<uint32_t> ranges; // 4 per slot
        std::vector<GpuLight> list;
        size_t totalLights = 0;
        uint32_t capacity = 0;
        float milliseconds = 0.0f;
    };
    static LightBuild computeLights(const std::vector<LightSnapshotEntry>& sections, uint32_t capacity);
    std::vector<LightSnapshotEntry> snapshotLights() const;
    void applyLights(LightBuild&& build, uint64_t retireValue);
    std::future<LightBuild> lightJob_;

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
    uint32_t lightGeneration_ = 0;
    uint32_t framesSinceLightRebuild_ = 0;
    uint32_t lightRangesCapacity_ = 0;
    size_t totalLights_ = 0;
    float lastLightRebuildMs_ = 0.0f;
};

} // namespace mcrt
