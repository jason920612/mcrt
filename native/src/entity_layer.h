#pragma once

#include "deletion_queue.h"
#include "section_manager.h"
#include "vk_context.h"

#include <array>
#include <mutex>
#include <vector>

namespace mcrt {

// Minecraft's entities (mobs, players, items...) as ray-tracing geometry. Minecraft still draws
// them itself; this copy only lets them cast shadows, show in reflections and block bounce light.
// Rebuilt every frame from the quads Minecraft prepared for that frame (see mcrt_entities).
class EntityLayer {
public:
    static constexpr uint32_t kFramesInFlight = 3;
    static constexpr uint32_t kMaxQuads = 1u << 17;
    static constexpr uint32_t kEntityVertexFlag = 64; // vertex light word bit 6 (pathtrace.slang)

    EntityLayer(VkContext& ctx, DeletionQueue& deletion, SectionManager& sections);
    ~EntityLayer();
    EntityLayer(const EntityLayer&) = delete;
    EntityLayer& operator=(const EntityLayer&) = delete;

    // Any thread: this frame's entity quads, 4 block-relative xyz vertices each.
    void setQuads(const float* positions, uint32_t vertexCount);

    // Render thread, inside the frame's command buffer before the TLAS build. Returns whether
    // there is geometry this frame (then appendInstance adds it).
    bool record(VkCommandBuffer cmd, uint32_t frameSlot, uint64_t retireValue);
    VkAccelerationStructureInstanceKHR instance(uint32_t frameSlot) const;

private:
    struct Slot {
        Buffer vertices;
        uint32_t vertexCapacity = 0; // in quads
        AccelerationStructure blas;
        VkDeviceSize blasSize = 0;
        Buffer scratch;
        uint32_t sectionSlot = 0;
        uint32_t quads = 0;
    };

    VkContext& ctx_;
    DeletionQueue& deletion_;
    SectionManager& sections_;
    std::array<Slot, kFramesInFlight> slots_{};
    std::mutex mutex_;
    std::vector<float> pending_;
};

} // namespace mcrt
