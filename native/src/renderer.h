#pragma once

#include "deletion_queue.h"
#include "mcrt/api.h"
#include "section_manager.h"
#include "vk_context.h"

#include <array>
#include <memory>
#include <vector>

namespace mcrt {

class Renderer {
public:
    explicit Renderer(std::unique_ptr<VkContext> context);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    SectionManager& sections() { return *sections_; }
    McrtStats stats() const;

    // Records this frame's work. Returns false when there is nothing to do.
    bool renderFrame(const McrtFrameInput& input, McrtFrameOutput& output);

private:
    static constexpr uint32_t kFramesInFlight = SectionManager::kFramesInFlight;

    struct FrameSlot {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        uint64_t signalValue = 0;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        Buffer uniforms;
        Buffer instances;
        uint32_t instanceCapacity = 0;
    };

    void createDescriptors();
    void createPipeline();
    void createShaderBindingTable();
    void ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue);
    void ensureAtlasView(const McrtFrameInput& input, uint64_t retireValue);
    void ensureTlas(uint32_t instanceCount, uint64_t retireValue);
    void recordTlasBuild(VkCommandBuffer cmd, FrameSlot& slot, const McrtFrameInput& input, uint64_t retireValue);
    void updateDescriptors(FrameSlot& slot);
    bool updateAccumulation(const McrtFrameInput& input, bool geometryChanged, bool targetsChanged);
    void waitForValue(uint64_t value);

    std::unique_ptr<VkContext> ctx_;
    DeletionQueue deletion_;
    std::unique_ptr<SectionManager> sections_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::array<FrameSlot, kFramesInFlight> slots_{};
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    VkQueryPool timestamps_ = VK_NULL_HANDLE; // two per frame slot: start, end of our pass
    float gpuFrameMs_ = 0.0f;
    uint64_t lastSignalValue_ = 0;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkSampler atlasSampler_ = VK_NULL_HANDLE;

    Buffer sbt_;
    VkStridedDeviceAddressRegionKHR raygenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};

    AccelerationStructure tlas_;
    uint32_t tlasCapacity_ = 0;
    Buffer tlasScratch_;
    std::vector<VkAccelerationStructureInstanceKHR> instanceScratch_;
    uint32_t tlasInstanceCount_ = 0;

    // Render targets (our side; copied into Minecraft's main target every frame).
    Image output_;       // RGBA8, tonemapped
    Image accumulation_; // RGBA32F, linear radiance history
    Buffer depth_;       // float per pixel, copied into the D32 depth target
    bool targetsNeedInit_ = false;

    VkImage atlasImage_ = VK_NULL_HANDLE;
    VkImageView atlasView_ = VK_NULL_HANDLE;

    // Progressive accumulation while the view is static.
    uint32_t accumulatedFrames_ = 0;
    float lastViewProj_[16] = {};
    int32_t lastCameraBlock_[3] = {};
    float lastCameraOffset_[3] = {};
    float lastSunDir_[3] = {};
};

} // namespace mcrt
