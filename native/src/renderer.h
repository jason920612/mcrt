#pragma once

#include "deletion_queue.h"
#include "denoiser.h"
#include "material_store.h"
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
    MaterialStore& materials() { return *materials_; }
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
    void createSkyPass();
    void recordSkyPass(VkCommandBuffer cmd, FrameSlot& slot, uint32_t slotIndex);
    void createPipeline();
    void createShaderBindingTable();
    void ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue);
    void ensureAtlasView(const McrtFrameInput& input, uint64_t retireValue);
    void ensureTlas(uint32_t instanceCount, uint64_t retireValue);
    void recordTlasBuild(VkCommandBuffer cmd, FrameSlot& slot, const McrtFrameInput& input, uint64_t retireValue);
    void updateDescriptors(FrameSlot& slot);

    void waitForValue(uint64_t value);

    std::unique_ptr<VkContext> ctx_;
    DeletionQueue deletion_;
    std::unique_ptr<SectionManager> sections_;
    std::unique_ptr<MaterialStore> materials_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::array<FrameSlot, kFramesInFlight> slots_{};
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    VkQueryPool timestamps_ = VK_NULL_HANDLE; // two per frame slot: start, end of our pass
    float gpuFrameMs_ = 0.0f;
    float cpuFrameMs_ = 0.0f;
    uint64_t lastSignalValue_ = 0;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkSampler atlasSampler_ = VK_NULL_HANDLE;

    // Sky-view and cloud LUTs (sky.slang), rebuilt every frame before tracing.
    static constexpr uint32_t kSkyWidth = 192, kSkyHeight = 108;
    static constexpr uint32_t kCloudWidth = 512, kCloudHeight = 256;
    Image skyView_;
    Image cloudView_;
    VkPipeline cloudPipeline_ = VK_NULL_HANDLE;
    bool skyViewNeedsInit_ = true;
    VkSampler skySampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout skySetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout skyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline skyPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool skyPool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kFramesInFlight> skySets_{};

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

    std::unique_ptr<Denoiser> denoiser_; // owns the G-buffer and the final image
    Buffer depth_;                       // float per pixel, copied into the D32 depth target
    uint32_t depthWidth_ = 0, depthHeight_ = 0;
    uint32_t historyIndex_ = 0;          // which denoiser history buffer is current

    VkImage atlasImage_ = VK_NULL_HANDLE;
    VkImageView atlasView_ = VK_NULL_HANDLE;

    // Previous frame's camera, for temporal reprojection.
    bool hasPrevious_ = false;
    float prevViewProj_[16] = {};
    int32_t prevCameraBlock_[3] = {};
    float prevCameraOffset_[3] = {};
};

} // namespace mcrt
