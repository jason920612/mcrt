#pragma once

#include "deletion_queue.h"
#include "vk_context.h"

#include <array>

namespace mcrt {

// TAAU (taa.slang): anti-aliasing plus upscaling from the internal render resolution to the
// output resolution, final tonemap, and depth upscaling for Minecraft's depth buffer.
class TemporalUpscaler {
public:
    static constexpr uint32_t kFramesInFlight = 3;

    TemporalUpscaler(VkContext& ctx, DeletionQueue& deletion);
    ~TemporalUpscaler();
    TemporalUpscaler(const TemporalUpscaler&) = delete;
    TemporalUpscaler& operator=(const TemporalUpscaler&) = delete;

    // Output-resolution targets. Returns true when recreated (history lost).
    bool ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue);
    void recordTargetInit(VkCommandBuffer cmd);

    void record(VkCommandBuffer cmd, uint32_t slot, const Buffer& frameUniforms, VkDeviceSize uniformSize,
                VkImageView currentHdr, VkImageView positions, const Buffer& depthIn, bool resetHistory);

    const Image& output() const { return output_; }
    const Buffer& depth() const { return depth_; }

private:
    VkContext& ctx_;
    DeletionQueue& deletion_;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kFramesInFlight> sets_{};
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;

    uint32_t width_ = 0, height_ = 0;
    bool needsInit_ = false;
    uint32_t historyIndex_ = 0;
    std::array<Image, 2> history_{};
    Image output_;
    Buffer depth_;
};

} // namespace mcrt
