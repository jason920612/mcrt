#pragma once

#include "deletion_queue.h"
#include "vk_context.h"

#include <array>

namespace mcrt {

// Screen-space denoiser + compose (denoise.slang): temporal accumulation with reprojection,
// a-trous wavelet filtering, tonemapping with auto exposure. Also owns the G-buffer the path
// tracer writes into.
class Denoiser {
public:
    static constexpr uint32_t kFramesInFlight = 3;
    static constexpr uint32_t kAtrousIterations = 5;

    Denoiser(VkContext& ctx, DeletionQueue& deletion);
    ~Denoiser();
    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;

    // (Re)creates screen-sized targets. Returns true when they were recreated (history is lost).
    bool ensureTargets(uint32_t width, uint32_t height, uint64_t retireValue);
    // Layout transitions and clears for freshly created targets; no-op otherwise.
    void recordTargetInit(VkCommandBuffer cmd);

    // Records the denoise + compose passes. Expects the path tracer's G-buffer writes to be visible
    // to compute shaders. The result lands in output().
    void record(VkCommandBuffer cmd, uint32_t slot, const Buffer& frameUniforms, VkDeviceSize uniformSize,
                uint32_t historyIndex);

    // G-buffer written by the path tracer.
    const Image& noisyIllumination() const { return noisy_; }
    const Image& albedoModulation() const { return albedo_; }
    const Image& foreground() const { return foreground_; }
    const Image& positions() const { return positions_; }
    const Image& normals(uint32_t index) const { return normals_[index]; }
    const Image& output() const { return output_; }

private:
    void createPipelines();
    void destroyTargets();

    VkContext& ctx_;
    DeletionQueue& deletion_;

    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kFramesInFlight> sets_{};
    VkPipeline temporal_ = VK_NULL_HANDLE;
    VkPipeline atrous_ = VK_NULL_HANDLE;
    VkPipeline compose_ = VK_NULL_HANDLE;
    VkPipeline exposure_ = VK_NULL_HANDLE;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool targetsNeedInit_ = false;
    Image noisy_, albedo_, foreground_, positions_;
    std::array<Image, 2> normals_{};
    std::array<Image, 2> history_{};
    std::array<Image, 2> moments_{};
    Image filterA_, filterB_;
    Image output_;
    Buffer luminancePartials_;
    Buffer exposureState_;
};

} // namespace mcrt
