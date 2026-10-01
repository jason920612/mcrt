#pragma once

#include "deletion_queue.h"
#include "vk_context.h"

#include <mutex>
#include <vector>

namespace mcrt {

// PBR material textures as two 2D array images (one layer per material), plus per-material
// parameters. Uploads arrive from Java at any time and are recorded into the next frame.
class MaterialStore {
public:
    // Matches MaterialParams in pathtrace.slang.
    struct Params {
        float scale;       // blocks covered by one texture repeat
        uint32_t flags;    // 1 = tinted by the biome color
        float textureSize; // texels per side
        float pad;
    };

    MaterialStore(VkContext& ctx, DeletionQueue& deletion);
    ~MaterialStore();
    MaterialStore(const MaterialStore&) = delete;
    MaterialStore& operator=(const MaterialStore&) = delete;

    // Thread-safe. Pixels are RGBA8, size x size. count/size must match across calls.
    void upload(uint32_t index, uint32_t count, uint32_t size, uint32_t scale, uint32_t flags, const void* albedo,
                const void* data);

    // Render thread: creates images and records pending uploads + mip generation.
    void recordUploads(VkCommandBuffer cmd, uint64_t retireValue);

    VkImageView albedoView() const { return albedo_.view; }
    VkImageView dataView() const { return data_.view; }
    VkSampler sampler() const { return sampler_; }
    const Buffer& params() const { return params_; }

private:
    struct ArrayImage {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
    };
    struct Pending {
        uint32_t index;
        std::vector<uint8_t> albedo;
        std::vector<uint8_t> data;
    };

    ArrayImage createArray(VkFormat format, uint32_t size, uint32_t layers);
    void destroyArray(ArrayImage& image);
    void recordLayerUpload(VkCommandBuffer cmd, const ArrayImage& image, uint32_t layer, VkDeviceSize stagingOffset,
                           VkBuffer staging);

    VkContext& ctx_;
    DeletionQueue& deletion_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    std::mutex mutex_;
    std::vector<Pending> pending_;
    std::vector<Params> paramsCpu_;
    uint32_t requestedCount_ = 0;
    uint32_t requestedSize_ = 0;
    bool paramsDirty_ = false;

    uint32_t count_ = 0;
    uint32_t size_ = 0;
    uint32_t mipLevels_ = 1;
    bool initialized_ = false;
    ArrayImage albedo_;
    ArrayImage data_;
    Buffer params_;
};

} // namespace mcrt
