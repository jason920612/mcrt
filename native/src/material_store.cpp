#include "material_store.h"

#include "terrain_mesher.h"

#include <algorithm>
#include <cstring>

namespace mcrt {

namespace {

constexpr VkFormat kFormats[3] = {VK_FORMAT_BC3_SRGB_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK};
constexpr uint32_t kHeaderBytes = 12; // "MCM1", size, levels

// BC3 and BC5 both store 16 bytes per 4x4 block.
VkDeviceSize levelBytes(uint32_t size) {
    return VkDeviceSize(size / 4) * (size / 4) * 16;
}

VkDeviceSize chainBytes(uint32_t size, uint32_t levels) {
    VkDeviceSize total = 0;
    for (uint32_t i = 0; i < levels; ++i)
        total += levelBytes(size >> i);
    return total;
}

void toGeneral(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
}

} // namespace

MaterialStore::MaterialStore(VkContext& ctx, DeletionQueue& deletion) : ctx_(ctx), deletion_(deletion) {
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    MCRT_VK_CHECK(vkCreateSampler(ctx_.device(), &samplerInfo, nullptr, &sampler_));
    // Placeholders so descriptors are valid before (or without) any material.
    for (int i = 0; i < 3; ++i)
        images_[i] = createArray(kFormats[i], 4, 1, 1);
    params_ = ctx_.createBuffer(sizeof(Params), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::HostUpload);
}

MaterialStore::~MaterialStore() {
    for (ArrayImage& image : images_)
        destroyArray(image);
    ctx_.destroyBuffer(params_);
    vkDestroySampler(ctx_.device(), sampler_, nullptr);
}

MaterialStore::ArrayImage MaterialStore::createArray(VkFormat format, uint32_t size, uint32_t levels,
                                                     uint32_t layers) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {size, size, 1};
    info.mipLevels = levels;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ArrayImage image;
    MCRT_VK_CHECK(vmaCreateImage(ctx_.allocator(), &info, &allocInfo, &image.image, &image.allocation, nullptr));
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers};
    MCRT_VK_CHECK(vkCreateImageView(ctx_.device(), &viewInfo, nullptr, &image.view));
    return image;
}

void MaterialStore::destroyArray(ArrayImage& image) {
    if (image.view)
        vkDestroyImageView(ctx_.device(), image.view, nullptr);
    if (image.image)
        vmaDestroyImage(ctx_.allocator(), image.image, image.allocation);
    image = {};
}

void MaterialStore::upload(uint32_t index, uint32_t count, uint32_t scale, uint32_t flags, const void* file,
                           size_t bytes) {
    const auto* data = static_cast<const uint8_t*>(file);
    if (bytes < kHeaderBytes || std::memcmp(data, "MCM1", 4) != 0)
        throw VulkanError("material " + std::to_string(index) + ": not an MCM1 file");
    uint32_t size, levels;
    std::memcpy(&size, data + 4, 4);
    std::memcpy(&levels, data + 8, 4);
    if (size < 4 || (size & (size - 1)) != 0 || levels == 0 || (size >> (levels - 1)) < 4 ||
        bytes != kHeaderBytes + 3 * chainBytes(size, levels))
        throw VulkanError("material " + std::to_string(index) + ": malformed MCM1 file");

    Pending pending{index, std::vector<uint8_t>(data + kHeaderBytes, data + bytes)};
    std::lock_guard lock(mutex_);
    if (count != requestedCount_ || size != requestedSize_ || levels != requestedLevels_) {
        requestedCount_ = count;
        requestedSize_ = size;
        requestedLevels_ = levels;
        paramsCpu_.assign(count, Params{1.0f, 0, float(size), 0.0f});
    }
    if (index < paramsCpu_.size())
        paramsCpu_[index] = Params{float(std::max(scale, 1u)), flags, float(size), 0.0f};
    paramsDirty_ = true;
    pending_.push_back(std::move(pending));
    terrain::setMaterialFlags(index + 1, flags); // material ids are 1-based
}

void MaterialStore::recordUploads(VkCommandBuffer cmd, uint64_t retireValue) {
    std::vector<Pending> pending;
    std::vector<Params> params;
    uint32_t count, size, levels;
    bool paramsDirty;
    {
        std::lock_guard lock(mutex_);
        pending.swap(pending_);
        params = paramsCpu_;
        count = requestedCount_;
        size = requestedSize_;
        levels = requestedLevels_;
        paramsDirty = paramsDirty_;
        paramsDirty_ = false;
    }

    if (count > 0 && (count != count_ || size != size_ || levels != levels_)) {
        std::array<ArrayImage, 3> old = images_;
        deletion_.push(retireValue, [this, old]() mutable {
            for (ArrayImage& image : old)
                destroyArray(image);
        });
        count_ = count;
        size_ = size;
        levels_ = levels;
        for (int i = 0; i < 3; ++i)
            images_[i] = createArray(kFormats[i], size, levels, count);
        initialized_ = false;
    }
    if (!initialized_) {
        for (const ArrayImage& image : images_)
            toGeneral(cmd, image.image);
        initialized_ = true;
    }

    if (paramsDirty && !params.empty()) {
        Buffer old = params_;
        deletion_.push(retireValue, [this, old]() mutable { ctx_.destroyBuffer(old); });
        params_ = ctx_.createBuffer(params.size() * sizeof(Params), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    MemoryKind::HostUpload);
        ctx_.writeBuffer(params_, params.data(), params_.size);
    }

    if (pending.empty())
        return;
    const VkDeviceSize payloadBytes = 3 * chainBytes(size_, levels_);
    Buffer staging = ctx_.createBuffer(payloadBytes * pending.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       MemoryKind::HostUpload);
    VkDeviceSize offset = 0;
    std::vector<VkBufferImageCopy> regions[3];
    for (const Pending& p : pending) {
        if (p.index >= count_ || p.payload.size() != payloadBytes)
            continue;
        std::memcpy(static_cast<uint8_t*>(staging.mapped) + offset, p.payload.data(), payloadBytes);
        VkDeviceSize cursor = offset;
        for (int image = 0; image < 3; ++image) {
            for (uint32_t level = 0; level < levels_; ++level) {
                VkBufferImageCopy copy{};
                copy.bufferOffset = cursor;
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, p.index, 1};
                copy.imageExtent = {size_ >> level, size_ >> level, 1};
                regions[image].push_back(copy);
                cursor += levelBytes(size_ >> level);
            }
        }
        offset += payloadBytes;
    }
    MCRT_VK_CHECK(vmaFlushAllocation(ctx_.allocator(), staging.allocation, 0, VK_WHOLE_SIZE));
    for (int image = 0; image < 3; ++image)
        if (!regions[image].empty())
            vkCmdCopyBufferToImage(cmd, staging.buffer, images_[image].image, VK_IMAGE_LAYOUT_GENERAL,
                                   static_cast<uint32_t>(regions[image].size()), regions[image].data());
    deletion_.push(retireValue, [this, staging]() mutable { ctx_.destroyBuffer(staging); });

    VkMemoryBarrier toShaders{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toShaders.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShaders.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                         &toShaders, 0, nullptr, 0, nullptr);
}

} // namespace mcrt
