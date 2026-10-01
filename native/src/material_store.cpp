#include "material_store.h"

#include <algorithm>
#include <cstring>

namespace mcrt {

namespace {

uint32_t mipCount(uint32_t size) {
    uint32_t levels = 1;
    while (size > 1) {
        size >>= 1;
        ++levels;
    }
    return levels;
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkAccessFlags srcAccess,
                  VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                  uint32_t baseMip, uint32_t mipCount, uint32_t layer, uint32_t layerCount) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, layer, layerCount};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
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
    albedo_ = createArray(VK_FORMAT_R8G8B8A8_SRGB, 1, 1);
    data_ = createArray(VK_FORMAT_R8G8B8A8_UNORM, 1, 1);
    params_ = ctx_.createBuffer(sizeof(Params), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::HostUpload);
}

MaterialStore::~MaterialStore() {
    destroyArray(albedo_);
    destroyArray(data_);
    ctx_.destroyBuffer(params_);
    vkDestroySampler(ctx_.device(), sampler_, nullptr);
}

MaterialStore::ArrayImage MaterialStore::createArray(VkFormat format, uint32_t size, uint32_t layers) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {size, size, 1};
    info.mipLevels = mipCount(size);
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ArrayImage image;
    MCRT_VK_CHECK(vmaCreateImage(ctx_.allocator(), &info, &allocInfo, &image.image, &image.allocation, nullptr));
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, info.mipLevels, 0, layers};
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

void MaterialStore::upload(uint32_t index, uint32_t count, uint32_t size, uint32_t scale, uint32_t flags,
                           const void* albedo, const void* data) {
    const size_t bytes = size_t(size) * size * 4;
    Pending pending{index, std::vector<uint8_t>(bytes), std::vector<uint8_t>(bytes)};
    std::memcpy(pending.albedo.data(), albedo, bytes);
    std::memcpy(pending.data.data(), data, bytes);

    std::lock_guard lock(mutex_);
    if (count != requestedCount_ || size != requestedSize_) {
        requestedCount_ = count;
        requestedSize_ = size;
        paramsCpu_.assign(count, Params{1.0f, 0, float(size), 0.0f});
    }
    if (index < paramsCpu_.size())
        paramsCpu_[index] = Params{float(std::max(scale, 1u)), flags, float(size), 0.0f};
    paramsDirty_ = true;
    pending_.push_back(std::move(pending));
}

void MaterialStore::recordLayerUpload(VkCommandBuffer cmd, const ArrayImage& image, uint32_t layer,
                                      VkDeviceSize stagingOffset, VkBuffer staging) {
    VkBufferImageCopy copy{};
    copy.bufferOffset = stagingOffset;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
    copy.imageExtent = {size_, size_, 1};
    vkCmdCopyBufferToImage(cmd, staging, image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);

    // Mip chain by successive blits; each level reads the one just written.
    int32_t width = int32_t(size_);
    for (uint32_t mip = 1; mip < mipLevels_; ++mip) {
        imageBarrier(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, mip - 1, 1, layer, 1);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip - 1, layer, 1};
        blit.srcOffsets[1] = {width, width, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
        width = std::max(width / 2, 1);
        blit.dstOffsets[1] = {width, width, 1};
        vkCmdBlitImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                       VK_FILTER_LINEAR);
    }
}

void MaterialStore::recordUploads(VkCommandBuffer cmd, uint64_t retireValue) {
    std::vector<Pending> pending;
    std::vector<Params> params;
    uint32_t count, size;
    bool paramsDirty;
    {
        std::lock_guard lock(mutex_);
        pending.swap(pending_);
        params = paramsCpu_;
        count = requestedCount_;
        size = requestedSize_;
        paramsDirty = paramsDirty_;
        paramsDirty_ = false;
    }

    if (count > 0 && (count != count_ || size != size_)) {
        ArrayImage oldAlbedo = albedo_, oldData = data_;
        deletion_.push(retireValue, [this, oldAlbedo, oldData]() mutable {
            destroyArray(oldAlbedo);
            destroyArray(oldData);
        });
        count_ = count;
        size_ = size;
        mipLevels_ = mipCount(size);
        albedo_ = createArray(VK_FORMAT_R8G8B8A8_SRGB, size, count);
        data_ = createArray(VK_FORMAT_R8G8B8A8_UNORM, size, count);
        initialized_ = false;
    }
    if (!initialized_) {
        for (const ArrayImage* image : {&albedo_, &data_})
            imageBarrier(cmd, image->image, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_REMAINING_MIP_LEVELS,
                         0, VK_REMAINING_ARRAY_LAYERS);
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
    const VkDeviceSize layerBytes = VkDeviceSize(size_) * size_ * 4;
    Buffer staging = ctx_.createBuffer(layerBytes * 2 * pending.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       MemoryKind::HostUpload);
    VkDeviceSize offset = 0;
    for (const Pending& p : pending) {
        if (p.index >= count_ || p.albedo.size() != layerBytes)
            continue;
        std::memcpy(static_cast<uint8_t*>(staging.mapped) + offset, p.albedo.data(), layerBytes);
        std::memcpy(static_cast<uint8_t*>(staging.mapped) + offset + layerBytes, p.data.data(), layerBytes);
        recordLayerUpload(cmd, albedo_, p.index, offset, staging.buffer);
        recordLayerUpload(cmd, data_, p.index, offset + layerBytes, staging.buffer);
        offset += layerBytes * 2;
    }
    MCRT_VK_CHECK(vmaFlushAllocation(ctx_.allocator(), staging.allocation, 0, VK_WHOLE_SIZE));
    deletion_.push(retireValue, [this, staging]() mutable { ctx_.destroyBuffer(staging); });

    VkMemoryBarrier toShaders{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    toShaders.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShaders.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                         &toShaders, 0, nullptr, 0, nullptr);
}

} // namespace mcrt
