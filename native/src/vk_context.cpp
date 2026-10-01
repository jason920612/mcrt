#define VOLK_IMPLEMENTATION
#include "vk_context.h"

#include <cstring>

namespace mcrt {

std::string vkResultName(VkResult result) {
    switch (result) {
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
    default: return "VkResult(" + std::to_string(static_cast<int>(result)) + ")";
    }
}

VkContext::VkContext(PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkInstance instance,
                     VkPhysicalDevice physicalDevice, VkDevice device, uint32_t queueFamilyIndex)
    : instance_(instance), physicalDevice_(physicalDevice), device_(device), queueFamilyIndex_(queueFamilyIndex) {
    volkInitializeCustom(getInstanceProcAddr);
    volkLoadInstanceOnly(instance_);
    volkLoadDevice(device_);
    if (!vkCmdTraceRaysKHR || !vkCreateAccelerationStructureKHR || !vkGetBufferDeviceAddress)
        throw VulkanError("Ray tracing entry points are missing; the device was created without ray tracing");

    rtProperties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    asProperties_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    rtProperties_.pNext = &asProperties_;
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &rtProperties_;
    vkGetPhysicalDeviceProperties2(physicalDevice_, &props);
    rtProperties_.pNext = nullptr;
    timestampPeriodNs_ = props.properties.limits.timestampPeriod;

    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo createInfo{};
    createInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    createInfo.vulkanApiVersion = VK_API_VERSION_1_2; // matches Minecraft's instance
    createInfo.instance = instance_;
    createInfo.physicalDevice = physicalDevice_;
    createInfo.device = device_;
    createInfo.pVulkanFunctions = &functions;
    MCRT_VK_CHECK(vmaCreateAllocator(&createInfo, &allocator_));
}

VkContext::~VkContext() {
    if (allocator_)
        vmaDestroyAllocator(allocator_);
}

Buffer VkContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind memory, VkDeviceSize alignment) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (memory == MemoryKind::HostUpload)
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    Buffer buffer;
    VmaAllocationInfo info{};
    MCRT_VK_CHECK(vmaCreateBufferWithAlignment(allocator_, &bufferInfo, &allocInfo, alignment, &buffer.buffer,
                                               &buffer.allocation, &info));
    buffer.mapped = info.pMappedData;
    buffer.size = size;
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer.buffer;
    buffer.address = vkGetBufferDeviceAddress(device_, &addressInfo);
    return buffer;
}

void VkContext::destroyBuffer(Buffer& buffer) {
    if (buffer.buffer)
        vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
    buffer = {};
}

void VkContext::writeBuffer(Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset) {
    if (!buffer.mapped)
        throw VulkanError("writeBuffer on a buffer that is not host mapped");
    std::memcpy(static_cast<char*>(buffer.mapped) + offset, data, size);
    MCRT_VK_CHECK(vmaFlushAllocation(allocator_, buffer.allocation, offset, size));
}

Image VkContext::createStorageImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags extraUsage) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | extraUsage;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    Image image;
    image.format = format;
    image.width = width;
    image.height = height;
    MCRT_VK_CHECK(vmaCreateImage(allocator_, &imageInfo, &allocInfo, &image.image, &image.allocation, nullptr));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    MCRT_VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &image.view));
    return image;
}

void VkContext::destroyImage(Image& image) {
    if (image.view)
        vkDestroyImageView(device_, image.view, nullptr);
    if (image.image)
        vmaDestroyImage(allocator_, image.image, image.allocation);
    image = {};
}

} // namespace mcrt
