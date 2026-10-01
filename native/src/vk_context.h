#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mcrt {

struct VulkanError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string vkResultName(VkResult result);

#define MCRT_VK_CHECK(expr)                                                                          \
    do {                                                                                             \
        VkResult mcrt_result_ = (expr);                                                              \
        if (mcrt_result_ != VK_SUCCESS)                                                              \
            throw ::mcrt::VulkanError(std::string(#expr " failed: ") + ::mcrt::vkResultName(mcrt_result_)); \
    } while (0)

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

enum class MemoryKind { DeviceLocal, HostUpload };

// Borrowed Vulkan objects owned by Minecraft's backend, plus our own allocator.
class VkContext {
public:
    VkContext(PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkInstance instance, VkPhysicalDevice physicalDevice,
              VkDevice device, uint32_t queueFamilyIndex);
    ~VkContext();
    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    uint32_t queueFamilyIndex() const { return queueFamilyIndex_; }
    VmaAllocator allocator() const { return allocator_; }
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProperties() const { return rtProperties_; }
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& asProperties() const { return asProperties_; }
    double timestampPeriodNs() const { return timestampPeriodNs_; }

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind memory, VkDeviceSize alignment = 0);
    void destroyBuffer(Buffer& buffer);
    // Uploads through a mapped pointer; only valid for MemoryKind::HostUpload buffers.
    void writeBuffer(Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);

    Image createStorageImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags extraUsage);
    void destroyImage(Image& image);

private:
    VkInstance instance_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    uint32_t queueFamilyIndex_;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties_{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties_{};
    double timestampPeriodNs_ = 1.0;
};

inline VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace mcrt
