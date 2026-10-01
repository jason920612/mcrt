#include "mcrt/api.h"

#include "renderer.h"

#include <exception>
#include <string>

namespace {

thread_local std::string g_lastError;

void setError(const char* where, const std::exception& e) {
    g_lastError = std::string(where) + ": " + e.what();
}

} // namespace

struct McrtContext {
    std::unique_ptr<mcrt::Renderer> renderer;
};

extern "C" {

MCRT_API McrtContext* mcrt_create(uint64_t get_instance_proc_addr, uint64_t instance, uint64_t physical_device,
                                  uint64_t device, uint32_t queue_family_index) {
    try {
        auto context = std::make_unique<mcrt::VkContext>(
            reinterpret_cast<PFN_vkGetInstanceProcAddr>(get_instance_proc_addr), reinterpret_cast<VkInstance>(instance),
            reinterpret_cast<VkPhysicalDevice>(physical_device), reinterpret_cast<VkDevice>(device),
            queue_family_index);
        auto* ctx = new McrtContext();
        ctx->renderer = std::make_unique<mcrt::Renderer>(std::move(context));
        return ctx;
    } catch (const std::exception& e) {
        setError("mcrt_create", e);
        return nullptr;
    }
}

MCRT_API int32_t mcrt_render_frame(McrtContext* ctx, const McrtFrameInput* input, McrtFrameOutput* output) {
    try {
        return ctx->renderer->renderFrame(*input, *output) ? 1 : 0;
    } catch (const std::exception& e) {
        setError("mcrt_render_frame", e);
        return -1;
    }
}

MCRT_API void mcrt_section_update(McrtContext* ctx, int32_t section_x, int32_t section_y, int32_t section_z,
                                  const void* solid, uint32_t solid_vertices, const void* cutout,
                                  uint32_t cutout_vertices, const void* translucent, uint32_t translucent_vertices,
                                  const uint32_t* lights, uint32_t light_count, const uint32_t* block_materials,
                                  const uint8_t* occupancy) {
    try {
        ctx->renderer->sections().enqueueUpdate(section_x, section_y, section_z, solid, solid_vertices, cutout,
                                                cutout_vertices, translucent, translucent_vertices, lights,
                                                light_count, block_materials, occupancy);
    } catch (const std::exception& e) {
        setError("mcrt_section_update", e);
    }
}

MCRT_API void mcrt_material_upload(McrtContext* ctx, uint32_t index, uint32_t count, uint32_t scale, uint32_t flags,
                                   uint64_t file, uint64_t file_bytes) {
    try {
        ctx->renderer->materials().upload(index, count, scale, flags, reinterpret_cast<const void*>(file),
                                          static_cast<size_t>(file_bytes));
    } catch (const std::exception& e) {
        setError("mcrt_material_upload", e);
    }
}

MCRT_API void mcrt_section_remove(McrtContext* ctx, int32_t section_x, int32_t section_y, int32_t section_z) {
    ctx->renderer->sections().enqueueRemove(section_x, section_y, section_z);
}

MCRT_API void mcrt_sections_clear(McrtContext* ctx) {
    ctx->renderer->sections().enqueueClear();
}

MCRT_API void mcrt_far_terrain(McrtContext* ctx, int32_t origin_x, int32_t origin_z, uint32_t size, uint32_t spacing,
                               int32_t sea_level, const float* heights, const uint32_t* colors) {
    try {
        ctx->renderer->sections().enqueueFarTerrain(origin_x, origin_z, size, spacing, sea_level, heights, colors);
    } catch (const std::exception& e) {
        setError("mcrt_far_terrain", e);
    }
}

MCRT_API void mcrt_get_stats(McrtContext* ctx, McrtStats* stats) {
    *stats = ctx->renderer->stats();
}

MCRT_API void mcrt_destroy(McrtContext* ctx) {
    delete ctx;
}

MCRT_API const char* mcrt_last_error(void) {
    return g_lastError.c_str();
}

} // extern "C"
