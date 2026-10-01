// C ABI consumed by the Java side through FFM (dev.mcrt.rt.NativeBridge / FrameInputLayout).
// Any layout change here must be mirrored on the Java side; static_asserts below pin the offsets.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define MCRT_API __declspec(dllexport)
#else
#define MCRT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct McrtContext McrtContext;

// Minecraft's BLOCK vertex format: float3 position (section-relative), RGBA8 color, float2 uv, short2 light.
#define MCRT_VERTEX_STRIDE 28

typedef struct McrtFrameInput {
    uint64_t color_image;        // main target color, R8G8B8A8_UNORM, GENERAL layout
    uint64_t depth_image;        // main target depth, D32_SFLOAT, GENERAL layout
    uint64_t atlas_image;        // block atlas
    uint32_t color_format;
    uint32_t depth_format;
    uint32_t atlas_format;
    uint32_t atlas_mip_levels;
    uint32_t width;
    uint32_t height;
    uint32_t atlas_width;
    uint32_t atlas_height;
    uint32_t frame_index;
    uint32_t debug_mode;
    int32_t camera_block_pos[3]; // floor(camera position)
    uint32_t flags;
    float camera_offset[4];      // camera position - camera_block_pos
    float view_proj[16];         // projection * viewRotation (camera-relative, as Minecraft rasterizes), column-major
    float inv_view_proj[16];
    float sun_dir[4];            // xyz towards the sun
    float moon_dir[4];
    float sky_color[4];          // Minecraft's sky color
    float time_seconds;
    float rain;
    uint32_t reserved[2];
} McrtFrameInput;

typedef struct McrtFrameOutput {
    uint64_t command_buffer;     // VkCommandBuffer to execute in Minecraft's submission
    uint64_t semaphore;          // timeline VkSemaphore to signal after it
    uint64_t signal_value;
} McrtFrameOutput;

typedef struct McrtStats {
    uint32_t resident_sections;
    uint32_t pending_sections;
    uint32_t tlas_instances;
    uint32_t accumulated_frames;
    float gpu_frame_ms;          // GPU time of our pass, a few frames old
    uint32_t reserved[3];
} McrtStats;

// Returns null on failure; see mcrt_last_error().
MCRT_API McrtContext* mcrt_create(uint64_t get_instance_proc_addr, uint64_t instance, uint64_t physical_device,
                                  uint64_t device, uint32_t queue_family_index);

// Returns 1 when a command buffer was recorded, 0 when the frame is skipped, <0 on error.
MCRT_API int32_t mcrt_render_frame(McrtContext* ctx, const McrtFrameInput* input, McrtFrameOutput* output);

// Thread-safe. Copies the vertex data (MCRT_VERTEX_STRIDE bytes per vertex, quads) before returning.
MCRT_API void mcrt_section_update(McrtContext* ctx, int32_t section_x, int32_t section_y, int32_t section_z,
                                  const void* solid, uint32_t solid_vertices,
                                  const void* cutout, uint32_t cutout_vertices,
                                  const void* translucent, uint32_t translucent_vertices);

// Thread-safe.
MCRT_API void mcrt_section_remove(McrtContext* ctx, int32_t section_x, int32_t section_y, int32_t section_z);

// Thread-safe. Drops every section (level change, resource reload).
MCRT_API void mcrt_sections_clear(McrtContext* ctx);

// Render thread.
MCRT_API void mcrt_get_stats(McrtContext* ctx, McrtStats* stats);

MCRT_API void mcrt_destroy(McrtContext* ctx);

MCRT_API const char* mcrt_last_error(void);

#ifdef __cplusplus
}

static_assert(offsetof(McrtFrameInput, color_format) == 24);
static_assert(offsetof(McrtFrameInput, width) == 40);
static_assert(offsetof(McrtFrameInput, frame_index) == 56);
static_assert(offsetof(McrtFrameInput, camera_block_pos) == 64);
static_assert(offsetof(McrtFrameInput, flags) == 76);
static_assert(offsetof(McrtFrameInput, camera_offset) == 80);
static_assert(offsetof(McrtFrameInput, view_proj) == 96);
static_assert(offsetof(McrtFrameInput, inv_view_proj) == 160);
static_assert(offsetof(McrtFrameInput, sun_dir) == 224);
static_assert(offsetof(McrtFrameInput, moon_dir) == 240);
static_assert(offsetof(McrtFrameInput, sky_color) == 256);
static_assert(offsetof(McrtFrameInput, time_seconds) == 272);
static_assert(offsetof(McrtFrameInput, rain) == 276);
static_assert(sizeof(McrtFrameInput) == 288);
static_assert(sizeof(McrtFrameOutput) == 24);
static_assert(sizeof(McrtStats) == 32);
#endif
