// Per-frame constants shared by the path tracer and the denoiser. Mirrors FrameUniforms in renderer.cpp.
//
// "Block-relative" positions are relative to the camera's block position (integer), which is the
// space the TLAS is built in. "Camera-relative" positions are relative to the exact camera position,
// which is what viewProj expects.

struct FrameUniforms
{
    float4 viewProj[4];      // columns; camera-relative, as Minecraft rasterizes
    float4 invViewProj[4];   // columns
    float4 cameraOffset;     // camera position, block-relative
    float4 sunDir;           // xyz towards the sun, w = intensity
    float4 moonDir;          // xyz towards the moon, w = intensity
    float4 skyColor;         // rgb
    float4 params;           // x = time, y = pixel spread angle, z = debug mode, w = rain
    uint4 frameInfo;         // x = frame index, y = width, z = height, w = flags (see FLAG_*)
    int4 cameraBlock;        // camera block position (absolute)
    float4 prevViewProj[4];  // last frame's viewProj (columns)
    float4 prevCameraShift;  // block-relative position + this = position relative to last frame's camera
    float4 atmosphere;       // x = cloud layer base (absolute y), zw = block atlas size in texels
};

static const uint SKY_OVERWORLD = 0;
static const uint SKY_NONE = 1;              // Nether: no sun, no clouds
static const uint SKY_END = 2;
static const uint FLAG_CAMERA_IN_WATER = 4;
static const uint FLAG_CAMERA_IN_LAVA = 8;
static const uint FLAG_CHECKERBOARD_LIGHTING = 16; // light half the pixels per frame, alternating

uint skyType(FrameUniforms f)
{
    return f.frameInfo.w & 3;
}

float4 transformColumns(float4 columns[4], float4 v)
{
    return columns[0] * v.x + columns[1] * v.y + columns[2] * v.z + columns[3] * v.w;
}

float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}
