// Sky-view LUT: sky radiance per direction for the current sun and moon, rebuilt every frame
// (192x108 texels) and sampled by the path tracer.

#include "frame.slang"

static const float PI = 3.14159265;
[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture2D<float4> skyView;
[[vk::binding(1, 0)]] ConstantBuffer<FrameUniforms> frame;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> cloudView;
[[vk::binding(3, 0)]] Texture2D<float4> skyViewSampled;
[[vk::binding(4, 0)]] SamplerState skySampler;

#include "clouds.slang"

// Light units match the path tracer: irradiance with pi folded in, so sky radiance (in the same
// units) is scattering * true irradiance.
static const float SUN_IRRADIANCE = 3.0 * 3.14159265;
static const float MOON_IRRADIANCE = 0.12 * 3.14159265;

[shader("compute")]
[numthreads(8, 8, 1)]
void skyViewLut(uint3 id : SV_DispatchThreadID)
{
    uint2 size;
    skyView.GetDimensions(size.x, size.y);
    if (any(id.xy >= size))
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(size);
    float3 dir = skyViewDirection(uv);
    float3 radiance = 0.0;
    if (frame.sunDir.y > -0.2)
        radiance += skyScattering(dir, frame.sunDir.xyz) * SUN_IRRADIANCE;
    if (frame.moonDir.y > -0.2)
        radiance += skyScattering(dir, frame.moonDir.xyz) * MOON_IRRADIANCE;
    radiance *= 1.0 - 0.6 * frame.params.w; // overcast when raining
    skyView[id.xy] = float4(radiance, 1.0);
}

// Cloud LUT (512x256, upper hemisphere): clouds as seen from the camera, after skyViewLut.
[shader("compute")]
[numthreads(8, 8, 1)]
void cloudViewLut(uint3 id : SV_DispatchThreadID)
{
    uint2 size;
    cloudView.GetDimensions(size.x, size.y);
    if (any(id.xy >= size))
        return;
    float3 dir = cloudViewDirection((float2(id.xy) + 0.5) / float2(size));
    float3 ambient = skyViewSampled.SampleLevel(skySampler, skyViewUv(float3(0.0, 1.0, 0.0)), 0).rgb * 1.4;
    cloudView[id.xy] = marchClouds(frame.cameraOffset.xyz, dir, ambient);
}
