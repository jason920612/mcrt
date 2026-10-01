// Temporal anti-aliasing and upscaling (TAAU), the last pass before the image goes to Minecraft.
//
// The path tracer jitters primary rays by a sub-pixel offset each frame and may render below
// output resolution. This pass reconstructs the current frame at output resolution, reprojects
// the previous output, clamps it to the current neighborhood (variance clipping in YCoCg) and
// blends, then tonemaps. It also upscales the depth Minecraft depth-tests its entities against.

#include "frame.slang"

struct TaaParams
{
    uint2 outputSize;
    uint historyIndex;
    uint resetHistory;
};

[[vk::push_constant]] ConstantBuffer<TaaParams> taa;

[[vk::binding(0, 0)]] Texture2D<float4> current;                              // HDR, exposure applied, internal res
[[vk::binding(1, 0)]] [format("rgba32f")] RWTexture2D<float4> positions;        // internal res
[[vk::binding(2, 0)]] Texture2D<float4> historyRead[2];                        // output res
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> historyWrite[2];
[[vk::binding(4, 0)]] [format("rgba8")] RWTexture2D<float4> outputImage;        // output res, display-ready
[[vk::binding(5, 0)]] ConstantBuffer<FrameUniforms> frame;
[[vk::binding(6, 0)]] StructuredBuffer<float> depthIn;                        // internal res
[[vk::binding(7, 0)]] RWStructuredBuffer<float> depthOut;                     // output res
[[vk::binding(8, 0)]] SamplerState linearClamp;

float3 toYCoCg(float3 c)
{
    return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25)));
}

float3 fromYCoCg(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Blend in a compressed range so bright pixels do not dominate (and flicker) at edges.
float3 compress(float3 c)
{
    return c / (1.0 + luminance(c));
}

float3 uncompress(float3 c)
{
    return c / max(1.0 - luminance(c), 1e-4);
}

float3 acesTonemap(float3 x)
{
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void temporalUpscale(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= taa.outputSize))
        return;
    uint2 inSize = frame.frameInfo.yz;
    float2 uv = (float2(id.xy) + 0.5) / float2(taa.outputSize);
    float2 jitter = frame.taa.xy;

    // Current frame at this output pixel: samples sit at internal texel centers + jitter.
    float2 currentUv = (uv * float2(inSize) - jitter) / float2(inSize);
    float3 color = compress(max(current.SampleLevel(linearClamp, currentUv, 0).rgb, 0.0));
    int2 nearest = clamp(int2(uv * float2(inSize)), int2(0, 0), int2(inSize) - 1);

    // Neighborhood statistics (YCoCg) for clipping the history.
    float3 m1 = 0.0, m2 = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            int2 q = clamp(nearest + int2(dx, dy), int2(0, 0), int2(inSize) - 1);
            float3 c = toYCoCg(compress(max(current.Load(int3(q, 0)).rgb, 0.0)));
            m1 += c;
            m2 += c * c;
        }
    m1 /= 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - m1 * m1, 0.0));
    float3 boxMin = m1 - 1.25 * sigma;
    float3 boxMax = m1 + 1.25 * sigma;

    // Where was this surface (or, for the sky, this direction) last frame?
    float4 position = positions[nearest];
    float4 prevClip;
    if (position.w > 0.0)
        prevClip = transformColumns(frame.prevViewProj, float4(position.xyz + frame.prevCameraShift.xyz, 1.0));
    else
    {
        float2 ndc = uv * 2.0 - 1.0;
        float4 a = transformColumns(frame.invViewProj, float4(ndc, 1.0, 1.0));
        float4 b = transformColumns(frame.invViewProj, float4(ndc, 0.5, 1.0));
        float3 dir = normalize(b.xyz / b.w - a.xyz / a.w);
        prevClip = transformColumns(frame.prevViewProj, float4(dir, 0.0));
    }
    float2 prevUv = prevClip.xy / prevClip.w * 0.5 + 0.5;

    uint cur = taa.historyIndex;
    float3 result = color;
    bool validHistory = taa.resetHistory == 0 && prevClip.w > 0.0 && all(prevUv >= 0.0) && all(prevUv <= 1.0);
    if (validHistory)
    {
        float3 history = toYCoCg(historyRead[1 - cur].SampleLevel(linearClamp, prevUv, 0).rgb);
        history = clamp(history, boxMin, boxMax);
        result = lerp(fromYCoCg(history), color, 0.1);
    }
    historyWrite[cur][id.xy] = float4(result, 1.0);

    float3 hdr = uncompress(result);
    float3 mapped = frame.params.z != 0.0 ? saturate(hdr) : acesTonemap(hdr);
    outputImage[id.xy] = float4(pow(mapped, 1.0 / 2.2), 1.0);
    depthOut[id.y * taa.outputSize.x + id.x] = depthIn[nearest.y * inSize.x + nearest.x];
}
