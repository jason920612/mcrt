// Denoiser and final compose, after the path tracer (SVGF-style):
//   temporalAccumulate -> atrous x N -> compose -> exposure
//
// The path tracer writes demodulated illumination (radiance / albedo of the first opaque surface),
// so texture detail never gets blurred; compose multiplies albedo back in.

#include "frame.slang"
#include "atmosphere.slang"

struct PassParams
{
    uint stepSize;           // a-trous tap spacing
    uint readFromA;          // a-trous ping-pong direction; compose: which buffer holds the result
    uint historyIndex;       // which of the two history buffers is "current"
    uint pad;
};

[[vk::push_constant]] ConstantBuffer<PassParams> pass;

[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture2D<float4> noisyIllumination;
[[vk::binding(1, 0)]] [format("rgba32f")] RWTexture2D<float4> positions;      // xyz block-relative, w = 1 on hit
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> normals[2];     // xyz normal, w = distance to camera
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> history[2];     // rgb illumination, a = history length
[[vk::binding(4, 0)]] [format("rgba16f")] RWTexture2D<float4> moments[2];     // x = E[lum], y = E[lum^2]
[[vk::binding(5, 0)]] [format("rgba16f")] RWTexture2D<float4> filterA;        // rgb illumination, a = variance
[[vk::binding(6, 0)]] [format("rgba16f")] RWTexture2D<float4> filterB;
[[vk::binding(7, 0)]] [format("rgba16f")] RWTexture2D<float4> albedoModulation;
[[vk::binding(8, 0)]] [format("rgba16f")] RWTexture2D<float4> foreground;
[[vk::binding(9, 0)]] [format("rgba16f")] RWTexture2D<float4> composed; // HDR, exposure applied (taa.slang tonemaps)
[[vk::binding(10, 0)]] ConstantBuffer<FrameUniforms> frame;
[[vk::binding(11, 0)]] RWStructuredBuffer<float> exposureState;              // [0] exposure, [1] avg luminance, [2] last time
[[vk::binding(12, 0)]] RWStructuredBuffer<float> luminancePartials;          // one log-luminance sum per compose group
[[vk::binding(13, 0)]] Sampler2D<float4> skyViewLut;                         // sky.slang

// Aerial perspective: haze that thickens with distance and thins with altitude, lit like the
// horizon sky in that direction (so it picks up the sun's glow and the time of day's colors).
static const float HAZE_DENSITY = 0.0028;     // extinction per block at sea level
static const float HAZE_SCALE_HEIGHT = 48.0;  // blocks
static const float HAZE_SEA_LEVEL = 63.0;
static const float3 HAZE_TINT = float3(0.82, 0.93, 1.12); // blue scatters more: distant land turns blue

float hazeDensityAt(float y)
{
    return exp(-(y - HAZE_SEA_LEVEL) / HAZE_SCALE_HEIGHT);
}

// View ray direction of a pixel, as the path tracer generated it (jitter included).
float3 pixelViewDirection(int2 p)
{
    float2 ndc = ((float2(p) + 0.5 + frame.taa.xy) / float2(screenSize())) * 2.0 - 1.0;
    float4 a = transformColumns(frame.invViewProj, float4(ndc, 1.0, 1.0));
    float4 b = transformColumns(frame.invViewProj, float4(ndc, 0.5, 1.0));
    float3 pa = a.xyz / a.w, pb = b.xyz / b.w;
    return dot(pa, pa) < dot(pb, pb) ? normalize(pb - pa) : normalize(pa - pb);
}

float3 applyAerialPerspective(float3 color, float3 dir, float dist)
{
    if ((frame.frameInfo.w & (FLAG_CAMERA_IN_WATER | FLAG_CAMERA_IN_LAVA)) != 0)
        return color;
    float3 rel = dir * dist;
    float3 inScatter;
    float opticalDepth;
    if (skyType(frame) == SKY_OVERWORLD)
    {
        // Exponential height fog integrated along the ray.
        float y0 = float(frame.cameraBlock.y) + frame.cameraOffset.y;
        float dy = rel.y / HAZE_SCALE_HEIGHT;
        float heightFactor = abs(dy) > 1e-3 ? (1.0 - exp(-dy)) / dy : 1.0 - 0.5 * dy;
        float density = HAZE_DENSITY * (1.0 + frame.params.w * 3.0); // rain thickens it
        opticalDepth = density * hazeDensityAt(y0) * heightFactor * dist;
        float3 lookup = normalize(float3(dir.x, max(dir.y, 0.02) * 0.5 + 0.02, dir.z));
        inScatter = skyViewLut.SampleLevel(skyViewUv(lookup), 0).rgb;
        float3 transmittance = exp(-opticalDepth * HAZE_TINT);
        color = color * transmittance + inScatter * (1.0 - transmittance);
        // Towards the render distance the land dissolves into the sky behind it, so there is no
        // edge of the world, just distance.
        float renderDistance = (frame.frameInfo.w & FLAG_FAR_TERRAIN) != 0 ? FAR_TERRAIN_RADIUS : frame.atmosphere.y;
        float edge = smoothstep(0.6 * renderDistance, 0.97 * renderDistance, length(rel.xz));
        float3 background = skyViewLut.SampleLevel(skyViewUv(skyLookupDirection(dir)), 0).rgb;
        return lerp(color, background, edge);
    }
    // Nether / End: thick dimension fog.
    float3 fog = pow(max(frame.skyColor.rgb, 0.0), 2.2) * (skyType(frame) == SKY_END ? 0.6 : 1.2);
    float fogDensity = skyType(frame) == SKY_END ? 0.004 : 0.009;
    float transmittance = exp(-fogDensity * dist);
    return color * transmittance + fog * (1.0 - transmittance);
}

static const float MAX_HISTORY = 48.0;

uint2 screenSize()
{
    return frame.frameInfo.yz;
}

bool inside(int2 p)
{
    return all(p >= 0) && all(p < int2(screenSize()));
}

// ---------------------------------------------------------------------------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void temporalAccumulate(uint3 id : SV_DispatchThreadID)
{
    int2 p = int2(id.xy);
    if (!inside(p))
        return;
    uint cur = pass.historyIndex;
    uint prev = 1 - cur;

    float4 position = positions[p];
    float4 noisy = noisyIllumination[p];
    float3 illumination = noisy.rgb;
    if (position.w == 0.0)
    {
        history[cur][p] = 0.0;
        moments[cur][p] = 0.0;
        filterA[p] = 0.0;
        return;
    }
    float3 normal = normals[cur][p].xyz;
    if (noisy.w == 0.0)
    {
        // Not lit this frame (checkerboard): reconstruct from the lit neighbors on the same surface.
        float3 sum = 0.0;
        float count = 0.0;
        const int2 offsets[4] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };
        for (int i = 0; i < 4; ++i)
        {
            int2 q = p + offsets[i];
            if (!inside(q) || positions[q].w == 0.0)
                continue;
            float4 neighbor = noisyIllumination[q];
            if (neighbor.w == 0.0 || dot(normals[cur][q].xyz, normal) < 0.9
                || abs(dot(normal, positions[q].xyz - position.xyz)) > 0.1)
                continue;
            sum += neighbor.rgb;
            count += 1.0;
        }
        illumination = count > 0.0 ? sum / count : illumination;
    }

    // Where was this surface point on screen last frame?
    float3 prevRelative = position.xyz + frame.prevCameraShift.xyz;
    float4 clip = transformColumns(frame.prevViewProj, float4(prevRelative, 1.0));
    float2 prevPixel = (clip.xy / clip.w * 0.5 + 0.5) * float2(screenSize()) - 0.5;
    float prevDistance = length(prevRelative);

    // Bilinear history fetch, keeping only taps that saw the same surface.
    int2 base = int2(floor(prevPixel));
    float2 f = prevPixel - float2(base);
    float3 historyIllumination = 0.0;
    float historyLength = 0.0;
    float2 historyMoments = 0.0;
    float weightSum = 0.0;
    if (clip.w > 0.0)
    {
        for (int i = 0; i < 4; ++i)
        {
            int2 offset = int2(i & 1, i >> 1);
            int2 q = base + offset;
            if (!inside(q))
                continue;
            float4 prevNormal = normals[prev][q];
            if (dot(prevNormal.xyz, normal) < 0.9 || abs(prevNormal.w - prevDistance) > 0.03 * prevDistance + 0.05)
                continue;
            float w = (offset.x == 1 ? f.x : 1.0 - f.x) * (offset.y == 1 ? f.y : 1.0 - f.y);
            float4 h = history[prev][q];
            historyIllumination += h.rgb * w;
            historyLength += h.a * w;
            historyMoments += moments[prev][q].xy * w;
            weightSum += w;
        }
    }
    if (weightSum > 0.01)
    {
        historyIllumination /= weightSum;
        historyLength /= weightSum;
        historyMoments /= weightSum;
    }
    else
    {
        historyLength = 0.0;
    }

    float newLength = min(historyLength + 1.0, MAX_HISTORY);
    float alpha = 1.0 / newLength;
    float lum = luminance(illumination);
    float3 accumulated = lerp(historyIllumination, illumination, alpha);
    float2 accumulatedMoments = lerp(historyMoments, float2(lum, lum * lum), max(alpha, 0.2));
    float variance = max(accumulatedMoments.y - accumulatedMoments.x * accumulatedMoments.x, 0.0);
    if (newLength < 4.0)
    {
        // Too little history for temporal moments (newly revealed pixels): estimate the variance
        // from same-surface neighbors instead, so the spatial filter knows to blur harder.
        float2 spatialMoments = 0.0;
        float spatialWeight = 0.0;
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int2 q = p + int2(dx * 2 + (dy & 1), dy); // mixes both checkerboard parities
                if (!inside(q) || positions[q].w == 0.0 || noisyIllumination[q].w == 0.0
                    || dot(normals[cur][q].xyz, normal) < 0.9)
                    continue;
                float l = luminance(noisyIllumination[q].rgb);
                spatialMoments += float2(l, l * l);
                spatialWeight += 1.0;
            }
        }
        spatialMoments /= max(spatialWeight, 1.0);
        variance = max(variance, spatialMoments.y - spatialMoments.x * spatialMoments.x) * (4.0 / newLength);
    }

    history[cur][p] = float4(accumulated, newLength);
    moments[cur][p] = float4(accumulatedMoments, 0.0, 0.0);
    filterA[p] = float4(accumulated, variance);
}

// ---------------------------------------------------------------------------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous(uint3 id : SV_DispatchThreadID)
{
    int2 p = int2(id.xy);
    if (!inside(p))
        return;
    bool fromA = pass.readFromA != 0;
    float4 center = fromA ? filterA[p] : filterB[p];
    float4 position = positions[p];
    if (position.w == 0.0)
    {
        if (fromA)
            filterB[p] = center;
        else
            filterA[p] = center;
        return;
    }

    float4 normalDistance = normals[pass.historyIndex][p];
    float3 normal = normalDistance.xyz;
    float centerLum = luminance(center.rgb);
    float sigmaLuminance = 4.0 * sqrt(max(center.a, 1e-8)) + 1e-4;
    int step = int(pass.stepSize);
    const float kernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

    float3 sum = 0.0;
    float varianceSum = 0.0;
    float weightSum = 0.0;
    for (int dy = -2; dy <= 2; ++dy)
    {
        for (int dx = -2; dx <= 2; ++dx)
        {
            int2 q = p + int2(dx, dy) * step;
            if (!inside(q))
                continue;
            float4 positionQ = positions[q];
            if (positionQ.w == 0.0)
                continue;
            float4 sample = fromA ? filterA[q] : filterB[q];
            float3 normalQ = normals[pass.historyIndex][q].xyz;

            float wNormal = pow(saturate(dot(normal, normalQ)), 64.0);
            // Distance of the neighbor from this pixel's surface plane, relative to the footprint.
            float planeDistance = abs(dot(normal, positionQ.xyz - position.xyz));
            float wPlane = exp(-planeDistance / (0.02 * normalDistance.w * float(step) / 4.0 + 0.05));
            float wLuminance = exp(-abs(centerLum - luminance(sample.rgb)) / sigmaLuminance);
            float w = kernel[abs(dx)] * kernel[abs(dy)] * wNormal * wPlane * wLuminance;

            sum += sample.rgb * w;
            varianceSum += sample.a * w * w;
            weightSum += w;
        }
    }
    float4 result = weightSum > 0.0 ? float4(sum / weightSum, varianceSum / (weightSum * weightSum)) : center;
    if (fromA)
        filterB[p] = result;
    else
        filterA[p] = result;
}

// ---------------------------------------------------------------------------------------------

float3 acesTonemap(float3 x)
{
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

groupshared float groupLogLuminance[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void compose(uint3 id : SV_DispatchThreadID, uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    int2 p = int2(id.xy);
    float logLuminance = 0.0;
    if (inside(p))
    {
        float3 illumination = pass.readFromA != 0 ? filterA[p].rgb : filterB[p].rgb;
        float3 color = foreground[p].rgb + albedoModulation[p].rgb * illumination;
        float firstSurfaceDistance = foreground[p].w;
        if (firstSurfaceDistance > 0.0 && frame.params.z == 0.0)
            color = applyAerialPerspective(color, pixelViewDirection(p), firstSurfaceDistance);
        logLuminance = log(max(luminance(color), 1e-4));

        float exposure = frame.params.z != 0.0 ? 1.0 : exposureState[0];
        composed[p] = float4(color * exposure, 1.0);
    }

    groupLogLuminance[groupIndex] = logLuminance;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (groupIndex < stride)
            groupLogLuminance[groupIndex] += groupLogLuminance[groupIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0)
    {
        uint groupsX = (screenSize().x + 15) / 16;
        luminancePartials[groupId.y * groupsX + groupId.x] = groupLogLuminance[0];
    }
}

groupshared float exposureSums[256];

// One group: averages log luminance and eases exposure towards it (used from the next frame on).
[shader("compute")]
[numthreads(256, 1, 1)]
void exposure(uint groupIndex : SV_GroupIndex)
{
    uint2 groups = (screenSize() + 15) / 16;
    uint count = groups.x * groups.y;
    float sum = 0.0;
    for (uint i = groupIndex; i < count; i += 256)
        sum += luminancePartials[i];
    exposureSums[groupIndex] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (groupIndex < stride)
            exposureSums[groupIndex] += exposureSums[groupIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0)
    {
        float averageLuminance = exp(exposureSums[0] / float(screenSize().x * screenSize().y));
        // Bounded so night stays dark and snow at noon is not crushed into grey.
        float target = clamp(0.18 / max(averageLuminance, 1e-4), 0.3, 1.5);
        float lastTime = exposureState[2];
        float now = frame.params.x;
        float dt = clamp(now - lastTime, 0.0, 0.25);
        bool uninitialized = !(exposureState[0] > 0.0) || lastTime <= 0.0;
        float current = uninitialized ? target : exposureState[0];
        // Eyes adapt faster to brightness than to darkness.
        float rate = target < current ? 3.0 : 1.2;
        exposureState[0] = lerp(current, target, 1.0 - exp(-dt * rate));
        exposureState[1] = averageLuminance;
        exposureState[2] = max(now, 1e-3);
    }
}
