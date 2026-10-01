// Volumetric clouds and the key light (sun or moon), shared by the cloud LUT pass (sky.slang) and
// the path tracer (cloud shadows). Expects `frame` (FrameUniforms) and PI to be declared.

#pragma once
#include "atmosphere.slang"

struct KeyLight
{
    float3 dir;
    float3 radiance;         // irradiance on a surface facing the light
};

// Sun and moon light reaching the ground, colored by atmospheric transmittance (white at noon,
// orange at dusk).
KeyLight keyLight()
{
    KeyLight light;
    float sun = frame.sunDir.w;
    float moon = frame.moonDir.w;
    if (sun >= moon)
    {
        light.dir = frame.sunDir.xyz;
        float3 up = normalize(float3(light.dir.x, max(light.dir.y, 0.02), light.dir.z));
        light.radiance = transmittanceToSpace(VIEWER, up) * 3.0 * sun;
    }
    else
    {
        light.dir = frame.moonDir.xyz;
        light.radiance = float3(0.55, 0.65, 1.0) * 0.12 * moon;
    }
    light.radiance *= 1.0 - 0.7 * frame.params.w; // rain
    return light;
}

float hash13(float3 p)
{
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float valueNoise(float3 p)
{
    float3 i = floor(p);
    float3 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    float a = lerp(hash13(i), hash13(i + float3(1, 0, 0)), f.x);
    float b = lerp(hash13(i + float3(0, 1, 0)), hash13(i + float3(1, 1, 0)), f.x);
    float c = lerp(hash13(i + float3(0, 0, 1)), hash13(i + float3(1, 0, 1)), f.x);
    float d = lerp(hash13(i + float3(0, 1, 1)), hash13(i + float3(1, 1, 1)), f.x);
    return lerp(lerp(a, b, f.y), lerp(c, d, f.y), f.z);
}

float fbm(float3 p, int octaves)
{
    float sum = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < octaves; ++i)
    {
        sum += valueNoise(p) * amplitude;
        p = p * 2.03 + float3(17.1, 3.7, 9.2);
        amplitude *= 0.5;
    }
    return sum;
}

// World position that stays precise far from the origin (wraps every 16384 blocks).
float3 worldPosition(float3 blockRelative)
{
    int3 wrapped = frame.cameraBlock.xyz % 16384;
    return float3(wrapped) + blockRelative;
}

static const float CLOUD_THICKNESS = 90.0;     // blocks
static const float CLOUD_EXTINCTION = 0.035;   // per block at density 1

float cloudCoverage()
{
    return 0.32 + 0.45 * frame.params.w;
}

float cloudDensity(float3 world, int octaves)
{
    float base = frame.atmosphere.x;
    float h = (world.y - base) / CLOUD_THICKNESS;
    if (h <= 0.0 || h >= 1.0)
        return 0.0;
    float3 wind = float3(frame.params.x * 2.0, 0.0, frame.params.x * 0.7);
    // Large-scale weather decides where clouds may form; fbm carves the individual cumulus.
    float weather = valueNoise(float3((world.xz + wind.xz) * 0.0012, 0.0));
    float3 q = (world + wind) * float3(0.011, 0.016, 0.011);
    float noise = fbm(q, octaves);
    float shape = saturate(h * 6.0) * saturate((1.0 - h) * 2.5); // flat bottoms, rounded tops
    float coverage = saturate(cloudCoverage() + (weather - 0.5) * 0.5);
    return saturate((noise * shape - (1.0 - coverage) * 0.6) * 6.0);
}

float henyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}


// How much the cloud layer above a point dims direct sunlight.
float cloudShadow(float3 pos, float3 lightDir)
{
    if (lightDir.y < 0.05)
        return 1.0;
    float3 world = worldPosition(pos);
    float mid = frame.atmosphere.x + CLOUD_THICKNESS * 0.5;
    float t = (mid - world.y) / lightDir.y;
    if (t <= 0.0)
        return 1.0;
    float density = cloudDensity(world + lightDir * t, 1);
    return lerp(1.0, exp(-density * CLOUD_EXTINCTION * CLOUD_THICKNESS * 0.6), 0.85);
}

// Raymarches the cloud layer from a point: rgb = light scattered towards the viewer,
// a = transmittance of the background sky.
float4 marchClouds(float3 origin, float3 dir, float3 ambient)
{
    if (dir.y < 0.01)
        return float4(0.0, 0.0, 0.0, 1.0);
    float3 start = worldPosition(origin);
    float base = frame.atmosphere.x;
    float t0 = max((base - start.y) / dir.y, 0.0);
    float t1 = (base + CLOUD_THICKNESS - start.y) / dir.y;
    if (t1 <= 0.0 || t0 > 6000.0)
        return float4(0.0, 0.0, 0.0, 1.0);
    t1 = min(t1, t0 + 800.0);

    KeyLight light = keyLight();
    float cosTheta = dot(dir, light.dir);
    float phase = lerp(henyeyGreenstein(cosTheta, -0.15), henyeyGreenstein(cosTheta, 0.65), 0.6) * 4.0 * PI;
    const int STEPS = 16;
    float dt = (t1 - t0) / STEPS;
    float transmittance = 1.0;
    float3 scattered = 0.0;
    for (int i = 0; i < STEPS && transmittance > 0.02; ++i)
    {
        float3 p = start + dir * (t0 + dt * (float(i) + 0.5));
        float density = cloudDensity(p, 3);
        if (density <= 0.001)
            continue;
        // Light reaching this point through the cloud above it, towards the key light.
        float lightDepth = 0.0;
        for (int j = 1; j <= 2; ++j)
            lightDepth += cloudDensity(p + light.dir * (18.0 * j), 1) * 18.0;
        float lightTransmittance = exp(-lightDepth * CLOUD_EXTINCTION);
        // Powder term: cloud interiors darken where little light has scattered in yet.
        float powder = 1.0 - exp(-density * 6.0);
        float sigma = density * CLOUD_EXTINCTION;
        float3 inScatter = light.radiance * phase * lightTransmittance * powder + ambient * (0.4 + 0.6 * (1.0 - lightTransmittance * 0.5));
        float stepTransmittance = exp(-sigma * dt);
        scattered += transmittance * inScatter * (1.0 - stepTransmittance);
        transmittance *= stepTransmittance;
    }
    // Thin out towards the horizon, where the layer is far away and undersampled.
    float fade = exp(-t0 / 4000.0) * saturate(dir.y * 8.0);
    return float4(scattered * fade, lerp(1.0, transmittance, fade));
}


// Cloud LUT parameterization: azimuth on u, sqrt(elevation) on v (upper hemisphere only).
float2 cloudViewUv(float3 dir)
{
    float elevation = asin(clamp(dir.y, 0.0, 1.0));
    return float2(atan2(dir.z, dir.x) / (2.0 * PI) + 0.5, sqrt(elevation / (PI * 0.5)));
}

float3 cloudViewDirection(float2 uv)
{
    float azimuth = (uv.x - 0.5) * 2.0 * PI;
    float elevation = uv.y * uv.y * PI * 0.5;
    return float3(cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth));
}
