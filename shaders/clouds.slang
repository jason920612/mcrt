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
    if (skyType(frame) != SKY_OVERWORLD)
    {
        light.dir = float3(0.0, 1.0, 0.0);
        light.radiance = 0.0;
        return light;
    }
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

static const float CLOUD_THICKNESS = 140.0;    // blocks
static const float CLOUD_EXTINCTION = 0.07;    // per block at density 1

float cloudCoverage()
{
    return 0.3 + 0.45 * frame.params.w;
}

// Billowy noise (puffy, rounded lobes) for cumulus surfaces.
float billow(float3 p)
{
    return 1.0 - abs(valueNoise(p) * 2.0 - 1.0);
}

// Cumulus: a weather map decides where clouds form, a low-frequency shape gives each cloud its
// mass, billows round its surface and fine noise erodes its edges into wisps. Flat bases, domed
// tops (the vertical profile narrows towards the top).
float cloudDensity(float3 world, int detail)
{
    float base = frame.atmosphere.x;
    float h = (world.y - base) / CLOUD_THICKNESS;
    if (h <= 0.0 || h >= 1.0)
        return 0.0;
    float3 wind = float3(frame.params.x * 2.5, 0.0, frame.params.x * 0.9);
    float3 p = world + wind;
    float weather = valueNoise(float3(p.xz * 0.0007, 0.0)) * 0.65 + valueNoise(float3(p.xz * 0.0021, 3.0)) * 0.35;
    float coverage = saturate(cloudCoverage() + (weather - 0.5) * 0.8);
    // Clouds in thin-coverage regions stay low; dense regions tower.
    float top = lerp(0.45, 1.0, coverage);
    float profile = smoothstep(0.0, 0.07, h) * smoothstep(top, top * 0.55, h);
    float3 q = p * float3(0.0065, 0.009, 0.0065);
    float shape = fbm(q, 3);
    if (detail > 0)
        shape = shape * 0.62 + (billow(q * 2.6 + 5.0) * 0.6 + billow(q * 5.3 + 11.0) * 0.4) * 0.38;
    float d = shape * profile - (1.0 - coverage) * 0.66;
    if (detail > 1 && d > -0.05 && d < 0.12)
        d -= (valueNoise(p * 0.06) - 0.5) * 0.08; // wispy erosion at the edges
    return saturate(d * 12.0);
}

float henyeyGreenstein(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}


// How much the cloud layer above a point dims direct sunlight.
float cloudShadow(float3 pos, float3 lightDir)
{
    if (lightDir.y < 0.05 || skyType(frame) != SKY_OVERWORLD)
        return 1.0;
    float3 world = worldPosition(pos);
    float mid = frame.atmosphere.x + CLOUD_THICKNESS * 0.35;
    float t = (mid - world.y) / lightDir.y;
    if (t <= 0.0)
        return 1.0;
    float density = cloudDensity(world + lightDir * t, 0);
    return lerp(1.0, exp(-density * CLOUD_EXTINCTION * CLOUD_THICKNESS * 0.5), 0.9);
}

// Raymarches the cloud layer from a point: rgb = light scattered towards the viewer,
// a = transmittance of the background sky. jitter in [0, 1) offsets the samples (no banding).
float4 marchClouds(float3 origin, float3 dir, float3 ambient, float jitter)
{
    if (dir.y < 0.01 || skyType(frame) != SKY_OVERWORLD)
        return float4(0.0, 0.0, 0.0, 1.0);
    float3 start = worldPosition(origin);
    float base = frame.atmosphere.x;
    float t0 = max((base - start.y) / dir.y, 0.0);
    float t1 = (base + CLOUD_THICKNESS - start.y) / dir.y;
    if (t1 <= 0.0 || t0 > 12000.0)
        return float4(0.0, 0.0, 0.0, 1.0);
    t1 = min(t1, t0 + 1200.0);

    KeyLight light = keyLight();
    float cosTheta = dot(dir, light.dir);
    const int STEPS = 32;
    float dt = (t1 - t0) / STEPS;
    float transmittance = 1.0;
    float3 scattered = 0.0;
    for (int i = 0; i < STEPS && transmittance > 0.01; ++i)
    {
        float3 p = start + dir * (t0 + dt * (float(i) + jitter));
        float density = cloudDensity(p, 2);
        if (density <= 0.001)
            continue;
        // Optical depth towards the key light, with growing steps.
        float lightDepth = 0.0;
        const float stepsToLight[4] = { 6.0, 14.0, 28.0, 52.0 };
        float traveled = 0.0;
        for (int j = 0; j < 4; ++j)
        {
            float step = stepsToLight[j] - traveled;
            lightDepth += cloudDensity(p + light.dir * stepsToLight[j], j < 2 ? 1 : 0) * step;
            traveled = stepsToLight[j];
        }
        // Multiple scattering, approximated by octaves of ever weaker extinction and phase anisotropy:
        // bright, soft interiors instead of black cores, and a silver lining against the sun.
        float3 sun = 0.0;
        float a = 1.0, b = 1.0, c = 1.0;
        for (int o = 0; o < 3; ++o)
        {
            float phase = lerp(henyeyGreenstein(cosTheta, -0.2 * c), henyeyGreenstein(cosTheta, 0.75 * c), 0.65) * 4.0 * PI;
            sun += light.radiance * a * phase * exp(-lightDepth * CLOUD_EXTINCTION * b);
            a *= 0.5;
            b *= 0.35;
            c *= 0.5;
        }
        float powder = 1.0 - exp(-density * 4.0);
        float h = saturate((p.y - base) / CLOUD_THICKNESS);
        float3 sky = ambient * lerp(0.22, 0.9, h * h); // bases see the dark ground, tops the bright sky
        float sigma = density * CLOUD_EXTINCTION;
        float3 inScatter = sun * lerp(0.6, 1.0, powder) + sky;
        float stepTransmittance = exp(-sigma * dt);
        scattered += transmittance * inScatter * (1.0 - stepTransmittance);
        transmittance *= stepTransmittance;
    }
    // Distant clouds sink into the horizon haze rather than ending.
    float fade = exp(-t0 / 7000.0) * saturate(dir.y * 10.0);
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
