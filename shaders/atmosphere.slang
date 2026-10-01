// Physical sky: Rayleigh + Mie single scattering in an Earth-like atmosphere (meters).
// Used by sky.slang to build the sky-view LUT and by pathtrace.slang for sunlight transmittance.

#pragma once

static const float PLANET_RADIUS = 6360e3;
static const float ATMOSPHERE_RADIUS = 6460e3;
static const float3 RAYLEIGH_SCATTERING = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float RAYLEIGH_HEIGHT = 8000.0;
static const float MIE_SCATTERING = 3.996e-6;
static const float MIE_EXTINCTION = 4.44e-6;
static const float MIE_HEIGHT = 1200.0;
static const float MIE_G = 0.8;
static const float3 OZONE_ABSORPTION = float3(0.650e-6, 1.881e-6, 0.085e-6);
static const float ATMOSPHERE_PI = 3.14159265;

// Viewer: a little above sea level.
static const float3 VIEWER = float3(0.0, PLANET_RADIUS + 100.0, 0.0);

float2 raySphere(float3 origin, float3 dir, float radius)
{
    float b = dot(origin, dir);
    float c = dot(origin, origin) - radius * radius;
    float h = b * b - c;
    if (h < 0.0)
        return float2(-1.0, -1.0);
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

float3 extinctionAt(float altitude)
{
    float rayleigh = exp(-altitude / RAYLEIGH_HEIGHT);
    float mie = exp(-altitude / MIE_HEIGHT);
    float ozone = max(0.0, 1.0 - abs(altitude - 25000.0) / 15000.0);
    return RAYLEIGH_SCATTERING * rayleigh + MIE_EXTINCTION * mie + OZONE_ABSORPTION * ozone;
}

// Transmittance from a point to space along dir (0 if the planet is in the way).
float3 transmittanceToSpace(float3 origin, float3 dir)
{
    if (raySphere(origin, dir, PLANET_RADIUS).x > 0.0)
        return 0.0;
    float t = raySphere(origin, dir, ATMOSPHERE_RADIUS).y;
    const int STEPS = 12;
    float dt = t / STEPS;
    float3 opticalDepth = 0.0;
    for (int i = 0; i < STEPS; ++i)
    {
        float3 p = origin + dir * (dt * (float(i) + 0.5));
        opticalDepth += extinctionAt(length(p) - PLANET_RADIUS) * dt;
    }
    return exp(-opticalDepth);
}

float rayleighPhase(float cosTheta)
{
    return 3.0 / (16.0 * ATMOSPHERE_PI) * (1.0 + cosTheta * cosTheta);
}

float miePhase(float cosTheta)
{
    float g2 = MIE_G * MIE_G;
    return 3.0 / (8.0 * ATMOSPHERE_PI) * ((1.0 - g2) * (1.0 + cosTheta * cosTheta))
         / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * cosTheta, 1.5));
}

// Radiance scattered towards the viewer along dir for a light of unit irradiance from lightDir.
float3 skyScattering(float3 dir, float3 lightDir)
{
    float tGround = raySphere(VIEWER, dir, PLANET_RADIUS).x;
    float tEnd = raySphere(VIEWER, dir, ATMOSPHERE_RADIUS).y;
    if (tGround > 0.0)
        tEnd = tGround;
    const int STEPS = 24;
    float dt = tEnd / STEPS;
    float cosTheta = dot(dir, lightDir);
    float phaseR = rayleighPhase(cosTheta);
    float phaseM = miePhase(cosTheta);
    float3 transmittance = 1.0;
    float3 radiance = 0.0;
    for (int i = 0; i < STEPS; ++i)
    {
        float3 p = VIEWER + dir * (dt * (float(i) + 0.5));
        float altitude = length(p) - PLANET_RADIUS;
        float3 scatterR = RAYLEIGH_SCATTERING * exp(-altitude / RAYLEIGH_HEIGHT);
        float scatterM = MIE_SCATTERING * exp(-altitude / MIE_HEIGHT);
        float3 stepTransmittance = exp(-extinctionAt(altitude) * dt);
        float3 lightTransmittance = transmittanceToSpace(p, lightDir);
        float3 inScatter = (scatterR * phaseR + scatterM * phaseM) * lightTransmittance;
        // Energy-conserving integration over the step.
        radiance += transmittance * inScatter * (1.0 - stepTransmittance) / max(extinctionAt(altitude), 1e-12);
        transmittance *= stepTransmittance;
    }
    return radiance;
}

// Sky-view LUT parameterization: azimuth on u, elevation on v with extra resolution at the horizon.
float2 skyViewUv(float3 dir)
{
    float azimuth = atan2(dir.z, dir.x);
    float elevation = asin(clamp(dir.y, -1.0, 1.0));
    float v = 0.5 + 0.5 * sign(elevation) * sqrt(abs(elevation) / (ATMOSPHERE_PI * 0.5));
    return float2(azimuth / (2.0 * ATMOSPHERE_PI) + 0.5, v);
}

float3 skyViewDirection(float2 uv)
{
    float azimuth = (uv.x - 0.5) * 2.0 * ATMOSPHERE_PI;
    float s = uv.y * 2.0 - 1.0;
    float elevation = sign(s) * s * s * ATMOSPHERE_PI * 0.5;
    return float3(cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth));
}
