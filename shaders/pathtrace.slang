// M1 path tracer over Minecraft's own section meshes.
//
// Space: everything is relative to the camera's block position (integer), so section instance
// transforms are exact and floats stay small. The camera itself sits at frame.cameraOffset.
//
// Hit groups (sbtRecordStride = 2 ray types):
//   0/1 solid primary/shadow, 2/3 cutout primary/shadow (alpha tested), 4/5 translucent primary/shadow.

static const uint RAY_PRIMARY = 0;
static const uint RAY_SHADOW = 1;
static const uint RAY_TYPE_COUNT = 2;

static const uint MASK_OPAQUE = 0x01;      // solid + cutout instances
static const uint MASK_TRANSLUCENT = 0x02;

static const float PI = 3.14159265;

struct FrameUniforms
{
    float4 viewProj[4];      // columns; camera-relative, as Minecraft rasterizes
    float4 invViewProj[4];   // columns
    float4 cameraOffset;     // camera position relative to camera block position
    float4 sunDir;           // xyz towards the sun, w = intensity
    float4 moonDir;          // xyz towards the moon, w = intensity
    float4 skyColor;         // rgb
    float4 params;           // x = time, y = accumulated frame count, z = debug mode, w = rain
    uint4 frameInfo;         // x = frame index, y = width, z = height
};

// Minecraft's BLOCK vertex format (28 bytes).
struct BlockVertex
{
    float3 position;         // relative to the section origin
    uint color;              // RGBA8: biome tint * baked shade/AO
    float2 uv;               // block atlas
    uint light;              // short2: block light * 16, sky light * 16
};

struct SectionInfo
{
    uint64_t vertexAddress;
    uint cutoutFirstVertex;
    uint translucentFirstVertex;
    int3 origin;             // section block origin relative to the camera block position
    uint pad;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> outputImage;
[[vk::binding(2, 0)]] [format("rgba32f")] RWTexture2D<float4> accumulation;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> depthOutput;
[[vk::binding(4, 0)]] ConstantBuffer<FrameUniforms> frame;
[[vk::binding(5, 0)]] StructuredBuffer<SectionInfo> sections;
[[vk::binding(6, 0)]] Texture2D<float4> blockAtlas;
[[vk::binding(7, 0)]] SamplerState atlasSampler;

struct Payload
{
    float3 albedo;
    float hitT;              // < 0 on miss
    float3 normal;
    float alpha;
    float blockLight;        // 0..1
    uint translucent;
};

struct ShadowPayload
{
    uint visible;
};

// ---------------------------------------------------------------------------------------------
// Geometry access

struct SurfaceSample
{
    float3 albedo;
    float alpha;
    float3 normal;
    float blockLight;
};

float4 unpackColor(uint c)
{
    return float4(c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, c >> 24) / 255.0;
}

SurfaceSample fetchSurface(BuiltInTriangleIntersectionAttributes attribs, uint firstVertexKind)
{
    SectionInfo info = sections[InstanceID()];
    uint first = 0;
    if (firstVertexKind == 1)
        first = info.cutoutFirstVertex;
    else if (firstVertexKind == 2)
        first = info.translucentFirstVertex;

    // Triangles come from quads: (0,1,2) and (2,3,0), matching the shared index buffer.
    uint prim = PrimitiveIndex();
    uint base = first + (prim >> 1) * 4;
    uint3 idx = (prim & 1) == 0 ? uint3(base, base + 1, base + 2) : uint3(base + 2, base + 3, base);

    BlockVertex* vertices = (BlockVertex*)info.vertexAddress;
    BlockVertex v0 = vertices[idx.x];
    BlockVertex v1 = vertices[idx.y];
    BlockVertex v2 = vertices[idx.z];

    float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    float2 uv = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;
    float4 tint = unpackColor(v0.color) * bary.x + unpackColor(v1.color) * bary.y + unpackColor(v2.color) * bary.z;
    float4 tex = blockAtlas.SampleLevel(atlasSampler, uv, 0);

    float3 n = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    if (dot(n, WorldRayDirection()) > 0.0)
        n = -n;

    float block0 = float(v0.light & 0xFFFF);
    float block1 = float(v1.light & 0xFFFF);
    float block2 = float(v2.light & 0xFFFF);

    SurfaceSample s;
    s.albedo = pow(tex.rgb * tint.rgb, 2.2); // atlas and tint are sRGB-encoded
    s.alpha = tex.a * tint.a;
    s.normal = n;
    s.blockLight = saturate((block0 * bary.x + block1 * bary.y + block2 * bary.z) / 240.0);
    return s;
}

float alphaAt(BuiltInTriangleIntersectionAttributes attribs)
{
    return fetchSurface(attribs, 1).alpha;
}

void writePayload(inout Payload p, SurfaceSample s, uint translucent)
{
    p.albedo = s.albedo;
    p.hitT = RayTCurrent();
    p.normal = s.normal;
    p.alpha = s.alpha;
    p.blockLight = s.blockLight;
    p.translucent = translucent;
}

[shader("closesthit")]
void closestHitSolid(inout Payload p, BuiltInTriangleIntersectionAttributes attribs)
{
    writePayload(p, fetchSurface(attribs, GeometryIndex() == 0 ? 0 : 1), 0);
}

[shader("closesthit")]
void closestHitTranslucent(inout Payload p, BuiltInTriangleIntersectionAttributes attribs)
{
    writePayload(p, fetchSurface(attribs, 2), 1);
}

[shader("anyhit")]
void anyHitCutoutPrimary(inout Payload p, BuiltInTriangleIntersectionAttributes attribs)
{
    if (alphaAt(attribs) < 0.5)
        IgnoreHit();
}

[shader("anyhit")]
void anyHitCutoutShadow(inout ShadowPayload p, BuiltInTriangleIntersectionAttributes attribs)
{
    if (alphaAt(attribs) < 0.5)
        IgnoreHit();
}

[shader("miss")]
void primaryMiss(inout Payload p)
{
    p.hitT = -1.0;
}

[shader("miss")]
void shadowMiss(inout ShadowPayload p)
{
    p.visible = 1;
}

// ---------------------------------------------------------------------------------------------
// Lighting

uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(inout uint seed)
{
    seed = pcgHash(seed);
    return float(seed) / 4294967296.0;
}

float3 cosineSampleHemisphere(float3 n, inout uint seed)
{
    float r1 = random(seed);
    float r2 = random(seed);
    float phi = 2.0 * PI * r1;
    float r = sqrt(r2);
    float3 t = abs(n.y) < 0.999 ? normalize(cross(n, float3(0, 1, 0))) : float3(1, 0, 0);
    float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(1.0 - r2));
}

struct KeyLight
{
    float3 dir;
    float3 radiance;         // irradiance on a surface facing the light
};

KeyLight keyLight()
{
    KeyLight light;
    float sun = frame.sunDir.w;
    float moon = frame.moonDir.w;
    if (sun >= moon)
    {
        light.dir = frame.sunDir.xyz;
        light.radiance = float3(1.0, 0.92, 0.80) * 3.0 * sun;
    }
    else
    {
        light.dir = frame.moonDir.xyz;
        light.radiance = float3(0.55, 0.65, 1.0) * 0.12 * moon;
    }
    light.radiance *= 1.0 - 0.7 * frame.params.w; // rain
    return light;
}

float3 skyRadiance(float3 d, bool includeSunDisk)
{
    float day = saturate(frame.sunDir.w);
    float3 zenith = pow(max(frame.skyColor.rgb, 0.0), 2.2);
    float3 horizon = lerp(zenith, float3(0.75, 0.80, 0.88) * max(day, 0.05), 0.65);
    float3 color = d.y >= 0.0 ? lerp(horizon, zenith, pow(saturate(d.y), 0.45)) : horizon * 0.35;

    // Warm glow around the sun near the horizon.
    float sunAmount = saturate(dot(d, frame.sunDir.xyz));
    color += float3(1.0, 0.55, 0.25) * pow(sunAmount, 8.0) * 0.6 * day * (1.0 - saturate(frame.sunDir.y * 2.0));

    if (includeSunDisk && sunAmount > 0.99989)
        color += float3(1.0, 0.92, 0.80) * 60.0 * day;
    return color * (1.0 - 0.5 * frame.params.w);
}

bool traceShadow(float3 origin, float3 dir)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = 1024.0;
    ShadowPayload shadow;
    shadow.visible = 0;
    TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             MASK_OPAQUE, RAY_SHADOW, RAY_TYPE_COUNT, 1, ray, shadow);
    return shadow.visible != 0;
}

// Stand-in for real emitters until ReSTIR (M2): Minecraft's baked block light lights the surface.
float3 blockLightEmission(float blockLight)
{
    return float3(1.0, 0.72, 0.42) * pow(blockLight, 3.0) * 1.6;
}

// Direct light at a surface point (no indirect).
float3 directLight(float3 pos, float3 n, float3 albedo, float blockLight, KeyLight light)
{
    float3 result = albedo * blockLightEmission(blockLight);
    float ndotl = dot(n, light.dir);
    if (ndotl > 0.0 && any(light.radiance > 0.0) && traceShadow(pos + n * 0.002, light.dir))
        result += albedo * light.radiance * ndotl;
    return result;
}

Payload tracePrimary(float3 origin, float3 dir, uint mask, float tMin)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = tMin;
    ray.TMax = 4096.0;
    Payload p;
    p.hitT = -1.0;
    TraceRay(scene, RAY_FLAG_NONE, mask, RAY_PRIMARY, RAY_TYPE_COUNT, 0, ray, p);
    return p;
}

// Full shading of an opaque hit: direct light + one diffuse bounce.
float3 shadeOpaque(float3 pos, float3 dir, Payload hit, KeyLight light, inout uint seed)
{
    float3 radiance = directLight(pos, hit.normal, hit.albedo, hit.blockLight, light);

    float3 bounceDir = cosineSampleHemisphere(hit.normal, seed);
    float3 bounceOrigin = pos + hit.normal * 0.002;
    Payload bounce = tracePrimary(bounceOrigin, bounceDir, MASK_OPAQUE, 0.0);
    float3 incoming;
    if (bounce.hitT < 0.0)
        incoming = skyRadiance(bounceDir, false);
    else
        incoming = directLight(bounceOrigin + bounceDir * bounce.hitT, bounce.normal, bounce.albedo, bounce.blockLight, light);
    // Cosine-weighted sampling: pdf cancels the cosine and 1/pi of the Lambertian BRDF.
    radiance += hit.albedo * incoming;
    return radiance;
}

float3 unproject(float2 ndc, float z)
{
    float4 p = frame.invViewProj[0] * ndc.x + frame.invViewProj[1] * ndc.y + frame.invViewProj[2] * z + frame.invViewProj[3];
    return p.xyz / p.w;
}

float projectDepth(float3 cameraRelative)
{
    float4 clip = frame.viewProj[0] * cameraRelative.x + frame.viewProj[1] * cameraRelative.y
                + frame.viewProj[2] * cameraRelative.z + frame.viewProj[3];
    return saturate(clip.z / clip.w);
}

float3 acesTonemap(float3 x)
{
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 size = DispatchRaysDimensions().xy;
    uint seed = pcgHash(pixel.x + pixel.y * size.x + frame.frameInfo.x * size.x * size.y);

    // Minecraft's image memory is bottom-up, so NDC y follows the row index directly.
    float2 jitter = frame.params.y > 0.0 ? float2(random(seed), random(seed)) : float2(0.5, 0.5);
    float2 ndc = ((float2(pixel) + jitter) / float2(size)) * 2.0 - 1.0;
    float3 a = unproject(ndc, 1.0);
    float3 b = unproject(ndc, 0.5);
    float3 nearPoint = dot(a, a) < dot(b, b) ? a : b;
    float3 farPoint = dot(a, a) < dot(b, b) ? b : a;
    float3 dir = normalize(farPoint - nearPoint);
    float3 origin = frame.cameraOffset.xyz + nearPoint;

    KeyLight light = keyLight();
    float3 radiance = 0.0;
    float3 throughput = 1.0;
    float depth = 0.0; // Minecraft clears depth to 0 (reverse Z)
    uint debugMode = uint(frame.params.z);

    float tMin = 0.0;
    for (uint layer = 0; layer < 4; ++layer)
    {
        Payload hit = tracePrimary(origin, dir, 0xFF, tMin);
        if (hit.hitT < 0.0)
        {
            radiance += throughput * skyRadiance(dir, true);
            break;
        }

        float3 pos = origin + dir * hit.hitT;
        if (debugMode == 1)
        {
            radiance = hit.albedo;
            break;
        }
        if (debugMode == 2)
        {
            radiance = hit.normal * 0.5 + 0.5;
            break;
        }

        if (hit.translucent != 0)
        {
            // Lit translucent surface (water, stained glass), then continue behind it.
            float3 surface = directLight(pos, hit.normal, hit.albedo, hit.blockLight, light)
                           + hit.albedo * skyRadiance(reflect(dir, hit.normal), false) * 0.3;
            radiance += throughput * hit.alpha * surface;
            throughput *= (1.0 - hit.alpha) * lerp(float3(1.0, 1.0, 1.0), hit.albedo, 0.5);
            origin = pos;
            tMin = 0.001;
            continue;
        }

        radiance += throughput * shadeOpaque(pos, dir, hit, light, seed);
        depth = projectDepth(pos - frame.cameraOffset.xyz);
        break;
    }

    float accumulated = frame.params.y;
    float4 history = accumulation[pixel];
    float3 averaged = accumulated > 0.0 ? (history.rgb * accumulated + radiance) / (accumulated + 1.0) : radiance;
    accumulation[pixel] = float4(averaged, 1.0);

    float3 color = debugMode != 0 ? averaged : acesTonemap(averaged);
    outputImage[pixel] = float4(pow(color, 1.0 / 2.2), 1.0);
    depthOutput[pixel.y * size.x + pixel.x] = depth;
}
