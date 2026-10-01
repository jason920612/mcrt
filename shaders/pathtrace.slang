// Path tracer over Minecraft's own section meshes. Writes a G-buffer for the denoiser
// (denoise.slang), which produces the final image.
//
// Space: everything is block-relative (see frame.slang), so section instance transforms are exact
// and floats stay small. The camera itself sits at frame.cameraOffset.
//
// Hit groups (sbtRecordStride = 2 ray types):
//   0/1 solid primary/shadow, 2/3 cutout primary/shadow (alpha tested), 4/5 translucent primary/shadow.

#include "frame.slang"
#include "atmosphere.slang"

static const uint RAY_PRIMARY = 0;
static const uint RAY_SHADOW = 1;
static const uint RAY_TYPE_COUNT = 2;

static const uint MASK_OPAQUE = 0x01;      // solid + cutout instances
static const uint MASK_TRANSLUCENT = 0x02;
static const uint MASK_ENTITY = 0x04;      // mobs, players, items: shadows, reflections, bounces only
static const uint MASK_PRIMARY = 0xFF & ~MASK_ENTITY; // Minecraft draws entities itself

static const float PI = 3.14159265;

// Minecraft's BLOCK vertex format (28 bytes).
struct BlockVertex
{
    float3 position;         // relative to the section origin
    uint color;              // RGBA8: biome tint * baked shade/AO
    float2 uv;               // block atlas
    uint light;              // rewritten natively: bits 12-15 emission level, 16-23 material id;
                             // smoothed terrain (bit 0 set): 16-23 top material, 24-31 side material,
                             // and uv holds the octahedral-encoded smooth normal
};

struct SectionInfo
{
    uint64_t vertexAddress;
    uint cutoutFirstVertex;
    uint translucentFirstVertex;
    int3 origin;             // section block origin (absolute)
    uint pad;
};

// Lights near each section (its own and its 26 neighbors'), indexed by section slot.
struct GpuLight
{
    int3 blockPos;           // absolute
    uint packed;             // bits 0-3 emission level, 8-31 RGB8 color
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> noisyIllumination; // radiance / albedo of the first opaque surface
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> albedoModulation;  // throughput * albedo of that surface
[[vk::binding(3, 0)]] RWStructuredBuffer<float> depthOutput;
[[vk::binding(4, 0)]] ConstantBuffer<FrameUniforms> frame;
[[vk::binding(5, 0)]] StructuredBuffer<SectionInfo> sections;
[[vk::binding(6, 0)]] Texture2D<float4> blockAtlas;
[[vk::binding(7, 0)]] SamplerState atlasSampler;
[[vk::binding(8, 0)]] StructuredBuffer<uint4> lightRanges; // offset, count, weight (float bits), 0
[[vk::binding(9, 0)]] StructuredBuffer<GpuLight> lights;
[[vk::binding(10, 0)]] [format("rgba16f")] RWTexture2D<float4> foregroundOutput;  // sky, translucent layers, emission
[[vk::binding(11, 0)]] [format("rgba32f")] RWTexture2D<float4> positionOutput;    // block-relative, w = 1 on opaque hit
[[vk::binding(12, 0)]] [format("rgba16f")] RWTexture2D<float4> normalOutput;      // xyz normal, w = distance to camera

// PBR materials (tools/materials): one array layer per material, layer = material id - 1.
struct MaterialParams
{
    float scale;             // blocks per texture repeat
    uint flags;              // 1 = tinted by the biome color
    float textureSize;
    float pad;
};
static const uint MATERIAL_TINTED = 1;
static const uint VERTEX_CARD = 4;     // vertex light word: vegetation card (see terrain_mesher.cpp)
static const uint VERTEX_TERRAIN = 8;  // natural terrain
static const uint VERTEX_FAR = 16;     // far landscape heightfield (past render distance)
static const uint VERTEX_FAR_WATER = 32;
static const uint VERTEX_ENTITY = 64;  // entity geometry (entity_layer.cpp)
static const uint VERTEX_FAR_FOREST = 256;
[[vk::binding(13, 0)]] Texture2DArray<float4> materialAlbedo; // sRGB view: rgb linear base color, a = height
[[vk::binding(14, 0)]] Texture2DArray<float4> materialNormal; // BC5: normal xy (OpenGL convention)
[[vk::binding(15, 0)]] SamplerState materialSampler;
[[vk::binding(16, 0)]] StructuredBuffer<MaterialParams> materialParams;
[[vk::binding(17, 0)]] Texture2D<float4> skyViewLut;   // sky.slang
[[vk::binding(18, 0)]] SamplerState skySampler;
[[vk::binding(19, 0)]] Texture2D<float4> cloudViewLut; // sky.slang: rgb scattered, a transmittance
[[vk::binding(20, 0)]] Texture2DArray<float4> materialSurface; // BC5: roughness, ambient occlusion

// Temporal ReSTIR for block lights at the primary surface: each pixel keeps the light it chose
// (a reservoir), and the next frame merges it with its own candidates.
struct Reservoir
{
    int3 blockPos;           // chosen light (absolute block position)
    uint packed;             // its GpuLight.packed
    float3 position;         // shading point, relative to that frame's camera
    float W;                 // unbiased contribution weight (0: none, or found occluded)
    float M;                 // number of candidates it summarizes
    uint generation;         // light list version it was chosen from
    float2 pad;
};
[[vk::binding(21, 0)]] StructuredBuffer<Reservoir> reservoirPrevious;
[[vk::binding(22, 0)]] RWStructuredBuffer<Reservoir> reservoirOutput;
static const float RESERVOIR_HISTORY = 20.0; // cap on reused candidates, relative to new ones

// Set by the ray generation shader for the primary surface's lighting: the pixel whose reservoir
// blockLightIllumination reads and writes (-1: none, e.g. for bounces).
static int2 gReservoirPixel = int2(-1, -1);

// Material data as one vector: normal xy, roughness, ao.
float4 sampleMaterialData(float3 uvLayer, float lod)
{
    float2 normal = materialNormal.SampleLevel(materialSampler, uvLayer, lod).xy;
    float2 surface = materialSurface.SampleLevel(materialSampler, uvLayer, lod).xy;
    return float4(normal, surface);
}

static const uint WATER_MATERIAL = 255; // see SectionScanner.WATER_MATERIAL

struct Payload
{
    float3 albedo;
    float hitT;              // < 0 on miss
    float3 normal;
    float alpha;
    float emission;          // 0..1, from the emitting block this surface belongs to
    uint translucent;
    uint sectionSlot;
    float3 geometricNormal;  // for ray offsets; normal may be perturbed by a normal map
    float roughness;
    uint materialId;
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
    float emission;
    float3 geometricNormal;
    float roughness;
    uint materialId;
};

// ---------------------------------------------------------------------------------------------
// Macro detail for natural terrain. Material textures repeat every couple of blocks and average
// out to one flat color a few dozen blocks away; real ground keeps varying at every scale (patches
// of dry and lush grass, dirty snow hollows, rock strata) and is never perfectly smooth. These
// world-space noise fields add that variation to albedo and normals, and do not fade with distance.

float macroHash(float3 p)
{
    p = frac(p * 0.3183099 + 0.1);
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float macroNoise(float3 x)
{
    float3 i = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(lerp(macroHash(i), macroHash(i + float3(1, 0, 0)), f.x),
                     lerp(macroHash(i + float3(0, 1, 0)), macroHash(i + float3(1, 1, 0)), f.x), f.y),
                lerp(lerp(macroHash(i + float3(0, 0, 1)), macroHash(i + float3(1, 0, 1)), f.x),
                     lerp(macroHash(i + float3(0, 1, 1)), macroHash(i + float3(1, 1, 1)), f.x), f.y), f.z);
}

// Ground relief in blocks: lumps a few blocks wide with finer bumps on top.
float macroRelief(float3 p)
{
    return macroNoise(p * (1.0 / 6.0)) * 0.55 + macroNoise(p * (1.0 / 2.3) + 17.0) * 0.3
         + macroNoise(p * (1.0 / 0.9) + 41.0) * 0.15;
}

void applyMacroDetail(inout SurfaceSample s, float3 world)
{
    float3 albedo = s.albedo;
    float lum = dot(albedo, float3(0.2126, 0.7152, 0.0722));
    bool snowy = min(albedo.r, min(albedo.g, albedo.b)) > 0.45;
    bool vegetation = albedo.g > albedo.r * 1.08 && albedo.g > albedo.b * 1.2;

    // Relief: bend the shading normal along the noise gradient.
    const float e = 0.2;
    float h = macroRelief(world);
    float3 grad = float3(macroRelief(world + float3(e, 0, 0)) - h,
                         macroRelief(world + float3(0, e, 0)) - h,
                         macroRelief(world + float3(0, 0, e)) - h) / e;
    float3 n = s.normal;
    float amplitude = snowy ? 0.9 : 1.6;
    s.normal = normalize(n - amplitude * (grad - n * dot(grad, n)));

    // Large-scale tone: three octaves, from field-sized patches down to single blocks.
    float broad = macroNoise(world * (1.0 / 29.0) + 3.0);
    float mid = macroNoise(world * (1.0 / 8.5) + 9.0);
    float fine = macroNoise(world * (1.0 / 2.1) + 27.0);
    float tone = broad * 0.5 + mid * 0.3 + fine * 0.2;
    // Hollows collect dirt and shade; crests catch light.
    float cavity = saturate((h - 0.25) * 2.2);
    if (snowy)
    {
        albedo *= lerp(0.88, 1.03, tone) * lerp(float3(0.9, 0.95, 1.0), 1.0, cavity);
    }
    else if (vegetation)
    {
        // Patches of dry, yellowed grass and of dark, lush grass.
        float dry = smoothstep(0.5, 0.75, macroNoise(world * (1.0 / 17.0) + 61.0));
        float lush = smoothstep(0.55, 0.8, macroNoise(world * (1.0 / 11.0) + 83.0));
        albedo = lerp(albedo, albedo * float3(1.45, 1.12, 0.55), dry * 0.75);
        albedo = lerp(albedo, albedo * float3(0.62, 0.8, 0.62), lush * 0.7);
        albedo *= lerp(0.7, 1.2, tone) * lerp(0.75, 1.0, cavity);
    }
    else
    {
        // Rock, dirt, sand: strata (horizontal bands) and weathering.
        float strata = macroNoise(float3(world.x * 0.05, world.y * 0.9, world.z * 0.05) + 5.0);
        albedo *= lerp(0.68, 1.22, tone * 0.65 + strata * 0.35) * lerp(0.7, 1.0, cavity);
        albedo = lerp(albedo, albedo * float3(1.08, 1.0, 0.9), broad); // warm/cool shift
    }
    s.albedo = albedo;
}

// Far landscape: the heightfield only knows height and biome, so the look is synthesized the way
// a painter would: rock where slopes are steep, snow above a ragged snow line, forest canopies
// as clumps of dark crowns, and relief noise at the scales the sample spacing cannot carry.
void shadeFarLand(inout SurfaceSample s, float3 world, bool forest)
{
    float3 n = s.normal;
    // Rock shows where the terrain itself is steep (decided before the relief noise, which would
    // otherwise speckle gentle snowfields with rock).
    float steep = smoothstep(0.8, 0.6, n.y + (macroNoise(world * 0.03) - 0.5) * 0.1);
    // Relief: ridges a few tens of blocks wide and gullies, bending the shading normal.
    const float e = 1.5;
    float3 p = world * float3(1.0 / 23.0, 1.0 / 23.0, 1.0 / 23.0);
    float h0 = macroNoise(p) + 0.5 * macroNoise(p * 2.7 + 11.0) + 0.25 * macroNoise(p * 7.1 + 23.0);
    float hx = macroNoise(p + float3(e / 23.0, 0, 0)) + 0.5 * macroNoise((p + float3(e / 23.0, 0, 0)) * 2.7 + 11.0)
             + 0.25 * macroNoise((p + float3(e / 23.0, 0, 0)) * 7.1 + 23.0);
    float hz = macroNoise(p + float3(0, 0, e / 23.0)) + 0.5 * macroNoise((p + float3(0, 0, e / 23.0)) * 2.7 + 11.0)
             + 0.25 * macroNoise((p + float3(0, 0, e / 23.0)) * 7.1 + 23.0);
    float3 grad = float3(hx - h0, 0.0, hz - h0) / e * 5.0;
    n = normalize(n - (grad - n * dot(grad, n)));

    float3 albedo = s.albedo;
    float3 rock = float3(0.13, 0.12, 0.11) * (0.75 + 0.5 * macroNoise(world * float3(0.02, 0.35, 0.02)));
    if (forest && steep < 0.5)
    {
        // Crowns: dark clumps with lit tops.
        float crowns = macroNoise(world * 0.45) * 0.6 + macroNoise(world * 1.3 + 7.0) * 0.4;
        albedo *= lerp(0.45, 1.25, smoothstep(0.3, 0.8, crowns));
        n = normalize(n + float3(macroNoise(world * 0.5 + 3.0) - 0.5, 0.0, macroNoise(world * 0.5 + 9.0) - 0.5) * 0.8);
    }
    albedo = lerp(albedo, rock, steep);
    // Snow: above a ragged line on mountains, and only where it can lie.
    float snowLine = 150.0 + (macroNoise(world * 0.004) - 0.5) * 60.0;
    float snow = smoothstep(snowLine - 8.0, snowLine + 8.0, world.y + (macroNoise(world * 0.05) - 0.5) * 30.0)
               * smoothstep(0.55, 0.75, n.y);
    albedo = lerp(albedo, float3(0.85, 0.88, 0.92), snow);
    s.albedo = albedo;
    s.normal = n;
}

// Plains grass color: tinted materials are authored for it and shift with the biome tint.
static const float3 REFERENCE_TINT = float3(0.569, 0.741, 0.349);
// Plains foliage color (0x48B518): foliage cards are authored for it.
static const float3 REFERENCE_FOLIAGE = float3(0.282, 0.71, 0.094);
static const uint CARD_GRASS = 1;

float3 octDecode(float2 e)
{
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

struct MaterialSample
{
    float3 albedo;
    float3 normal;
    float roughness;
    float height;            // 0..1
};

// Triplanar sampling for smoothed (non axis-aligned) terrain, with UDN normal blending.
// UV orientations keep images upright on walls (v down the image = down in the world).
MaterialSample sampleTriplanar(uint materialId, float3 pos, float3 n, float distance)
{
    MaterialParams params = materialParams[materialId - 1];
    int3 cell = frame.cameraBlock.xyz + int3(floor(pos));
    int period = int(params.scale);
    int3 wrapped = ((cell % period) + period) % period;
    float3 coord = (float3(wrapped) + (pos - floor(pos))) / params.scale;
    float lod = log2(max(distance * frame.params.y * params.textureSize / params.scale, 1.0));
    float layer = float(materialId - 1);

    float3 weights = pow(abs(n), 4.0);
    weights /= weights.x + weights.y + weights.z;

    MaterialSample result;
    result.albedo = 0.0;
    result.roughness = 0.0;
    result.height = 0.0;
    float3 blended = 0.0;
    if (weights.x > 0.01)
    {
        float3 uv = float3(coord.z, -coord.y, layer);
        float4 data = sampleMaterialData(uv, lod);
        float4 albedo = materialAlbedo.SampleLevel(materialSampler, uv, lod);
        result.albedo += albedo.rgb * weights.x;
        result.height += albedo.a * weights.x;
        result.roughness += data.z * weights.x;
        float3 t = float3(data.xy * 2.0 - 1.0 + n.zy, n.x);
        blended += t.zyx * weights.x;
    }
    if (weights.y > 0.01)
    {
        float3 uv = float3(coord.x, -coord.z, layer);
        float4 data = sampleMaterialData(uv, lod);
        float4 albedo = materialAlbedo.SampleLevel(materialSampler, uv, lod);
        result.albedo += albedo.rgb * weights.y;
        result.height += albedo.a * weights.y;
        result.roughness += data.z * weights.y;
        float3 t = float3(data.xy * 2.0 - 1.0 + n.xz, n.y);
        blended += t.xzy * weights.y;
    }
    if (weights.z > 0.01)
    {
        float3 uv = float3(coord.x, -coord.y, layer);
        float4 data = sampleMaterialData(uv, lod);
        float4 albedo = materialAlbedo.SampleLevel(materialSampler, uv, lod);
        result.albedo += albedo.rgb * weights.z;
        result.height += albedo.a * weights.z;
        result.roughness += data.z * weights.z;
        float3 t = float3(data.xy * 2.0 - 1.0 + n.xy, n.z);
        blended += t.xyz * weights.z;
    }
    float total = (weights.x > 0.01 ? weights.x : 0.0) + (weights.y > 0.01 ? weights.y : 0.0) + (weights.z > 0.01 ? weights.z : 0.0);
    result.albedo /= total;
    result.roughness /= total;
    result.height /= total;
    result.normal = normalize(blended);
    return result;
}

// Smoothed blocks that keep Minecraft's texture (ores): triplanar over the block's atlas sprite,
// one sprite per block like vanilla. Sprite origin is packed in the color, width in the word.
void applySmoothAtlasSprite(inout SurfaceSample s, BlockVertex v, float3 smoothNormal)
{
    float2 origin = float2(v.color & 0xFFFF, v.color >> 16) / 65535.0;
    float width = float(v.light >> 16) / 65535.0;
    float2 size = float2(width, width * frame.atmosphere.z / max(frame.atmosphere.w, 1.0));
    float3 pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent() - s.geometricNormal * 0.001;
    float3 local = pos - floor(pos);
    float3 weights = pow(abs(smoothNormal), 4.0);
    weights /= weights.x + weights.y + weights.z;
    float3 color = 0.0;
    color += blockAtlas.SampleLevel(atlasSampler, origin + float2(local.z, 1.0 - local.y) * size, 0).rgb * weights.x;
    color += blockAtlas.SampleLevel(atlasSampler, origin + float2(local.x, local.z) * size, 0).rgb * weights.y;
    color += blockAtlas.SampleLevel(atlasSampler, origin + float2(local.x, 1.0 - local.y) * size, 0).rgb * weights.z;
    s.albedo = pow(color, 2.2);
    s.normal = smoothNormal;
    s.roughness = 0.8;
}

// Smoothed terrain: top material on gentle slopes, side material on steep ones, blended between.
void applySmoothMaterial(inout SurfaceSample s, uint word, float3 smoothNormal, float3 tint)
{
    uint topId = (word >> 16) & 0xFF;
    uint sideId = (word >> 24) & 0xFF;
    float3 pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float distance = length(pos - frame.cameraOffset.xyz);
    float3 inside = pos - s.geometricNormal * 0.001;
    // Snow (relief kind 4) clings to steep slopes too; other tops give way to their side material sooner.
    bool snowTop = topId != 0 && ((materialParams[topId - 1].flags >> 8) & 0xFF) == 4;
    float topWeight = snowTop ? smoothstep(0.12, 0.4, smoothNormal.y) : smoothstep(0.38, 0.65, smoothNormal.y);

    MaterialSample m;
    if (topId == sideId || topWeight >= 1.0 || sideId == 0)
        m = sampleTriplanar(topId, inside, smoothNormal, distance);
    else if (topWeight <= 0.0 || topId == 0)
        m = sampleTriplanar(sideId, inside, smoothNormal, distance);
    else
    {
        MaterialSample top = sampleTriplanar(topId, inside, smoothNormal, distance);
        MaterialSample side = sampleTriplanar(sideId, inside, smoothNormal, distance);
        if ((materialParams[topId - 1].flags & MATERIAL_TINTED) != 0)
            top.albedo *= clamp(pow(tint, 2.2) / pow(REFERENCE_TINT, 2.2), 0.0, 2.0);
        // Height blend: whichever material stands taller wins locally, so grass fills the gaps
        // between stones instead of fading uniformly into dirt.
        float blend = saturate((topWeight - 0.5) * 2.0 + (top.height - side.height) * 1.5);
        blend = smoothstep(0.35, 0.65, lerp(topWeight, blend, 0.8));
        m.albedo = lerp(side.albedo, top.albedo, blend);
        m.roughness = lerp(side.roughness, top.roughness, blend);
        m.normal = normalize(lerp(side.normal, top.normal, blend));
        s.albedo = m.albedo;
        s.normal = m.normal;
        s.roughness = clamp(m.roughness, 0.08, 1.0);
        return;
    }
    uint used = topWeight >= 0.5 ? topId : sideId;
    if (used != 0 && (materialParams[used - 1].flags & MATERIAL_TINTED) != 0)
        m.albedo *= clamp(pow(tint, 2.2) / pow(REFERENCE_TINT, 2.2), 0.0, 2.0);
    s.albedo = m.albedo;
    s.normal = m.normal;
    s.roughness = clamp(m.roughness, 0.08, 1.0);
}

// Parallax occlusion mapping: march the view ray into the height map (albedo alpha; 1 = surface)
// so close-up materials show depth. Fades out with distance, where it is invisible anyway.
static const float PARALLAX_DEPTH = 0.06;   // blocks
static const float PARALLAX_DISTANCE = 24.0; // blocks
static const int PARALLAX_STEPS = 12;

float2 parallaxOffset(float2 uv, float layer, float lod, float3 n, float3 tangent, float3 bitangent, float scale,
                      float distance)
{
    float3 v = -WorldRayDirection();
    float cosView = dot(v, n);
    // Fade out with distance and at grazing angles, where the offset would smear the texture.
    float amplitude = PARALLAX_DEPTH * saturate(1.0 - distance / PARALLAX_DISTANCE) * saturate((cosView - 0.15) * 3.0);
    if (amplitude <= 0.0)
        return uv;
    float vz = max(cosView, 0.3);
    // UV change per block of depth along the view ray (u follows the tangent, v runs against the bitangent).
    float2 uvPerDepth = float2(-dot(v, tangent), dot(v, bitangent)) / (vz * scale);

    float stepDepth = amplitude / float(PARALLAX_STEPS);
    float depth = 0.0;
    float prevDepth = 0.0;
    float prevSurface = 0.0;
    float2 current = uv;
    for (int i = 0; i < PARALLAX_STEPS; ++i)
    {
        float surface = amplitude * (1.0 - materialAlbedo.SampleLevel(materialSampler, float3(current, layer), lod).a);
        if (depth >= surface)
        {
            // Interpolate between the last two steps for a smooth intersection.
            float before = prevSurface - prevDepth;
            float after = depth - surface;
            float t = before / max(before + after, 1e-5);
            float hit = lerp(prevDepth, depth, t);
            return uv + uvPerDepth * hit;
        }
        prevDepth = depth;
        prevSurface = surface;
        depth += stepDepth;
        current = uv + uvPerDepth * depth;
    }
    return current;
}

// Blocks with a PBR material: world-space UVs so one texture spans `scale` blocks seamlessly,
// projected along the face's dominant axis, with mip level from the pixel footprint.
void applyMaterial(inout SurfaceSample s, uint materialId, float3 tint)
{
    MaterialParams params = materialParams[materialId - 1];
    float3 pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent(); // block-relative
    float3 n = s.geometricNormal;
    float3 inside = pos - n * 0.001;
    int3 cell = frame.cameraBlock.xyz + int3(floor(inside));
    float3 local = pos - floor(inside);
    int period = int(params.scale);
    int3 wrapped = ((cell % period) + period) % period;
    float3 coord = (float3(wrapped) + local) / params.scale;

    // UV axes per face; tangent = dP/du, bitangent = -dP/dv (OpenGL normal maps, v down the image).
    float3 an = abs(n);
    float2 uv;
    float3 tangent, bitangent;
    if (an.y >= an.x && an.y >= an.z)
    {
        uv = coord.xz;
        tangent = float3(1, 0, 0);
        bitangent = float3(0, 0, -1);
    }
    else if (an.x >= an.z)
    {
        uv = float2(coord.z, -coord.y);
        tangent = float3(0, 0, 1);
        bitangent = float3(0, 1, 0);
    }
    else
    {
        uv = float2(coord.x, -coord.y);
        tangent = float3(1, 0, 0);
        bitangent = float3(0, 1, 0);
    }

    float distance = length(pos - frame.cameraOffset.xyz);
    float cosView = max(abs(dot(n, WorldRayDirection())), 0.2);
    // Isotropic filtering over-blurs surfaces seen at an angle; sqrt keeps them textured (TAA hides the aliasing).
    float footprint = distance * frame.params.y * params.textureSize / params.scale / sqrt(cosView);
    float lod = log2(max(footprint, 1.0));
    float layer = float(materialId - 1);

    uv = parallaxOffset(uv, layer, lod, n, tangent, bitangent, params.scale, distance);
    float4 albedo = materialAlbedo.SampleLevel(materialSampler, float3(uv, layer), lod);
    float4 data = sampleMaterialData(float3(uv, layer), lod);

    s.albedo = albedo.rgb;
    if ((params.flags & MATERIAL_TINTED) != 0)
        s.albedo *= clamp(pow(tint, 2.2) / pow(REFERENCE_TINT, 2.2), 0.0, 2.0);
    float2 nxy = data.xy * 2.0 - 1.0;
    float3 tangentNormal = float3(nxy, sqrt(saturate(1.0 - dot(nxy, nxy))));
    s.normal = normalize(tangent * tangentNormal.x + bitangent * tangentNormal.y + n * tangentNormal.z);
    if (dot(s.normal, n) < 0.05) // keep the shading normal on the visible side
        s.normal = normalize(s.normal + n * (0.05 - dot(s.normal, n)));
    s.roughness = clamp(data.z, 0.08, 1.0);
}

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

    SurfaceSample s;
    s.albedo = pow(tex.rgb * tint.rgb, 2.2); // atlas and tint are sRGB-encoded
    s.alpha = tex.a * tint.a;
    s.normal = n;
    s.geometricNormal = n;
    s.roughness = 0.9;
    s.materialId = 0;
    s.emission = float((v0.light >> 12) & 0xF) / 15.0;
    if ((v0.light & VERTEX_ENTITY) != 0)
    {
        s.albedo = pow(unpackColor(v0.color).rgb, 2.2);
        s.alpha = 1.0;
        s.roughness = 0.8;
        return s;
    }
    if ((v0.light & VERTEX_FAR) != 0)
    {
        // Far landscape: only past the loaded chunks (inside them it would cover the real world).
        float3 pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float2 rel = pos.xz - frame.cameraOffset.xz;
        float inner = frame.atmosphere.y - 12.0;
        s.alpha = dot(rel, rel) < inner * inner ? 0.0 : 1.0;
        s.albedo = pow(saturate(tint.rgb), 2.2);
        float3 farNormal = octDecode(uv);
        s.normal = dot(farNormal, n) < 0.0 ? -farNormal : farNormal;
        bool water = (v0.light & VERTEX_FAR_WATER) != 0 && (v1.light & VERTEX_FAR_WATER) != 0 && (v2.light & VERTEX_FAR_WATER) != 0;
        s.roughness = water ? 0.06 : 0.95;
        if (water)
            s.materialId = WATER_MATERIAL; // shaded as water: reflection, refraction into deep sea
        if (!water && s.alpha > 0.0)
            shadeFarLand(s, pos + float3(frame.cameraBlock.xyz), (v0.light & VERTEX_FAR_FOREST) != 0);
        return s;
    }
    if ((v0.light & VERTEX_CARD) != 0)
    {
        // Vegetation card: material texture with coverage alpha, tinted by the biome color, lit
        // with a mostly upward normal so a field of cards shades like a meadow, not like planes.
        uint cardId = (v0.light >> 16) & 0xFF;
        MaterialParams params = materialParams[cardId - 1];
        float3 pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        float distance = length(pos - frame.cameraOffset.xyz);
        float lod = log2(max(distance * frame.params.y * params.textureSize * 0.6, 1.0));
        float4 t = materialAlbedo.SampleLevel(materialSampler, float3(uv, float(cardId - 1)), lod);
        bool grass = ((params.flags >> 16) & 0xFF) == CARD_GRASS;
        float3 reference = grass ? REFERENCE_TINT : REFERENCE_FOLIAGE;
        // Grass follows the biome color fully; foliage textures are authored natural greens, so the
        // biome foliage color only nudges them (it is far more saturated than real leaves).
        float3 ratio = clamp(pow(tint.rgb, 2.2) / pow(reference, 2.2), 0.0, 2.5);
        s.albedo = t.rgb * (grass ? ratio : lerp(1.0, clamp(sqrt(ratio), 0.6, 1.5), 0.45));
        s.alpha = t.a;
        s.normal = normalize(n * (grass ? 0.35 : 0.8) + float3(0.0, 1.0, 0.0) * (grass ? 1.0 : 0.4));
        s.materialId = grass ? 1 : 0; // marks grass for the shadow any-hit
        s.roughness = 0.7;
        return s;
    }
    if ((v0.light & 1) != 0)
    {
        float3 smoothNormal = normalize(octDecode(v0.uv) * bary.x + octDecode(v1.uv) * bary.y + octDecode(v2.uv) * bary.z);
        if (dot(smoothNormal, n) < 0.0)
            smoothNormal = -smoothNormal;
        s.albedo = 1.0;
        s.alpha = 1.0;
        float3 world = WorldRayOrigin() + WorldRayDirection() * RayTCurrent() + float3(frame.cameraBlock.xyz);
        // Each vertex carries its nearest block's materials. Blend them by barycentrics broken up
        // with noise (seeded per material, so a boundary meanders consistently across triangles),
        // sharpened so materials meet in irregular patches instead of smearing into each other.
        BlockVertex corners[3] = { v0, v1, v2 };
        float weights[3];
        for (uint c = 0; c < 3; ++c)
        {
            float seed = float((corners[c].light >> 16) & 0xFFFF) * 0.137;
            float jitter = 0.45 + 1.1 * macroNoise(world * 0.85 + seed);
            weights[c] = pow(max(bary[c] * jitter, 1e-4), 4.0);
        }
        // Fold vertices that share materials.
        for (uint c = 1; c < 3; ++c)
            for (uint d = 0; d < c; ++d)
                if (weights[d] > 0.0 && corners[c].light == corners[d].light)
                {
                    weights[d] += weights[c];
                    weights[c] = 0.0;
                }
        float total = weights[0] + weights[1] + weights[2];
        float3 albedo = 0.0, normal = 0.0;
        float roughness = 0.0, used = 0.0;
        for (uint c = 0; c < 3; ++c)
        {
            if (weights[c] < total * 0.02)
                continue;
            SurfaceSample m = s;
            if ((corners[c].light & 2) != 0)
                applySmoothAtlasSprite(m, corners[c], smoothNormal);
            else
                applySmoothMaterial(m, corners[c].light, smoothNormal, unpackColor(corners[c].color).rgb);
            float w = weights[c];
            used += w;
            albedo += m.albedo * w;
            normal += m.normal * w;
            roughness += m.roughness * w;
        }
        s.albedo = albedo / used;
        s.normal = normalize(normal + smoothNormal * 1e-3);
        s.roughness = roughness / used;
        if ((v0.light & VERTEX_TERRAIN) != 0)
            applyMacroDetail(s, world);
        if (dot(s.normal, n) < 0.05)
            s.normal = normalize(s.normal + n * (0.05 - dot(s.normal, n)));
        return s;
    }
    uint materialId = (v0.light >> 16) & 0xFF;
    s.materialId = materialId;
    if (materialId != 0 && materialId != WATER_MATERIAL)
        applyMaterial(s, materialId, tint.rgb);
    return s;
}

void writePayload(inout Payload p, SurfaceSample s, uint translucent)
{
    p.albedo = s.albedo;
    p.hitT = RayTCurrent();
    p.normal = s.normal;
    p.alpha = s.alpha;
    p.emission = s.emission;
    p.translucent = translucent;
    p.sectionSlot = InstanceID();
    p.geometricNormal = s.geometricNormal;
    p.roughness = s.roughness;
    p.materialId = s.materialId;
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

// Far landscape inside render distance lies under the real terrain, so rays meet it constantly;
// reject those hits from one vertex word and the hit point, without sampling the surface.
bool farLandscapeHidden()
{
    SectionInfo info = sections[InstanceID()];
    BlockVertex* vertices = (BlockVertex*)info.vertexAddress;
    uint prim = PrimitiveIndex();
    uint word = vertices[info.cutoutFirstVertex + (prim >> 1) * 4].light;
    if ((word & VERTEX_FAR) == 0)
        return false;
    float2 rel = (WorldRayOrigin() + WorldRayDirection() * RayTCurrent()).xz - frame.cameraOffset.xz;
    float inner = frame.atmosphere.y - 12.0;
    return dot(rel, rel) < inner * inner;
}

[shader("anyhit")]
void anyHitCutoutPrimary(inout Payload p, BuiltInTriangleIntersectionAttributes attribs)
{
    if (farLandscapeHidden())
    {
        IgnoreHit();
        return;
    }
    if (fetchSurface(attribs, 1).alpha < 0.5)
        IgnoreHit();
}

[shader("anyhit")]
void anyHitCutoutShadow(inout ShadowPayload p, BuiltInTriangleIntersectionAttributes attribs)
{
    if (farLandscapeHidden())
    {
        IgnoreHit();
        return;
    }
    SurfaceSample s = fetchSurface(attribs, 1);
    // Grass blades are thin: a card blocks only part of the light passing it (stochastically,
    // which the denoiser turns into soft shade instead of solid blotches).
    float coverage = s.materialId == 1 ? s.alpha * 0.35 : s.alpha;
    float threshold = s.materialId == 1 ? frac(sin(dot(WorldRayOrigin() + WorldRayDirection() * RayTCurrent(), float3(12.9898, 78.233, 37.719))) * 43758.5453) : 0.5;
    if (coverage < threshold)
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
// Sampling helpers

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

// ---------------------------------------------------------------------------------------------
// Lighting. Functions named *Illumination return incoming light weighted for a Lambertian surface
// with albedo 1 (pi folded into the light units); multiply by albedo for outgoing radiance.

#include "clouds.slang"

float3 skyRadiance(float3 d, bool includeSunDisk)
{
    if (skyType(frame) != SKY_OVERWORLD)
    {
        // Nether / End: the dimension's fog color is the only light from "outside".
        float3 fog = pow(max(frame.skyColor.rgb, 0.0), 2.2);
        return fog * (skyType(frame) == SKY_END ? 0.6 : 1.2);
    }
    float3 color = skyViewLut.SampleLevel(skySampler, skyViewUv(skyLookupDirection(d)), 0).rgb;
    if (includeSunDisk)
    {
        float sunAmount = dot(d, frame.sunDir.xyz);
        if (sunAmount > 0.99989 && d.y > -0.01)
            color += transmittanceToSpace(VIEWER, d) * 60.0 * saturate(frame.sunDir.w * 4.0);
        // Moon disc (Minecraft's own sky pass is skipped), with a soft halo.
        float moonAmount = dot(d, frame.moonDir.xyz);
        if (frame.moonDir.y > -0.05)
        {
            if (moonAmount > 0.99985)
                color += float3(0.85, 0.9, 1.0) * 6.0;
            color += float3(0.55, 0.65, 0.9) * pow(saturate(moonAmount), 400.0) * 0.08;
        }
        // Stars, fading in as the sky darkens.
        float night = saturate(1.0 - frame.sunDir.w * 3.0) * (1.0 - frame.params.w);
        if (night > 0.0 && d.y > 0.0)
        {
            float3 cell = floor(d * 400.0);
            float star = hash13(cell);
            if (star > 0.9985)
                color += float3(0.9, 0.95, 1.0) * (star - 0.9985) * 400.0 * night * saturate(d.y * 4.0);
        }
    }
    return color;
}


// Sky seen through the volumetric cloud layer (from the cloud LUT, marched once per frame).
float3 skyWithClouds(float3 origin, float3 dir, bool includeSunDisk)
{
    float3 sky = skyRadiance(dir, includeSunDisk);
    if (dir.y < 0.01)
        return sky;
    float4 clouds = cloudViewLut.SampleLevel(skySampler, cloudViewUv(dir), 0);
    float nearHorizon = smoothstep(0.01, 0.15, dir.y); // far-away clouds sink into the haze
    return lerp(sky, sky * clouds.a + clouds.rgb, nearHorizon);
}

bool traceShadow(float3 origin, float3 dir, float maxT)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = maxT;
    ShadowPayload shadow;
    shadow.visible = 0;
    TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             MASK_OPAQUE | MASK_ENTITY, RAY_SHADOW, RAY_TYPE_COUNT, 1, ray, shadow);
    return shadow.visible != 0;
}

// Radiant intensity of a block light (pi folded in, like the sun term): level 15 lights a
// surface 2 blocks away about as brightly as open sky.
static const float BLOCK_LIGHT_INTENSITY = 5.0;
// Radiance of an emitting block's own surface, relative to its texture.
static const float EMISSIVE_SURFACE_RADIANCE = 4.0;
static const uint LIGHT_CANDIDATES = 8;

float3 lightRadiance(uint packed)
{
    float level = float(packed & 0xF) / 15.0;
    float3 color = float3((packed >> 8) & 0xFF, (packed >> 16) & 0xFF, packed >> 24) / 255.0;
    return pow(color, 2.2) * level * level * BLOCK_LIGHT_INTENSITY;
}

// Block lights near a surface: resampled importance sampling (RIS) over the section's light list,
// then a single shadow ray for the chosen candidate.
uint reservoirIndex(int2 pixel)
{
    return uint(pixel.y) * DispatchRaysDimensions().x + uint(pixel.x);
}

// Last frame's reservoir for the surface at pos (block-relative), if it is the same surface.
bool previousReservoir(float3 pos, out Reservoir r)
{
    r = (Reservoir)0;
    float3 previousRelative = pos + frame.prevCameraShift.xyz;
    float4 clip = transformColumns(frame.prevViewProj, float4(previousRelative, 1.0));
    if (clip.w <= 0.0 || frame.taa.w < 0.0)
        return false;
    float2 uv = clip.xy / clip.w * 0.5 + 0.5;
    if (any(uv < 0.0) || any(uv >= 1.0))
        return false;
    int2 pixel = int2(uv * float2(DispatchRaysDimensions().xy));
    r = reservoirPrevious[reservoirIndex(pixel)];
    float3 d = r.position - previousRelative;
    return r.M > 0.0 && r.generation == uint(frame.taa.w) && dot(d, d) < 0.09;
}

// Pixels lit only every other frame (checkerboard) keep their reservoir alive.
void carryReservoir(int2 pixel, float3 pos)
{
    Reservoir r;
    if (previousReservoir(pos, r))
        r.position = pos - frame.cameraOffset.xyz;
    else
        r = (Reservoir)0;
    reservoirOutput[reservoirIndex(pixel)] = r;
}

float3 blockLightIllumination(float3 pos, float3 n, uint sectionSlot, inout uint seed)
{
    int2 reservoirPixel = gReservoirPixel;
    gReservoirPixel = int2(-1, -1); // only the primary surface's own call
    bool primary = reservoirPixel.x >= 0;
    uint4 range = lightRanges[sectionSlot];
    if (range.y == 0)
    {
        if (primary)
            reservoirOutput[reservoirIndex(reservoirPixel)] = (Reservoir)0;
        return 0.0;
    }
    float listWeight = asfloat(range.z); // each listed light stands for this many

    uint candidates = min(range.y, LIGHT_CANDIDATES);
    float weightSum = 0.0;
    float chosenTarget = 0.0;
    float3 chosenPoint = 0.0;
    float3 chosenRadiance = 0.0;
    int3 chosenBlock = 0;
    uint chosenPacked = 0;
    for (uint i = 0; i < candidates; ++i)
    {
        uint index = range.x + min(uint(random(seed) * range.y), range.y - 1);
        GpuLight light = lights[index];
        // A point inside the emitting block; soft shadows come from varying it.
        float3 jitter = float3(random(seed), random(seed), random(seed)) - 0.5;
        float3 lightPoint = float3(light.blockPos - frame.cameraBlock.xyz) + 0.5 + jitter * 0.4;
        float3 toLight = lightPoint - pos;
        float dist2 = max(dot(toLight, toLight), 0.25);
        float cosTheta = dot(n, toLight) * rsqrt(dist2);
        float3 radiance = lightRadiance(light.packed);
        float target = cosTheta > 0.0 ? luminance(radiance) * cosTheta / dist2 : 0.0;
        float weight = target * float(range.y); // target / source pdf (uniform over the list)
        weightSum += weight;
        if (weight > 0.0 && random(seed) * weightSum < weight)
        {
            chosenTarget = target;
            chosenPoint = lightPoint;
            chosenRadiance = radiance;
            chosenBlock = light.blockPos;
            chosenPacked = light.packed;
        }
    }
    float M = float(candidates);

    // Temporal reuse: last frame's choice competes with this frame's, weighted by how many
    // candidates it stands for (capped, so lighting still reacts to change).
    if (primary)
    {
        Reservoir previous;
        if (previousReservoir(pos, previous))
        {
            float3 lightPoint = float3(previous.blockPos - frame.cameraBlock.xyz) + 0.5
                              + (float3(random(seed), random(seed), random(seed)) - 0.5) * 0.4;
            float3 toLight = lightPoint - pos;
            float dist2 = max(dot(toLight, toLight), 0.25);
            float cosTheta = dot(n, toLight) * rsqrt(dist2);
            float3 radiance = lightRadiance(previous.packed);
            float target = cosTheta > 0.0 ? luminance(radiance) * cosTheta / dist2 : 0.0;
            float previousM = min(previous.M, RESERVOIR_HISTORY * float(candidates));
            float weight = target * previous.W * previousM;
            weightSum += weight;
            M += previousM;
            if (weight > 0.0 && random(seed) * weightSum < weight)
            {
                chosenTarget = target;
                chosenPoint = lightPoint;
                chosenRadiance = radiance;
                chosenBlock = previous.blockPos;
                chosenPacked = previous.packed;
            }
        }
    }

    float W = chosenTarget > 0.0 ? weightSum / (M * chosenTarget) : 0.0;
    float3 result = 0.0;
    if (W > 0.0)
    {
        float3 toLight = chosenPoint - pos;
        float dist = length(toLight);
        float3 dir = toLight / dist;
        // Stop short of the light's own block (half diagonal), which would otherwise shadow itself.
        if (dist > 0.9 && !traceShadow(pos + n * 0.002, dir, dist - 0.87))
            W = 0.0; // occluded: do not hand this light on
        else
            result = chosenRadiance * dot(n, dir) / max(dist * dist, 0.25) * W * listWeight;
    }
    if (primary)
    {
        Reservoir r;
        r.blockPos = chosenBlock;
        r.packed = chosenPacked;
        r.position = pos - frame.cameraOffset.xyz;
        r.W = W;
        r.M = M;
        r.generation = uint(max(frame.taa.w, 0.0));
        r.pad = 0.0;
        reservoirOutput[reservoirIndex(reservoirPixel)] = r;
    }
    return result;
}

float3 directIllumination(float3 pos, float3 n, uint sectionSlot, KeyLight light, inout uint seed)
{
    float3 result = blockLightIllumination(pos, n, sectionSlot, seed);
    float ndotl = dot(n, light.dir);
    if (ndotl > 0.0 && any(light.radiance > 0.0) && traceShadow(pos + n * 0.004, light.dir, 1024.0))
        result += light.radiance * ndotl * cloudShadow(pos, light.dir);
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

// GGX specular from the key light (dielectric, F0 = 0.04), already multiplied by n.l.
float3 keyLightSpecular(float3 n, float3 viewDir, float roughness, KeyLight light)
{
    float3 l = light.dir;
    float ndotl = dot(n, l);
    float ndotv = dot(n, viewDir);
    if (ndotl <= 0.0 || ndotv <= 0.0)
        return 0.0;
    float3 h = normalize(l + viewDir);
    float ndoth = saturate(dot(n, h));
    float a = roughness * roughness;
    float a2 = a * a;
    float d = a2 / (PI * pow(ndoth * ndoth * (a2 - 1.0) + 1.0, 2.0));
    float k = a * 0.5;
    float g = (ndotl / (ndotl * (1.0 - k) + k)) * (ndotv / (ndotv * (1.0 - k) + k));
    float fresnel = 0.04 + 0.96 * pow(1.0 - saturate(dot(h, viewDir)), 5.0);
    // pi folded into light units, matching the diffuse term.
    return light.radiance * (d * g * fresnel / (4.0 * ndotl * ndotv)) * ndotl * PI;
}

// Direct light + one diffuse bounce at an opaque surface (diffuse illumination, albedo excluded).
float3 surfaceIllumination(float3 pos, Payload hit, KeyLight light, inout uint seed)
{
    float3 illumination = directIllumination(pos, hit.normal, hit.sectionSlot, light, seed);

    float3 bounceDir = cosineSampleHemisphere(hit.normal, seed);
    float3 bounceOrigin = pos + hit.geometricNormal * 0.002;
    Payload bounce = tracePrimary(bounceOrigin, bounceDir, MASK_OPAQUE | MASK_ENTITY, 0.0);
    // Cosine-weighted sampling: the pdf cancels the cosine and 1/pi of the Lambertian BRDF.
    if (bounce.hitT < 0.0)
        illumination += skyRadiance(bounceDir, false);
    else // emission of a hit light is skipped: next event estimation already covers it
        illumination += bounce.albedo * directIllumination(bounceOrigin + bounceDir * bounce.hitT, bounce.normal,
                                                           bounce.sectionSlot, light, seed);
    // Rare very bright paths (e.g. a bounce straight onto a lit wall next to a light) would
    // otherwise survive denoising as fireflies.
    float lum = luminance(illumination);
    return lum > 64.0 ? illumination * (64.0 / lum) : illumination;
}

// Water: absorption and in-scattering per block of path, and an animated wave normal.
static const float3 WATER_EXTINCTION = float3(0.30, 0.075, 0.05);
static const float3 WATER_SCATTER = float3(0.010, 0.045, 0.060);

float waveHeight(float2 p, float time)
{
    float h = 0.0;
    h += sin(dot(p, float2(0.92, 0.38)) * 1.6 + time * 1.7) * 0.035;
    h += sin(dot(p, float2(-0.45, 0.89)) * 2.3 + time * 2.1) * 0.025;
    h += (valueNoise(float3(p * 1.7, time * 0.6)) - 0.5) * 0.05;
    h += (valueNoise(float3(p * 4.1 + 13.0, time * 0.9)) - 0.5) * 0.02;
    return h;
}

float3 waterNormal(float3 pos)
{
    float2 p = worldPosition(pos).xz;
    float time = frame.params.x;
    const float e = 0.05;
    float h = waveHeight(p, time);
    float dx = (waveHeight(p + float2(e, 0.0), time) - h) / e;
    float dz = (waveHeight(p + float2(0.0, e), time) - h) / e;
    return normalize(float3(-dx, 1.0, -dz));
}

float3 unproject(float2 ndc, float z)
{
    float4 p = transformColumns(frame.invViewProj, float4(ndc, z, 1.0));
    return p.xyz / p.w;
}

float projectDepth(float3 cameraRelative)
{
    float4 clip = transformColumns(frame.viewProj, float4(cameraRelative, 1.0));
    return saturate(clip.z / clip.w);
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 size = DispatchRaysDimensions().xy;
    uint seed = pcgHash(pixel.x + pixel.y * size.x + frame.frameInfo.x * size.x * size.y);

    // Minecraft's image memory is bottom-up, so NDC y follows the row index directly.
    // Sub-pixel jitter per frame; the TAA pass turns it into anti-aliasing (and upscaling).
    float2 ndc = ((float2(pixel) + 0.5 + frame.taa.xy) / float2(size)) * 2.0 - 1.0;
    float3 a = unproject(ndc, 1.0);
    float3 b = unproject(ndc, 0.5);
    float3 nearPoint = dot(a, a) < dot(b, b) ? a : b;
    float3 farPoint = dot(a, a) < dot(b, b) ? b : a;
    float3 dir = normalize(farPoint - nearPoint);
    float3 origin = frame.cameraOffset.xyz + nearPoint;

    KeyLight light = keyLight();
    uint debugMode = uint(frame.params.z);
    // Checkerboard lighting: primary visibility everywhere, expensive lighting on alternating halves.
    bool lightThisPixel = (frame.frameInfo.w & FLAG_CHECKERBOARD_LIGHTING) == 0
                       || ((pixel.x + pixel.y + frame.frameInfo.x) & 1) == 0;
    float3 foreground = 0.0;
    float3 throughput = 1.0;
    float3 albedo = 0.0;
    float3 illumination = 0.0;
    float4 position = 0.0;
    float4 normal = 0.0;
    float depth = 0.0; // Minecraft clears depth to 0 (reverse Z)
    float firstSurfaceDistance = 0.0; // camera to the first thing the ray meets (0 = sky); compose hazes by it
    bool reservoirWritten = false;

    float tMin = 0.0;
    bool layerLimitReached = false;
    bool inWater = (frame.frameInfo.w & FLAG_CAMERA_IN_WATER) != 0;
    bool sawWater = inWater;
    float3 waterAmbient = skyRadiance(float3(0, 1, 0), false) + light.radiance * saturate(light.dir.y);
    if ((frame.frameInfo.w & FLAG_CAMERA_IN_LAVA) != 0)
    {
        foreground = float3(2.0, 0.5, 0.08); // inside lava: nothing else is visible
        layerLimitReached = true;
    }
    for (uint layer = 0; layer < 4 && !layerLimitReached; ++layer)
    {
        Payload hit = tracePrimary(origin, dir, MASK_PRIMARY, tMin);
        bool fromWater = inWater;
        if (inWater)
        {
            // Light lost and scattered along the underwater path to whatever comes next.
            float pathLength = hit.hitT < 0.0 ? 64.0 : hit.hitT;
            float3 absorption = exp(-WATER_EXTINCTION * pathLength);
            foreground += throughput * WATER_SCATTER * waterAmbient * (1.0 - absorption);
            throughput *= absorption;
            inWater = false;
        }
        if (hit.hitT < 0.0)
        {
            // Under water, a ray heading down that meets nothing (a crack in the sea floor, or the
            // far sea, which has no floor) ends in the deep: its scattering was added above.
            if (!(fromWater && dir.y < 0.0))
                foreground += throughput * skyWithClouds(origin, dir, true);
            break;
        }

        float3 pos = origin + dir * hit.hitT;
        if (firstSurfaceDistance == 0.0)
            firstSurfaceDistance = max(length(pos - frame.cameraOffset.xyz), 1e-3);
        if (debugMode == 1 || debugMode == 2)
        {
            foreground = debugMode == 1 ? hit.albedo : hit.normal * 0.5 + 0.5;
            break;
        }

        // Water surfaces (and the far landscape's sea, the only opaque surface tagged as water).
        if (hit.materialId == WATER_MATERIAL)
        {
            // Water: wave normal, Fresnel reflection, sun glint, then refract into the volume.
            float3 n = hit.geometricNormal; // faces the ray
            if (abs(n.y) > 0.5)
            {
                float3 wave = waterNormal(pos);
                n = n.y > 0.0 ? wave : float3(wave.x, -wave.y, wave.z);
            }
            float cosIncident = saturate(dot(-dir, n));
            float fresnel = 0.02 + 0.98 * pow(1.0 - cosIncident, 5.0);

            float3 reflected = reflect(dir, n);
            if (!fromWater)
                reflected.y = abs(reflected.y);
            float3 reflectOrigin = pos + hit.geometricNormal * 0.01;
            Payload r = tracePrimary(reflectOrigin, reflected, MASK_OPAQUE | MASK_ENTITY, 0.0);
            float3 reflection;
            if (r.hitT < 0.0)
                reflection = skyWithClouds(reflectOrigin, reflected, false);
            else
            {
                // Cheap shading: the reflection is dimmed by Fresnel and broken up by waves anyway.
                reflection = r.albedo * (light.radiance * saturate(dot(r.normal, light.dir))
                                         + skyRadiance(float3(0, 1, 0), false) * 0.6 + r.emission * EMISSIVE_SURFACE_RADIANCE);
            }
            float3 glint = keyLightSpecular(n, -dir, 0.06, light);
            if (any(glint > 0.0) && traceShadow(reflectOrigin, light.dir, 1024.0))
                reflection += glint * cloudShadow(pos, light.dir) / max(fresnel, 0.02);
            foreground += throughput * fresnel * reflection;
            throughput *= 1.0 - fresnel;

            waterAmbient = skyRadiance(float3(0, 1, 0), false) + light.radiance * saturate(light.dir.y);
            // Entering water from air bends towards the normal; leaving it can totally reflect.
            float3 refracted = refract(dir, n, fromWater ? 1.33 : 1.0 / 1.33);
            bool totalReflection = dot(refracted, refracted) <= 0.0;
            dir = totalReflection ? reflected : normalize(refracted);
            origin = pos;
            tMin = 0.001;
            inWater = fromWater ? totalReflection : dir.y < 0.0;
            sawWater = true;
            continue;
        }

        if (hit.translucent != 0)
        {
            // Lit translucent surface (water, stained glass), then continue behind it.
            float3 surface = hit.albedo * (directIllumination(pos, hit.normal, hit.sectionSlot, light, seed)
                                           + skyRadiance(reflect(dir, hit.normal), false) * 0.3);
            foreground += throughput * hit.alpha * surface;
            throughput *= (1.0 - hit.alpha) * lerp(float3(1.0, 1.0, 1.0), hit.albedo, 0.5);
            origin = pos;
            tMin = 0.001;
            continue;
        }

        foreground += throughput * hit.albedo * hit.emission * EMISSIVE_SURFACE_RADIANCE;
        reservoirWritten = true;
        if (!lightThisPixel)
            carryReservoir(int2(pixel), pos);
        if (lightThisPixel)
        {
            gReservoirPixel = int2(pixel);
            // Seen through water, most of the light is absorbed: direct light is enough.
            // Beyond render distance (the far landscape) a bounce is invisible: sky fill instead.
            float2 fromCamera = pos.xz - frame.cameraOffset.xz;
            bool far = dot(fromCamera, fromCamera) > frame.atmosphere.y * frame.atmosphere.y;
            illumination = sawWater || far ? directIllumination(pos, hit.normal, hit.sectionSlot, light, seed)
                                                 + skyRadiance(float3(0, 1, 0), false) * (far ? 0.8 : 0.5)
                                           : surfaceIllumination(pos, hit, light, seed);
            // Specular rides along in the demodulated buffer (divided by albedo here, multiplied back in compose).
            float3 specular = keyLightSpecular(hit.normal, -dir, hit.roughness, light);
            if (any(specular > 0.0) && traceShadow(pos + hit.geometricNormal * 0.004, light.dir, 1024.0))
                illumination += specular * cloudShadow(pos, light.dir) / max(hit.albedo, 0.03);
        }
        albedo = throughput * hit.albedo;
        float3 cameraRelative = pos - frame.cameraOffset.xyz;
        position = float4(pos, 1.0);
        normal = float4(hit.normal, length(cameraRelative));
        depth = projectDepth(cameraRelative);
        break;
    }

    // w = 1 when this pixel was lit this frame; the denoiser fills the others from neighbors.
    noisyIllumination[pixel] = float4(illumination, lightThisPixel ? 1.0 : 0.0);
    albedoModulation[pixel] = float4(albedo, 0.0);
    foregroundOutput[pixel] = float4(foreground, firstSurfaceDistance);
    if (!reservoirWritten)
        reservoirOutput[reservoirIndex(int2(pixel))] = (Reservoir)0;
    positionOutput[pixel] = position;
    normalOutput[pixel] = normal;
    depthOutput[pixel.y * size.x + pixel.x] = depth;
}
