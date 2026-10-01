#pragma once

#include <cstdint>
#include <vector>

// Natural terrain, re-sculpted (visual only; collision stays blocky).
//
// Blocks only mark where ground is and what it is made of. The visible surface is the zero set of
// a density field sampled every half block: a smooth vote of the surrounding blocks, displaced by
// material-specific relief (crags and strata on rock, clumps on soil, dunes on sand), with every
// block center pinned to its own side so the surface stays within about half a block of the
// collision shape. Meshed with Surface Nets; vertices carry their nearest block's materials and
// the shader blends materials across triangles.
namespace mcrt::terrain {

constexpr int kPad = 3;                 // occupancy: section plus kPad blocks on each side
constexpr int kBorder = 16 + 2 * kPad;
// kPin: see-through full blocks (ice, glass, leaves); terrain faces them like open space, but the
// corners touching them stay put, since Minecraft culled the faces that would show a gap.
enum Occupancy : uint8_t { kOpen = 0, kSmooth = 1, kSolid = 2, kCover = 3, kPin = 4 };

// Local block index ((y << 8) | (z << 4) | x) a Minecraft quad belongs to, or -1 if it lies
// outside the section. Writes the quad's unit outward normal.
int quadBlock(const uint8_t* quad, float normal[3]);

constexpr uint32_t kAtlasMaterial = 254; // smoothed, but textured with the block's atlas sprite

// Atlas sprite of a block, taken from its own Minecraft quads.
struct SpriteRect {
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
};

// Drops quads belonging to smooth or cover blocks from one layer, remembering the biome tint of
// smooth blocks (their top face's color when present) in tints[4096] and their atlas sprite in
// sprites[4096].
void removeReplacedQuads(std::vector<uint8_t>& layer, const uint8_t* occupancy, uint32_t* tints,
                         SpriteRect* sprites);

// Terrain input (see mcrt_section_update): the kBorder^3 occupancy grid, then uint16 top|side<<8
// materials for the same grid, then uint32 RGBA8 grass colors for its kBorder^2 columns.
constexpr size_t kTerrainInputBytes = size_t(kBorder) * kBorder * kBorder * 3 + size_t(kBorder) * kBorder * 4;

// Appends the terrain quads (Minecraft's vertex layout, see pathtrace.slang) to `solid` and grass
// cards (alpha-tested, rooted on the surface wherever it is grass) to `cutout`. Section
// coordinates make the relief noise and the scattering continuous across sections.
void appendSmoothTerrain(std::vector<uint8_t>& solid, std::vector<uint8_t>& cutout, const uint8_t* terrainInput,
                         const SpriteRect* sprites, int sectionX, int sectionY, int sectionZ);

// Material flags as uploaded (bit 0 tinted, bit 1 vegetation card, bits 8-15 relief kind,
// bits 16-23 card kind).
void setMaterialFlags(uint32_t materialId, uint32_t flags);

enum CardKind : uint32_t { kCardGrass = 1, kCardBroadleaf = 2, kCardNeedle = 3 };
// Material id of the card texture of a kind, or 0 if none was uploaded.
uint32_t cardMaterial(uint32_t kind);
constexpr uint32_t kCardVertexFlag = 4; // vertex light word bit 2: vegetation card

// Extends water surface quads (material 255 in the light word) half a block under neighbouring
// smoothed terrain, so the shoreline becomes the curve where the smooth surface meets the water
// plane instead of the block grid's staircase.
void extendWaterUnderShore(std::vector<uint8_t>& translucentLayer, const uint8_t* occupancy);

} // namespace mcrt::terrain
