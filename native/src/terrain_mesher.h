#pragma once

#include <cstdint>
#include <vector>

// Smoothed natural terrain (visual only; collision stays blocky).
//
// Surface Nets on the block grid: every exposed face of a smooth block becomes one quad whose four
// vertices are the face's corners, each moved to a distance-weighted average of the inside/outside
// crossings in the 4x4x4 blocks around that corner. Flat ground stays exactly flat; steps become
// slopes and edges round off, moving no vertex by more than half a block on any axis.
namespace mcrt::terrain {

constexpr int kPad = 2;                 // occupancy: section plus kPad blocks on each side
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

// Appends the smoothed terrain quads (Minecraft's vertex layout, see pathtrace.slang) to a layer.
void appendSmoothTerrain(std::vector<uint8_t>& layer, const uint8_t* occupancy, const uint32_t* blockMaterials,
                         const uint32_t* tints, const SpriteRect* sprites);

// Extends water surface quads (material 255 in the light word) half a block under neighbouring
// smoothed terrain, so the shoreline becomes the curve where the smooth surface meets the water
// plane instead of the block grid's staircase.
void extendWaterUnderShore(std::vector<uint8_t>& translucentLayer, const uint8_t* occupancy);

} // namespace mcrt::terrain
