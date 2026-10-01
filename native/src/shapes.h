#pragma once

#include "terrain_mesher.h"

#include <cstdint>
#include <vector>

// Replacement geometry for blocks whose cube shape gives Minecraft away (visual only):
// leaves become crossed, alpha-cut foliage cards; logs become octagonal trunks.
// The shape of a block comes from bits 24-31 of its blockMaterials entry (SectionScanner.SHAPE_*).
namespace mcrt::shapes {

constexpr uint32_t kLeaves = 1;
constexpr uint32_t kLogX = 2;
constexpr uint32_t kLogY = 3;
constexpr uint32_t kLogZ = 4;

inline uint32_t shapeOf(const uint32_t* blockMaterials, int local) {
    return blockMaterials ? blockMaterials[local] >> 24 : 0;
}

// What a reshaped block looked like in Minecraft's own mesh.
struct BlockLook {
    bool present = false;          // Minecraft emitted at least one quad for it (it is visible)
    uint32_t color = 0xFFFFFFFFu;  // vertex color (biome tint) of its quads
    terrain::SpriteRect side;      // atlas sprite of a side face (bark, leaves)
    terrain::SpriteRect cap;       // atlas sprite of an end face (log rings)
    uint8_t capFaces = 0;          // bit 0: positive end exposed, bit 1: negative end exposed
};

// Drops the quads of reshaped blocks from one layer, recording their look in looks[4096].
void removeShapedQuads(std::vector<uint8_t>& layer, const uint32_t* blockMaterials, BlockLook* looks);

// Appends foliage cards to `cutout` and trunks to `solid` for every reshaped, visible block.
// sectionX/Y/Z seed the per-block randomness, so cards stay put across recompiles.
void appendShapes(std::vector<uint8_t>& solid, std::vector<uint8_t>& cutout, int32_t sectionX, int32_t sectionY,
                  int32_t sectionZ, const uint32_t* blockMaterials, const BlockLook* looks);

} // namespace mcrt::shapes
