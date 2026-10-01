#pragma once

#include <cstdint>
#include <vector>

// Beveled edges for full-cube blocks (visual only).
//
// A perfectly sharp 90-degree edge is what most gives a block away. Every full-cube block's six
// face quads are replaced by faces inset from their convex edges, plus 45-degree bevel strips and
// corner triangles. An edge is convex when both faces meeting there are visible, which Minecraft's
// own face culling already tells us: faces it emitted are exposed, faces it culled are covered.
// Coplanar neighbours therefore join seamlessly (no grooves in walls or floors).
namespace mcrt::bevel {

constexpr float kBevelWidth = 0.0625f; // one texel of a 16x texture

// Rewrites one layer in place. Blocks whose quads are not exactly full axis-aligned faces (stairs,
// slabs, plants, overlays) are left untouched.
void bevelFullCubes(std::vector<uint8_t>& layer);

} // namespace mcrt::bevel
