#include "shapes.h"

#include "mcrt/api.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mcrt::shapes {

namespace {

constexpr size_t kQuadBytes = size_t(MCRT_VERTEX_STRIDE) * 4;
constexpr float kPi = 3.14159265f;

// Foliage: a few large cards per leaf block break the cube silhouette into a ragged canopy.
constexpr int kCardsPerLeaf = 3;
constexpr float kCardHalfSize = 0.72f;
// Trunks: octagon slightly inside the block so neighbouring logs read as separate stems.
constexpr float kTrunkRadius = 0.44f;

struct Vec3 {
    float x, y, z;
};

uint32_t hash(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    v ^= v >> 16;
    return v;
}

float unit(uint32_t& state) {
    state = hash(state + 0x9e3779b9U);
    return float(state & 0xFFFFFF) / float(0x1000000);
}

void octEncode(Vec3 n, float out[2]) {
    float l1 = std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z);
    float x = n.x / l1, y = n.y / l1;
    if (n.z < 0.0f) {
        float ox = (1.0f - std::fabs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        float oy = (1.0f - std::fabs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = ox;
        y = oy;
    }
    out[0] = x;
    out[1] = y;
}

void putVertex(uint8_t* dst, Vec3 p, uint32_t color, float u, float v, uint32_t word) {
    float position[3] = {p.x, p.y, p.z};
    float uv[2] = {u, v};
    std::memcpy(dst, position, 12);
    std::memcpy(dst + 12, &color, 4);
    std::memcpy(dst + 16, uv, 8);
    std::memcpy(dst + 24, &word, 4);
}

uint8_t* appendQuad(std::vector<uint8_t>& layer) {
    size_t offset = layer.size();
    layer.resize(offset + kQuadBytes);
    return layer.data() + offset;
}

// Maps a local position in the trunk's frame (a = along axis, b/c = cross section) to the block.
Vec3 trunkPoint(uint32_t shape, float a, float b, float c) {
    switch (shape) {
    case kLogX: return {a, b, c};
    case kLogZ: return {b, c, a};
    default: return {b, a, c};
    }
}

void appendLeafCards(std::vector<uint8_t>& cutout, Vec3 base, uint32_t seed, const BlockLook& look) {
    const terrain::SpriteRect& s = look.side;
    for (int i = 0; i < kCardsPerLeaf; ++i) {
        uint32_t state = seed * 3 + uint32_t(i);
        float yaw = unit(state) * kPi + float(i) * kPi / kCardsPerLeaf;
        float tilt = (unit(state) - 0.5f) * 0.9f;
        Vec3 center{base.x + 0.5f + (unit(state) - 0.5f) * 0.3f, base.y + 0.5f + (unit(state) - 0.5f) * 0.3f,
                    base.z + 0.5f + (unit(state) - 0.5f) * 0.3f};
        Vec3 u{std::cos(yaw), 0.0f, std::sin(yaw)};
        // "Up" leaning by `tilt` around u.
        Vec3 v{-std::sin(yaw) * std::sin(tilt), std::cos(tilt), std::cos(yaw) * std::sin(tilt)};
        auto corner = [&](float su, float sv) {
            return Vec3{center.x + (u.x * su + v.x * sv) * kCardHalfSize, center.y + (u.y * su + v.y * sv) * kCardHalfSize,
                        center.z + (u.z * su + v.z * sv) * kCardHalfSize};
        };
        uint8_t* q = appendQuad(cutout);
        putVertex(q, corner(-1, -1), look.color, s.u0, s.v1, 0);
        putVertex(q + MCRT_VERTEX_STRIDE, corner(1, -1), look.color, s.u1, s.v1, 0);
        putVertex(q + 2 * MCRT_VERTEX_STRIDE, corner(1, 1), look.color, s.u1, s.v0, 0);
        putVertex(q + 3 * MCRT_VERTEX_STRIDE, corner(-1, 1), look.color, s.u0, s.v0, 0);
    }
}

void appendTrunk(std::vector<uint8_t>& solid, Vec3 base, uint32_t shape, uint32_t materials, const BlockLook& look) {
    // Side surface: smoothed-terrain vertices (bit 0) with radial normals, so the shader lights it
    // as a round trunk. With a PBR bark material it uses that; otherwise the block's own sprite.
    const uint32_t sideMaterial = (materials >> 8) & 0xFF;
    uint32_t word, color;
    if (sideMaterial != 0) {
        word = 1u | (sideMaterial << 16) | (sideMaterial << 24);
        color = 0xFFFFFFFFu;
    } else {
        auto unorm16 = [](float v) { return uint32_t(std::clamp(v, 0.0f, 1.0f) * 65535.0f + 0.5f); };
        word = 1u | 2u | (unorm16(look.side.u1 - look.side.u0) << 16);
        color = unorm16(look.side.u0) | (unorm16(look.side.v0) << 16);
    }
    auto at = [&](float a, float b, float c) {
        Vec3 p = trunkPoint(shape, a, b, c);
        return Vec3{base.x + p.x, base.y + p.y, base.z + p.z};
    };
    float cb[8], cc[8];
    for (int k = 0; k < 8; ++k) {
        float angle = (float(k) + 0.5f) * kPi / 4.0f;
        cb[k] = 0.5f + std::cos(angle) * kTrunkRadius;
        cc[k] = 0.5f + std::sin(angle) * kTrunkRadius;
    }
    for (int k = 0; k < 8; ++k) {
        int n = (k + 1) % 8;
        uint8_t* q = appendQuad(solid);
        const float a[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        const int idx[4] = {k, n, n, k};
        for (int v = 0; v < 4; ++v) {
            Vec3 radial = trunkPoint(shape, 0.0f, cb[idx[v]] - 0.5f, cc[idx[v]] - 0.5f);
            float len = std::sqrt(radial.x * radial.x + radial.y * radial.y + radial.z * radial.z);
            Vec3 normal{radial.x / len, radial.y / len, radial.z / len};
            float oct[2];
            octEncode(normal, oct);
            putVertex(q + v * MCRT_VERTEX_STRIDE, at(a[v], cb[idx[v]], cc[idx[v]]), color, oct[0], oct[1], word);
        }
    }
    // End caps (only where Minecraft showed the end): the octagon as three quads, textured with the
    // ring sprite mapped across the block face.
    const terrain::SpriteRect& s = look.cap;
    for (int end = 0; end < 2; ++end) {
        if ((look.capFaces & (1 << end)) == 0)
            continue;
        float a = end == 0 ? 1.0f : 0.0f;
        const int fan[3][4] = {{0, 1, 2, 3}, {0, 3, 4, 7}, {4, 5, 6, 7}};
        for (const auto& f : fan) {
            uint8_t* q = appendQuad(solid);
            for (int v = 0; v < 4; ++v) {
                int k = f[v];
                float u = s.u0 + (s.u1 - s.u0) * cb[k];
                float w = s.v0 + (s.v1 - s.v0) * cc[k];
                putVertex(q + v * MCRT_VERTEX_STRIDE, at(a, cb[k], cc[k]), 0xFFFFFFFFu, u, w, 0);
            }
        }
    }
}

} // namespace

void removeShapedQuads(std::vector<uint8_t>& layer, const uint32_t* blockMaterials, BlockLook* looks) {
    const size_t quads = layer.size() / kQuadBytes;
    size_t kept = 0;
    for (size_t q = 0; q < quads; ++q) {
        const uint8_t* quad = layer.data() + q * kQuadBytes;
        float normal[3];
        const int local = terrain::quadBlock(quad, normal);
        const uint32_t shape = local < 0 ? 0 : shapeOf(blockMaterials, local);
        if (shape == 0) {
            if (kept != q)
                std::memmove(layer.data() + kept * kQuadBytes, quad, kQuadBytes);
            ++kept;
            continue;
        }
        BlockLook& look = looks[local];
        look.present = true;
        std::memcpy(&look.color, quad + 12, 4);
        terrain::SpriteRect rect;
        float uv[2];
        std::memcpy(uv, quad + 16, sizeof(uv));
        rect = {uv[0], uv[1], uv[0], uv[1]};
        for (int v = 1; v < 4; ++v) {
            std::memcpy(uv, quad + v * MCRT_VERTEX_STRIDE + 16, sizeof(uv));
            rect.u0 = std::min(rect.u0, uv[0]);
            rect.v0 = std::min(rect.v0, uv[1]);
            rect.u1 = std::max(rect.u1, uv[0]);
            rect.v1 = std::max(rect.v1, uv[1]);
        }
        // End faces of a log point along its axis.
        const int axis = shape == kLogX ? 0 : (shape == kLogZ ? 2 : 1);
        if (shape != kLeaves && std::fabs(normal[axis]) > 0.7f) {
            look.cap = rect;
            look.capFaces |= normal[axis] > 0.0f ? 1 : 2;
        } else {
            look.side = rect;
        }
    }
    layer.resize(kept * kQuadBytes);
}

void appendShapes(std::vector<uint8_t>& solid, std::vector<uint8_t>& cutout, int32_t sectionX, int32_t sectionY,
                  int32_t sectionZ, const uint32_t* blockMaterials, const BlockLook* looks) {
    for (int local = 0; local < 4096; ++local) {
        const BlockLook& look = looks[local];
        if (!look.present)
            continue;
        const int x = local & 15, y = local >> 8, z = (local >> 4) & 15;
        const Vec3 base{float(x), float(y), float(z)};
        const uint32_t shape = shapeOf(blockMaterials, local);
        if (shape == kLeaves) {
            uint32_t seed = hash(uint32_t(sectionX * 16 + x) * 73856093u ^ uint32_t(sectionY * 16 + y) * 19349663u ^
                                 uint32_t(sectionZ * 16 + z) * 83492791u);
            appendLeafCards(cutout, base, seed, look);
        } else if (look.side.u1 > look.side.u0) {
            appendTrunk(solid, base, shape, blockMaterials[local], look);
        }
    }
}

} // namespace mcrt::shapes
