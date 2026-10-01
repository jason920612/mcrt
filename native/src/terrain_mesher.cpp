#include "terrain_mesher.h"

#include "mcrt/api.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mcrt::terrain {

namespace {

constexpr size_t kQuadBytes = size_t(MCRT_VERTEX_STRIDE) * 4;
constexpr uint32_t kSmoothFlag = 1; // vertex light word bit 0: smoothed terrain vertex
constexpr uint32_t kAtlasFlag = 2;  // bit 1: textured from the atlas sprite packed into color/word

struct Vec3 {
    float x, y, z;
};

uint8_t occupancyAt(const uint8_t* occupancy, int x, int y, int z) {
    return occupancy[(y + kPad) * kBorder * kBorder + (z + kPad) * kBorder + (x + kPad)];
}

bool isInside(uint8_t type) {
    return type == kSmooth || type == kSolid;
}

// Octahedral encoding of a unit vector into two floats in [-1, 1].
void octEncode(const Vec3& n, float out[2]) {
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

struct Corner {
    Vec3 position;
    Vec3 normal;
    bool computed = false;
};

class CornerField {
public:
    // Heap storage: this runs on Minecraft's compile threads, whose stacks are not ours to fill.
    explicit CornerField(const uint8_t* occupancy) : occupancy_(occupancy), corners_(17 * 17 * 17) {}

    const Corner& at(int cx, int cy, int cz) {
        Corner& corner = corners_[(cy * 17 + cz) * 17 + cx];
        if (!corner.computed)
            compute(corner, cx, cy, cz);
        return corner;
    }

private:
    void compute(Corner& corner, int cx, int cy, int cz) {
        // Blocks cx-2..cx+1 (and likewise for y, z) around lattice point (cx, cy, cz):
        // types[j][k][i] is block (cx-2+i, cy-2+j, cz-2+k); centers sit at offset (i - 1.5, ...).
        uint8_t types[4][4][4];
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                for (int i = 0; i < 4; ++i)
                    types[j][k][i] = occupancyAt(occupancy_, cx - 2 + i, cy - 2 + j, cz - 2 + k);
        // Corners touching other solid blocks stay put so smoothed terrain meets them without gaps.
        bool pinned = false;
        for (int j = 1; j < 3; ++j)
            for (int k = 1; k < 3; ++k)
                for (int i = 1; i < 3; ++i)
                    pinned |= types[j][k][i] == kSolid || types[j][k][i] == kPin;

        // Weighted crossings: every block-center edge in the neighborhood that crosses the surface
        // contributes its midpoint, weighted by closeness to the corner.
        Vec3 sum{0, 0, 0};
        Vec3 gradient{0, 0, 0};
        float weightSum = 0.0f;
        auto edge = [&](int i0, int j0, int k0, int i1, int j1, int k1) {
            bool inside0 = isInside(types[j0][k0][i0]);
            if (inside0 == isInside(types[j1][k1][i1]))
                return;
            Vec3 mid{(i0 + i1) * 0.5f - 1.5f, (j0 + j1) * 0.5f - 1.5f, (k0 + k1) * 0.5f - 1.5f};
            float d2 = mid.x * mid.x + mid.y * mid.y + mid.z * mid.z;
            float w = std::exp(-d2 * 1.2f);
            sum.x += mid.x * w;
            sum.y += mid.y * w;
            sum.z += mid.z * w;
            weightSum += w;
            // Edge direction from inside to outside, for the normal.
            float s = inside0 ? 1.0f : -1.0f;
            gradient.x += float(i1 - i0) * s * w;
            gradient.y += float(j1 - j0) * s * w;
            gradient.z += float(k1 - k0) * s * w;
        };
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                for (int c = 0; c < 3; ++c) {
                    edge(c, a, b, c + 1, a, b); // along x
                    edge(a, c, b, a, c + 1, b); // along y
                    edge(a, b, c, a, b, c + 1); // along z
                }

        float len = std::sqrt(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z);
        corner.normal = len > 1e-4f ? Vec3{gradient.x / len, gradient.y / len, gradient.z / len} : Vec3{0, 1, 0};
        corner.position = {float(cx), float(cy), float(cz)};
        if (!pinned && weightSum > 0.0f) {
            auto clampHalf = [](float v) { return v < -0.5f ? -0.5f : (v > 0.5f ? 0.5f : v); };
            corner.position.x += clampHalf(sum.x / weightSum);
            corner.position.y += clampHalf(sum.y / weightSum);
            corner.position.z += clampHalf(sum.z / weightSum);
        }
        corner.computed = true;
    }

    const uint8_t* occupancy_;
    std::vector<Corner> corners_;
};

void writeVertex(uint8_t* dst, const Corner& corner, uint32_t color, uint32_t word) {
    float position[3] = {corner.position.x, corner.position.y, corner.position.z};
    float oct[2];
    octEncode(corner.normal, oct);
    std::memcpy(dst, position, 12);
    std::memcpy(dst + 12, &color, 4);
    std::memcpy(dst + 16, oct, 8);
    std::memcpy(dst + 24, &word, 4);
}

} // namespace

int quadBlock(const uint8_t* quad, float normal[3]) {
    float p[4][3];
    for (int v = 0; v < 4; ++v)
        std::memcpy(p[v], quad + v * MCRT_VERTEX_STRIDE, sizeof(p[v]));
    float e1[3], e2[3], c[3];
    for (int i = 0; i < 3; ++i) {
        e1[i] = p[1][i] - p[0][i];
        e2[i] = p[2][i] - p[0][i];
        c[i] = (p[0][i] + p[1][i] + p[2][i] + p[3][i]) * 0.25f;
    }
    float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-8f)
        return -1;
    for (int i = 0; i < 3; ++i)
        normal[i] = n[i] / len;
    // Minecraft quads wind counter-clockwise seen from outside, so -n points into the block.
    int b[3];
    for (int i = 0; i < 3; ++i)
        b[i] = static_cast<int>(std::floor(c[i] - normal[i] * 0.02f));
    if (b[0] < 0 || b[0] > 15 || b[1] < 0 || b[1] > 15 || b[2] < 0 || b[2] > 15)
        return -1;
    return (b[1] << 8) | (b[2] << 4) | b[0];
}

void removeReplacedQuads(std::vector<uint8_t>& layer, const uint8_t* occupancy, uint32_t* tints,
                         SpriteRect* sprites) {
    const size_t quads = layer.size() / kQuadBytes;
    size_t kept = 0;
    for (size_t q = 0; q < quads; ++q) {
        const uint8_t* quad = layer.data() + q * kQuadBytes;
        float normal[3];
        int local = quadBlock(quad, normal);
        uint8_t type = local < 0 ? kOpen
                                 : occupancyAt(occupancy, local & 15, local >> 8, (local >> 4) & 15);
        if (type == kSmooth || type == kCover) {
            if (type == kSmooth) {
                // The top face carries the biome tint (grass); other faces are tinted only as overlays.
                uint32_t color;
                std::memcpy(&color, quad + 12, 4);
                if (normal[1] > 0.7f || tints[local] == 0)
                    tints[local] = color;
                SpriteRect& sprite = sprites[local];
                if (sprite.u1 <= sprite.u0) {
                    float uv[2];
                    std::memcpy(uv, quad + 16, sizeof(uv));
                    sprite = {uv[0], uv[1], uv[0], uv[1]};
                    for (int v = 1; v < 4; ++v) {
                        std::memcpy(uv, quad + v * MCRT_VERTEX_STRIDE + 16, sizeof(uv));
                        sprite.u0 = std::min(sprite.u0, uv[0]);
                        sprite.v0 = std::min(sprite.v0, uv[1]);
                        sprite.u1 = std::max(sprite.u1, uv[0]);
                        sprite.v1 = std::max(sprite.v1, uv[1]);
                    }
                }
            }
            continue;
        }
        if (kept != q)
            std::memmove(layer.data() + kept * kQuadBytes, quad, kQuadBytes);
        ++kept;
    }
    layer.resize(kept * kQuadBytes);
}

void appendSmoothTerrain(std::vector<uint8_t>& layer, const uint8_t* occupancy, const uint32_t* blockMaterials,
                         const uint32_t* tints, const SpriteRect* sprites) {
    CornerField field(occupancy);
    // Per face direction: neighbor offset and the 4 face corners (offsets from the block's min
    // corner) in perimeter order.
    struct Face {
        int dx, dy, dz;
        int corners[4][3];
    };
    static const Face faces[6] = {
        {+1, 0, 0, {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}},
        {-1, 0, 0, {{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}}},
        {0, +1, 0, {{0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}}},
        {0, -1, 0, {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}},
        {0, 0, +1, {{0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}},
        {0, 0, -1, {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}},
    };

    for (int y = 0; y < 16; ++y)
        for (int z = 0; z < 16; ++z)
            for (int x = 0; x < 16; ++x) {
                if (occupancyAt(occupancy, x, y, z) != kSmooth)
                    continue;
                const int local = (y << 8) | (z << 4) | x;
                const uint32_t materials = blockMaterials ? blockMaterials[local] : 0;
                // Shader picks top vs side material from the smoothed normal.
                uint32_t word = kSmoothFlag | ((materials & 0xFF) << 16) | (((materials >> 8) & 0xFF) << 24);
                uint32_t color = tints[local] ? tints[local] : 0xFFFFFFFFu;
                if ((materials & 0xFF) == kAtlasMaterial) {
                    // Atlas sprite: origin in color (2 x 16-bit normalized), width in the word's high half.
                    const SpriteRect& sprite = sprites[local];
                    auto unorm16 = [](float v) { return uint32_t(std::clamp(v, 0.0f, 1.0f) * 65535.0f + 0.5f); };
                    color = unorm16(sprite.u0) | (unorm16(sprite.v0) << 16);
                    word = kSmoothFlag | kAtlasFlag | (unorm16(sprite.u1 - sprite.u0) << 16);
                }
                for (const Face& face : faces) {
                    uint8_t neighbor = occupancyAt(occupancy, x + face.dx, y + face.dy, z + face.dz);
                    if (neighbor != kOpen && neighbor != kCover && neighbor != kPin)
                        continue;
                    const size_t offset = layer.size();
                    layer.resize(offset + kQuadBytes);
                    for (int v = 0; v < 4; ++v) {
                        const Corner& corner =
                            field.at(x + face.corners[v][0], y + face.corners[v][1], z + face.corners[v][2]);
                        writeVertex(layer.data() + offset + v * MCRT_VERTEX_STRIDE, corner, color, word);
                    }
                }
            }
}

void extendWaterUnderShore(std::vector<uint8_t>& translucentLayer, const uint8_t* occupancy) {
    constexpr uint32_t kWaterMaterial = 255;
    constexpr float kReach = 0.5f;
    const size_t quads = translucentLayer.size() / kQuadBytes;
    for (size_t q = 0; q < quads; ++q) {
        uint8_t* quad = translucentLayer.data() + q * kQuadBytes;
        uint32_t word;
        std::memcpy(&word, quad + 24, 4);
        if (((word >> 16) & 0xFF) != kWaterMaterial)
            continue;
        float normal[3];
        const int local = quadBlock(quad, normal);
        if (local < 0 || std::fabs(normal[1]) < 0.9f)
            continue; // only the (near) horizontal surface
        const int bx = local & 15, by = local >> 8, bz = (local >> 4) & 15;
        const bool reach[4] = {occupancyAt(occupancy, bx + 1, by, bz) == kSmooth,  // +x
                               occupancyAt(occupancy, bx - 1, by, bz) == kSmooth,  // -x
                               occupancyAt(occupancy, bx, by, bz + 1) == kSmooth,  // +z
                               occupancyAt(occupancy, bx, by, bz - 1) == kSmooth}; // -z
        if (!reach[0] && !reach[1] && !reach[2] && !reach[3])
            continue;
        for (int v = 0; v < 4; ++v) {
            float p[3];
            std::memcpy(p, quad + v * MCRT_VERTEX_STRIDE, 12);
            if (reach[0] && std::fabs(p[0] - (bx + 1)) < 1e-3f) p[0] += kReach;
            if (reach[1] && std::fabs(p[0] - bx) < 1e-3f) p[0] -= kReach;
            if (reach[2] && std::fabs(p[2] - (bz + 1)) < 1e-3f) p[2] += kReach;
            if (reach[3] && std::fabs(p[2] - bz) < 1e-3f) p[2] -= kReach;
            std::memcpy(quad + v * MCRT_VERTEX_STRIDE, p, 12);
        }
    }
}

} // namespace mcrt::terrain
