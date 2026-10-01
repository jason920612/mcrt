#include "terrain_mesher.h"

#include "mcrt/api.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace mcrt::terrain {

namespace {

constexpr size_t kQuadBytes = size_t(MCRT_VERTEX_STRIDE) * 4;
constexpr uint32_t kSmoothFlag = 1; // vertex light word bit 0: smoothed terrain vertex
constexpr uint32_t kAtlasFlag = 2;  // bit 1: textured from the atlas sprite packed into color/word
constexpr uint32_t kCardFlag = 4;   // bit 2: vegetation card (material texture with coverage alpha)
constexpr uint32_t kTerrainFlag = 8; // bit 3: natural terrain (gets macro detail in the shader)

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

// ---------------------------------------------------------------------------------------------
// Terrain as a density field. Blocks are only markers of where ground is and what it is made of:
// the visible surface is the zero set of a smoothed, sculpted field sampled every half block, so
// it is neither the block grid's staircase nor a uniformly melted blob.
//
//   f(p) = kernel-weighted vote of the blocks around p (+1 solid, -1 open)       -> soft landforms
//        + relief noise by material (rock: ridges and strata, soil: clumps, ...) -> character
//   then each block's center sample is pinned to the block's side of zero       -> gameplay shape
//
// The pins keep every block (and every one-block hole) visible, so the surface never strays more
// than about half a block from the collision shape.

constexpr float kSpacing = 0.5f;          // field sample spacing (blocks)
constexpr int kSampleMin = -2;            // sample indices kSampleMin..kSampleMax per axis
constexpr int kSampleMax = 33;
constexpr int kSamples = kSampleMax - kSampleMin + 1;
constexpr float kKernelRadius = 1.6f;     // blocks
constexpr float kPinMargin = 0.2f;        // field value forced at block centers

enum Relief : uint8_t { kReliefNone = 0, kReliefSoil = 1, kReliefRock = 2, kReliefSand = 3, kReliefSnow = 4 };

std::atomic<uint32_t> g_materialFlags[256];
std::atomic<uint32_t> g_cardMaterials[4];

constexpr uint32_t kMaterialTinted = 1;
constexpr uint32_t kMaterialCard = 2;

uint32_t hash3(int x, int y, int z) {
    uint32_t h = uint32_t(x) * 0x8da6b343u ^ uint32_t(y) * 0xd8163841u ^ uint32_t(z) * 0xcb1ab31fu;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

float lattice(int x, int y, int z) {
    return float(hash3(x, y, z) & 0xFFFFFF) * (1.0f / 16777215.0f);
}

float valueNoise(float x, float y, float z) {
    const float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    const int ix = int(fx), iy = int(fy), iz = int(fz);
    float tx = x - fx, ty = y - fy, tz = z - fz;
    tx = tx * tx * (3.0f - 2.0f * tx);
    ty = ty * ty * (3.0f - 2.0f * ty);
    tz = tz * tz * (3.0f - 2.0f * tz);
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    const float c00 = lerp(lattice(ix, iy, iz), lattice(ix + 1, iy, iz), tx);
    const float c10 = lerp(lattice(ix, iy + 1, iz), lattice(ix + 1, iy + 1, iz), tx);
    const float c01 = lerp(lattice(ix, iy, iz + 1), lattice(ix + 1, iy, iz + 1), tx);
    const float c11 = lerp(lattice(ix, iy + 1, iz + 1), lattice(ix + 1, iy + 1, iz + 1), tx);
    return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
}

// Relief displacement in field units (about 0.75 blocks per unit at a surface), roughly -1..1.
float reliefNoise(uint8_t kind, float x, float y, float z) {
    switch (kind) {
    case kReliefRock: {
        // Ridged noise for crags, plus horizontal strata that make ledges.
        float ridge = 1.0f - std::fabs(2.0f * valueNoise(x * 0.28f, y * 0.28f, z * 0.28f) - 1.0f);
        float detail = valueNoise(x * 0.9f + 31.0f, y * 0.9f, z * 0.9f);
        float strata = valueNoise(x * 0.06f, y * 0.85f + 7.0f, z * 0.06f);
        strata = std::floor(strata * 4.0f) / 4.0f; // stepped bands
        return (ridge * 0.5f + detail * 0.25f + strata * 0.25f) * 2.0f - 1.0f;
    }
    case kReliefSoil:
        return (valueNoise(x * 0.35f, y * 0.35f, z * 0.35f) * 0.65f
              + valueNoise(x * 1.1f + 13.0f, y * 1.1f, z * 1.1f) * 0.35f) * 2.0f - 1.0f;
    case kReliefSand: {
        // Wind ripples and dunes: stretched along one direction.
        float dune = valueNoise(x * 0.12f + z * 0.05f, y * 0.2f, z * 0.3f);
        return dune * 2.0f - 1.0f;
    }
    case kReliefSnow:
        return valueNoise(x * 0.25f, y * 0.25f, z * 0.25f) * 2.0f - 1.0f;
    default:
        return 0.0f;
    }
}

float reliefAmplitude(uint8_t kind) {
    switch (kind) {
    case kReliefRock: return 0.55f;
    case kReliefSoil: return 0.22f;
    case kReliefSand: return 0.12f;
    case kReliefSnow: return 0.1f;
    default: return 0.0f;
    }
}

struct TerrainInput {
    const uint8_t* occupancy;
    const uint16_t* materials; // top | side << 8, padded grid
    const uint32_t* columnTints;

    int index(int x, int y, int z) const { return (y + kPad) * kBorder * kBorder + (z + kPad) * kBorder + (x + kPad); }
    bool inGrid(int x, int y, int z) const {
        return x >= -kPad && x < 16 + kPad && y >= -kPad && y < 16 + kPad && z >= -kPad && z < 16 + kPad;
    }
    uint8_t type(int x, int y, int z) const { return inGrid(x, y, z) ? occupancy[index(x, y, z)] : kOpen; }
    uint16_t material(int x, int y, int z) const { return inGrid(x, y, z) ? materials[index(x, y, z)] : 0; }
    uint32_t tint(int x, int z) const {
        x = std::clamp(x, -kPad, 15 + kPad);
        z = std::clamp(z, -kPad, 15 + kPad);
        return columnTints[(z + kPad) * kBorder + (x + kPad)];
    }
};

uint8_t reliefOf(uint16_t materials) {
    uint32_t top = materials & 0xFF, side = materials >> 8;
    auto kind = [](uint32_t id) -> uint8_t {
        if (id == kAtlasMaterial) return kReliefRock; // ores sit in stone
        return id ? uint8_t((g_materialFlags[id].load(std::memory_order_relaxed) >> 8) & 0xFF) : uint8_t(kReliefNone);
    };
    uint8_t topKind = kind(top);
    if (topKind == kReliefSnow || topKind == kReliefSand)
        return topKind; // a covering layer shapes the surface
    uint8_t sideKind = kind(side);
    return sideKind != kReliefNone ? sideKind : topKind;
}

bool fieldInside(uint8_t type) {
    return type == kSmooth || type == kSolid || type == kPin;
}

class DensityField {
public:
    DensityField(const TerrainInput& in, int originX, int originY, int originZ)
        : in_(in), values_(size_t(kSamples) * kSamples * kSamples) {
        for (int j = kSampleMin; j <= kSampleMax; ++j)
            for (int k = kSampleMin; k <= kSampleMax; ++k)
                for (int i = kSampleMin; i <= kSampleMax; ++i)
                    at(i, j, k) = evaluate(i, j, k, originX, originY, originZ);
    }

    float& at(int i, int j, int k) {
        return values_[(size_t(j - kSampleMin) * kSamples + (k - kSampleMin)) * kSamples + (i - kSampleMin)];
    }
    float value(int i, int j, int k) const {
        return values_[(size_t(j - kSampleMin) * kSamples + (k - kSampleMin)) * kSamples + (i - kSampleMin)];
    }

    // Field gradient at a sample (central differences, one-sided at the grid's edge).
    void gradient(int i, int j, int k, float g[3]) const {
        auto diff = [&](int di, int dj, int dk) {
            int i0 = std::max(i - di, kSampleMin), i1 = std::min(i + di, kSampleMax);
            int j0 = std::max(j - dj, kSampleMin), j1 = std::min(j + dj, kSampleMax);
            int k0 = std::max(k - dk, kSampleMin), k1 = std::min(k + dk, kSampleMax);
            float span = float((i1 - i0) + (j1 - j0) + (k1 - k0)) * kSpacing;
            return (value(i1, j1, k1) - value(i0, j0, k0)) / std::max(span, 1e-6f);
        };
        g[0] = diff(1, 0, 0);
        g[1] = diff(0, 1, 0);
        g[2] = diff(0, 0, 1);
    }

private:
    float evaluate(int i, int j, int k, int ox, int oy, int oz) const {
        const float px = i * kSpacing, py = j * kSpacing, pz = k * kSpacing;
        // Blocks whose centers lie within the kernel radius.
        const int bx0 = int(std::ceil(px - kKernelRadius - 0.5f)), bx1 = int(std::floor(px + kKernelRadius - 0.5f));
        const int by0 = int(std::ceil(py - kKernelRadius - 0.5f)), by1 = int(std::floor(py + kKernelRadius - 0.5f));
        const int bz0 = int(std::ceil(pz - kKernelRadius - 0.5f)), bz1 = int(std::floor(pz + kKernelRadius - 0.5f));
        const float r2max = kKernelRadius * kKernelRadius;
        float vote = 0.0f, weightSum = 0.0f;
        float reliefWeight[5] = {0, 0, 0, 0, 0};
        for (int by = by0; by <= by1; ++by)
            for (int bz = bz0; bz <= bz1; ++bz)
                for (int bx = bx0; bx <= bx1; ++bx) {
                    const float dx = bx + 0.5f - px, dy = by + 0.5f - py, dz = bz + 0.5f - pz;
                    const float r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= r2max)
                        continue;
                    float t = 1.0f - r2 / r2max;
                    const float w = t * t * t;
                    const uint8_t type = in_.type(bx, by, bz);
                    vote += fieldInside(type) ? w : -w;
                    weightSum += w;
                    if (type == kSmooth)
                        reliefWeight[reliefOf(in_.material(bx, by, bz))] += w;
                }
        float f = weightSum > 0.0f ? vote / weightSum : -1.0f;
        if (weightSum > 0.0f) {
            const float wx = px + ox, wy = py + oy, wz = pz + oz; // absolute: identical across sections
            for (uint8_t kind = 1; kind < 5; ++kind)
                if (reliefWeight[kind] > 0.0f)
                    f += reliefWeight[kind] / weightSum * reliefAmplitude(kind) * reliefNoise(kind, wx, wy, wz);
        }
        return pin(i, j, k, f);
    }

    // Gameplay shape: centers of blocks keep their side of the surface, and smooth ground that
    // touches a non-smooth solid block (whose face toward it Minecraft culled) covers that face.
    // Sample index s sits at s/2 blocks: odd s is the center of block (s-1)/2, even s the boundary
    // between blocks s/2-1 and s/2.
    float pin(int i, int j, int k, float f) const {
        if ((i & 1) && (j & 1) && (k & 1))
            return fieldInside(in_.type((i - 1) / 2, (j - 1) / 2, (k - 1) / 2)) ? std::max(f, kPinMargin)
                                                                               : std::min(f, -kPinMargin);
        auto touching = [](int s, int out[2]) {
            if (s & 1) {
                out[0] = (s - 1) / 2;
                return 1;
            }
            out[0] = s / 2 - 1;
            out[1] = s / 2;
            return 2;
        };
        int xs[2], ys[2], zs[2];
        const int nx = touching(i, xs), ny = touching(j, ys), nz = touching(k, zs);
        bool smooth = false, rigid = false;
        for (int a = 0; a < ny; ++a)
            for (int b = 0; b < nz; ++b)
                for (int c = 0; c < nx; ++c) {
                    const uint8_t type = in_.type(xs[c], ys[a], zs[b]);
                    smooth |= type == kSmooth;
                    rigid |= type == kSolid || type == kPin;
                }
        return smooth && rigid ? std::max(f, kPinMargin) : f;
    }

    const TerrainInput& in_;
    std::vector<float> values_;
};

// Vertex material: the nearest smooth block's top/side materials (deterministic in absolute
// coordinates, so both sections sharing a vertex agree). Returns the block via bx/by/bz.
bool nearestSmoothBlock(const TerrainInput& in, const float v[3], bool allowAtlas, int& bx, int& by, int& bz) {
    const int fx = int(std::floor(v[0])), fy = int(std::floor(v[1])), fz = int(std::floor(v[2]));
    float best = 1e9f;
    bool found = false;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz)
            for (int dx = -1; dx <= 1; ++dx) {
                const int x = fx + dx, y = fy + dy, z = fz + dz;
                if (in.type(x, y, z) != kSmooth)
                    continue;
                if (!allowAtlas && (in.material(x, y, z) & 0xFF) == kAtlasMaterial)
                    continue;
                const float ex = x + 0.5f - v[0], ey = y + 0.5f - v[1], ez = z + 0.5f - v[2];
                const float d = ex * ex + ey * ey + ez * ez;
                if (d < best - 1e-5f) {
                    best = d;
                    bx = x;
                    by = y;
                    bz = z;
                    found = true;
                }
            }
    return found;
}

struct CellVertex {
    float position[3];
    float normal[3];
    bool computed = false;
    bool valid = false;
};

} // namespace

void setMaterialFlags(uint32_t materialId, uint32_t flags) {
    if (materialId >= 256)
        return;
    g_materialFlags[materialId].store(flags, std::memory_order_relaxed);
    const uint32_t kind = (flags >> 16) & 0xFF;
    if ((flags & kMaterialCard) && kind < 4)
        g_cardMaterials[kind].store(materialId, std::memory_order_relaxed);
}

uint32_t cardMaterial(uint32_t kind) {
    return kind < 4 ? g_cardMaterials[kind].load(std::memory_order_relaxed) : 0;
}

void appendSmoothTerrain(std::vector<uint8_t>& layer, std::vector<uint8_t>& cutout, const uint8_t* terrainInput,
                         const SpriteRect* sprites, int sectionX, int sectionY, int sectionZ) {
    constexpr size_t kCells = size_t(kBorder) * kBorder * kBorder;
    TerrainInput in{terrainInput, reinterpret_cast<const uint16_t*>(terrainInput + kCells),
                    reinterpret_cast<const uint32_t*>(terrainInput + kCells * 3)};
    // Heap storage: this runs on Minecraft's compile threads, whose stacks are not ours to fill.
    DensityField field(in, sectionX * 16, sectionY * 16, sectionZ * 16);

    // Surface Nets: one vertex per sign-changing cell, at the mean of its edge crossings.
    constexpr int kCellsPerAxis = kSamples - 1; // cells kSampleMin..kSampleMax-1
    std::vector<CellVertex> cells(size_t(kCellsPerAxis) * kCellsPerAxis * kCellsPerAxis);
    auto cellVertex = [&](int ci, int cj, int ck) -> const CellVertex& {
        CellVertex& cell =
            cells[(size_t(cj - kSampleMin) * kCellsPerAxis + (ck - kSampleMin)) * kCellsPerAxis + (ci - kSampleMin)];
        if (cell.computed)
            return cell;
        cell.computed = true;
        // Corner c: x offset bit 0, z offset bit 1, y offset bit 2.
        float corner[8];
        for (int c = 0; c < 8; ++c)
            corner[c] = field.value(ci + (c & 1), cj + ((c >> 2) & 1), ck + ((c >> 1) & 1));
        static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                         {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        static const int axisBit[3] = {1, 4, 2};
        float sum[3] = {0, 0, 0};
        int crossings = 0;
        for (const auto& e : edges) {
            const float a = corner[e[0]], b = corner[e[1]];
            if ((a > 0.0f) == (b > 0.0f))
                continue;
            const float t = a / (a - b);
            for (int axis = 0; axis < 3; ++axis) {
                const float pa = float((e[0] & axisBit[axis]) != 0), pb = float((e[1] & axisBit[axis]) != 0);
                sum[axis] += pa + (pb - pa) * t;
            }
            ++crossings;
        }
        if (crossings == 0)
            return cell;
        const float local[3] = {sum[0] / crossings, sum[1] / crossings, sum[2] / crossings};
        cell.position[0] = (ci + local[0]) * kSpacing;
        cell.position[1] = (cj + local[1]) * kSpacing;
        cell.position[2] = (ck + local[2]) * kSpacing;
        // Normal: trilinear blend of the corner gradients, pointing out of the ground.
        float g[3] = {0, 0, 0};
        for (int c = 0; c < 8; ++c) {
            const float wx = (c & 1) ? local[0] : 1.0f - local[0];
            const float wy = ((c >> 2) & 1) ? local[1] : 1.0f - local[1];
            const float wz = ((c >> 1) & 1) ? local[2] : 1.0f - local[2];
            float cg[3];
            field.gradient(ci + (c & 1), cj + ((c >> 2) & 1), ck + ((c >> 1) & 1), cg);
            for (int axis = 0; axis < 3; ++axis)
                g[axis] += cg[axis] * wx * wy * wz;
        }
        const float len = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (len > 1e-6f) {
            for (int axis = 0; axis < 3; ++axis)
                cell.normal[axis] = -g[axis] / len;
        } else {
            cell.normal[0] = 0.0f;
            cell.normal[1] = 1.0f;
            cell.normal[2] = 0.0f;
        }
        cell.valid = true;
        return cell;
    };

    auto unorm16 = [](float v) { return uint32_t(std::clamp(v, 0.0f, 1.0f) * 65535.0f + 0.5f); };
    const float ox = float(sectionX * 16), oy = float(sectionY * 16), oz = float(sectionZ * 16);
    // Materials are looked up at a noise-displaced point, so material boundaries meander through
    // the terrain instead of following the block grid.
    auto materialProbe = [&](const float p[3], float probe[3]) {
        const float wx = p[0] + ox, wy = p[1] + oy, wz = p[2] + oz;
        probe[0] = p[0] + (valueNoise(wx * 0.4f, wy * 0.4f, wz * 0.4f + 5.3f) - 0.5f) * 1.7f;
        probe[1] = p[1] + (valueNoise(wx * 0.4f + 9.1f, wy * 0.4f, wz * 0.4f) - 0.5f) * 0.6f;
        probe[2] = p[2] + (valueNoise(wx * 0.4f, wy * 0.4f + 3.7f, wz * 0.4f) - 0.5f) * 1.7f;
    };
    auto writeCell = [&](uint8_t* dst, const CellVertex& cell) {
        uint32_t word = kSmoothFlag | kTerrainFlag;
        uint32_t color = 0xFFFFFFFFu;
        int bx = 0, by = 0, bz = 0;
        float probe[3];
        materialProbe(cell.position, probe);
        if (nearestSmoothBlock(in, probe, true, bx, by, bz) || nearestSmoothBlock(in, cell.position, true, bx, by, bz)) {
            uint16_t materials = in.material(bx, by, bz);
            const bool inSection = bx >= 0 && bx < 16 && by >= 0 && by < 16 && bz >= 0 && bz < 16;
            if ((materials & 0xFF) == kAtlasMaterial && inSection && sprites) {
                // Atlas sprite: origin in color (2 x 16-bit normalized), width in the word's high half.
                const SpriteRect& sprite = sprites[(by << 8) | (bz << 4) | bx];
                color = unorm16(sprite.u0) | (unorm16(sprite.v0) << 16);
                word = kSmoothFlag | kAtlasFlag | kTerrainFlag | (unorm16(sprite.u1 - sprite.u0) << 16);
            } else {
                if ((materials & 0xFF) == kAtlasMaterial && nearestSmoothBlock(in, probe, false, bx, by, bz))
                    materials = in.material(bx, by, bz);
                if ((materials & 0xFF) == kAtlasMaterial)
                    materials = 0;
                word = kSmoothFlag | kTerrainFlag | (uint32_t(materials & 0xFF) << 16) | (uint32_t(materials >> 8) << 24);
                color = in.tint(bx, bz);
            }
        }
        float oct[2];
        octEncode(Vec3{cell.normal[0], cell.normal[1], cell.normal[2]}, oct);
        std::memcpy(dst, cell.position, 12);
        std::memcpy(dst + 12, &color, 4);
        std::memcpy(dst + 16, oct, 8);
        std::memcpy(dst + 24, &word, 4);
    };

    // A sample belongs to terrain if a smooth block touches it.
    auto touchesSmooth = [&](int i, int j, int k) {
        auto range = [](int s, int& lo, int& hi) {
            if (s & 1) {
                lo = hi = (s - 1) / 2;
            } else {
                lo = s / 2 - 1;
                hi = s / 2;
            }
        };
        int x0, x1, y0, y1, z0, z1;
        range(i, x0, x1);
        range(j, y0, y1);
        range(k, z0, z1);
        for (int y = y0; y <= y1; ++y)
            for (int z = z0; z <= z1; ++z)
                for (int x = x0; x <= x1; ++x)
                    if (in.type(x, y, z) == kSmooth)
                        return true;
        return false;
    };

    // Quads: one per sign-changing sample edge owned by this section (start sample in [0, 32)).
    for (int axis = 0; axis < 3; ++axis)
        for (int j = 0; j < 32; ++j)
            for (int k = 0; k < 32; ++k)
                for (int i = 0; i < 32; ++i) {
                    int s0[3] = {i, j, k};
                    int s1[3] = {i, j, k};
                    s1[axis] += 1;
                    const float f0 = field.value(s0[0], s0[1], s0[2]);
                    const float f1 = field.value(s1[0], s1[1], s1[2]);
                    if ((f0 > 0.0f) == (f1 > 0.0f))
                        continue;
                    if (!touchesSmooth(s0[0], s0[1], s0[2]) && !touchesSmooth(s1[0], s1[1], s1[2]))
                        continue;
                    // The four cells around the edge.
                    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                    static const int du[4] = {-1, 0, 0, -1}, dv[4] = {-1, -1, 0, 0};
                    const CellVertex* q[4];
                    bool ok = true;
                    for (int n = 0; n < 4; ++n) {
                        int c[3] = {s0[0], s0[1], s0[2]};
                        c[u] += du[n];
                        c[v] += dv[n];
                        q[n] = &cellVertex(c[0], c[1], c[2]);
                        ok = ok && q[n]->valid;
                    }
                    if (!ok)
                        continue;
                    // Wind counter-clockwise seen from outside (inside -> outside along the axis).
                    float d1[3], d2[3];
                    for (int a = 0; a < 3; ++a) {
                        d1[a] = q[2]->position[a] - q[0]->position[a];
                        d2[a] = q[3]->position[a] - q[1]->position[a];
                    }
                    const float cross[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
                                            d1[0] * d2[1] - d1[1] * d2[0]};
                    const float outward = f0 > 0.0f ? 1.0f : -1.0f;
                    const bool flip = cross[axis] * outward < 0.0f;
                    const size_t offset = layer.size();
                    layer.resize(offset + kQuadBytes);
                    for (int n = 0; n < 4; ++n)
                        writeCell(layer.data() + offset + n * MCRT_VERTEX_STRIDE, *q[flip ? 3 - n : n]);
                }

    // Grass: alpha-tested cards rooted on the surface wherever the ground is grass and gentle.
    const uint32_t card = cardMaterial(kCardGrass);
    if (card == 0)
        return;
    // Field value by trilinear interpolation between samples.
    auto fieldAt = [&](float x, float y, float z) {
        const float gx = x / kSpacing, gy = y / kSpacing, gz = z / kSpacing;
        const int i = std::clamp(int(std::floor(gx)), kSampleMin, kSampleMax - 1);
        const int j = std::clamp(int(std::floor(gy)), kSampleMin, kSampleMax - 1);
        const int k = std::clamp(int(std::floor(gz)), kSampleMin, kSampleMax - 1);
        const float tx = std::clamp(gx - i, 0.0f, 1.0f), ty = std::clamp(gy - j, 0.0f, 1.0f), tz = std::clamp(gz - k, 0.0f, 1.0f);
        auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
        const float c00 = lerp(field.value(i, j, k), field.value(i + 1, j, k), tx);
        const float c10 = lerp(field.value(i, j + 1, k), field.value(i + 1, j + 1, k), tx);
        const float c01 = lerp(field.value(i, j, k + 1), field.value(i + 1, j, k + 1), tx);
        const float c11 = lerp(field.value(i, j + 1, k + 1), field.value(i + 1, j + 1, k + 1), tx);
        return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
    };
    auto isGrass = [&](uint16_t materials) {
        const uint32_t top = materials & 0xFF;
        if (top == 0 || top == kAtlasMaterial)
            return false;
        const uint32_t flags = g_materialFlags[top].load(std::memory_order_relaxed);
        return (flags & kMaterialTinted) != 0 && ((flags >> 8) & 0xFF) == kReliefSoil;
    };
    auto rand01 = [](uint32_t& state) {
        state = state * 1664525u + 1013904223u;
        return float(state >> 8) * (1.0f / 16777216.0f);
    };
    for (int y = 0; y < 16; ++y)
        for (int z = 0; z < 16; ++z)
            for (int x = 0; x < 16; ++x) {
                if (in.type(x, y, z) != kSmooth || !isGrass(in.material(x, y, z)))
                    continue;
                const uint8_t above = in.type(x, y + 1, z);
                if (above != kOpen)
                    continue;
                const int ax = x + sectionX * 16, ay = y + sectionY * 16, az = z + sectionZ * 16;
                uint32_t state = hash3(ax, ay, az);
                // Meadows are patchy: denser in some places, thin in others.
                const float density = valueNoise(ax * 0.15f, ay * 0.15f, az * 0.15f);
                const int count = int(3.0f + density * 5.0f);
                for (int c = 0; c < count; ++c) {
                    const float px = x + rand01(state), pz = z + rand01(state);
                    // Find the ground: the field crosses zero going up through this column.
                    float lo = y - 0.6f, hi = y + 1.6f;
                    if (!(fieldAt(px, lo, pz) > 0.0f) || fieldAt(px, hi, pz) > 0.0f)
                        continue;
                    for (int it = 0; it < 12; ++it) {
                        const float mid = 0.5f * (lo + hi);
                        (fieldAt(px, mid, pz) > 0.0f ? lo : hi) = mid;
                    }
                    const float py = 0.5f * (lo + hi);
                    // Only on gentle ground, and only where the meandering material lookup says grass.
                    const float e = 0.25f;
                    const float gx = fieldAt(px + e, py, pz) - fieldAt(px - e, py, pz);
                    const float gy = fieldAt(px, py + e, pz) - fieldAt(px, py - e, pz);
                    const float gz = fieldAt(px, py, pz + e) - fieldAt(px, py, pz - e);
                    const float glen = std::sqrt(gx * gx + gy * gy + gz * gz);
                    if (glen < 1e-6f || -gy / glen < 0.75f)
                        continue;
                    const float point[3] = {px, py, pz};
                    float probe[3];
                    materialProbe(point, probe);
                    int bx, by, bz;
                    if (!nearestSmoothBlock(in, probe, false, bx, by, bz) || !isGrass(in.material(bx, by, bz)))
                        continue;

                    const float yaw = rand01(state) * 6.2831853f;
                    const float width = 0.7f + rand01(state) * 0.6f;
                    const float height = (0.3f + rand01(state) * 0.35f) * (0.7f + density * 0.6f);
                    const float dirX = std::cos(yaw) * width * 0.5f, dirZ = std::sin(yaw) * width * 0.5f;
                    const float leanX = (rand01(state) - 0.5f) * 0.2f, leanZ = (rand01(state) - 0.5f) * 0.2f;
                    const float u0 = rand01(state), u1 = u0 + width * 0.6f;
                    const float base = py - 0.06f; // sink the roots slightly into the ground
                    const float corners[4][5] = {
                        {px - dirX, base, pz - dirZ, u0, 1.0f},
                        {px + dirX, base, pz + dirZ, u1, 1.0f},
                        {px + dirX + leanX, base + height, pz + dirZ + leanZ, u1, 0.0f},
                        {px - dirX + leanX, base + height, pz - dirZ + leanZ, u0, 0.0f},
                    };
                    const uint32_t color = in.tint(bx, bz);
                    const uint32_t word = kCardFlag | (card << 16);
                    const size_t offset = cutout.size();
                    cutout.resize(offset + kQuadBytes);
                    for (int v = 0; v < 4; ++v) {
                        uint8_t* dst = cutout.data() + offset + v * MCRT_VERTEX_STRIDE;
                        std::memcpy(dst, corners[v], 12);
                        std::memcpy(dst + 12, &color, 4);
                        std::memcpy(dst + 16, corners[v] + 3, 8);
                        std::memcpy(dst + 24, &word, 4);
                    }
                }
            }
}


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
