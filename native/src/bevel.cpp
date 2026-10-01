#include "bevel.h"

#include "mcrt/api.h"
#include "terrain_mesher.h"

#include <array>
#include <cmath>
#include <cstring>

namespace mcrt::bevel {

namespace {

constexpr size_t kQuadBytes = size_t(MCRT_VERTEX_STRIDE) * 4;
constexpr float kEps = 1e-3f;

// Face directions: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z.
int axisOf(int dir) { return dir / 2; }
float signOf(int dir) { return (dir & 1) ? -1.0f : 1.0f; }

struct Vertex {
    float p[3];
    uint32_t color;
    float uv[2];
};

Vertex readVertex(const uint8_t* src) {
    Vertex v;
    std::memcpy(v.p, src, 12);
    std::memcpy(&v.color, src + 12, 4);
    std::memcpy(v.uv, src + 16, 8);
    return v;
}

void writeVertex(uint8_t* dst, const float p[3], uint32_t color, const float uv[2]) {
    const uint32_t word = 0; // annotateQuads fills in emission and material afterwards
    std::memcpy(dst, p, 12);
    std::memcpy(dst + 12, &color, 4);
    std::memcpy(dst + 16, uv, 8);
    std::memcpy(dst + 24, &word, 4);
}

// One exposed face of a block, with a bilinear map from in-plane coordinates to atlas UVs.
struct Face {
    int quad = -1;
    float uv[2][2][2]; // [s][t][uv] at the face's in-plane corners (s, t in {0, 1})
    uint32_t color = 0xFFFFFFFFu;
};

// In-plane axes of a face: the two axes other than its normal axis, in increasing order.
void planeAxes(int axis, int& a, int& b) {
    a = axis == 0 ? 1 : 0;
    b = axis == 2 ? 1 : 2;
}

// Returns the face direction if the quad is exactly one full face of block `base`, else -1.
int fullFaceDirection(const Vertex (&v)[4], const int base[3], const float normal[3]) {
    int dir = -1;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(normal[axis]) > 0.999f)
            dir = axis * 2 + (normal[axis] > 0.0f ? 0 : 1);
    }
    if (dir < 0)
        return -1;
    const int axis = axisOf(dir);
    const float plane = float(base[axis]) + (signOf(dir) > 0 ? 1.0f : 0.0f);
    int a, b;
    planeAxes(axis, a, b);
    int corners = 0;
    for (const Vertex& vertex : v) {
        if (std::fabs(vertex.p[axis] - plane) > kEps)
            return -1;
        const float s = vertex.p[a] - float(base[a]);
        const float t = vertex.p[b] - float(base[b]);
        const bool sEdge = std::fabs(s) < kEps || std::fabs(s - 1.0f) < kEps;
        const bool tEdge = std::fabs(t) < kEps || std::fabs(t - 1.0f) < kEps;
        if (!sEdge || !tEdge)
            return -1;
        corners |= 1 << ((s > 0.5f ? 1 : 0) + (t > 0.5f ? 2 : 0));
    }
    return corners == 0xF ? dir : -1;
}

struct BlockFaces {
    bool invalid = false;
    std::array<Face, 6> faces{};
    int count = 0;
};

// Point on the block surface: coordinates in [0,1]^3 relative to the block.
struct Emitter {
    std::vector<uint8_t>& out;
    const int* base;

    void quad(const float (&p)[4][3], const uint32_t color, const float (&uv)[4][2]) {
        // Wind counter-clockwise seen from outside, like Minecraft's quads: later passes find a
        // quad's block by stepping against its normal.
        float e1[3], e2[3], c[3];
        for (int i = 0; i < 3; ++i) {
            e1[i] = p[1][i] - p[0][i];
            e2[i] = p[2][i] - p[0][i];
            c[i] = (p[0][i] + p[1][i] + p[2][i] + p[3][i]) * 0.25f - 0.5f;
        }
        const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        const bool inward = n[0] * c[0] + n[1] * c[1] + n[2] * c[2] < 0.0f;
        const bool triangle = p[2][0] == p[3][0] && p[2][1] == p[3][1] && p[2][2] == p[3][2];
        int order[4] = {0, 1, 2, 3};
        if (inward) {
            if (triangle) {
                order[1] = 2;
                order[2] = 1;
                order[3] = 1;
            } else {
                order[1] = 3;
                order[3] = 1;
            }
        }
        const size_t offset = out.size();
        out.resize(offset + kQuadBytes);
        for (int i = 0; i < 4; ++i) {
            const int k = order[i];
            float world[3] = {base[0] + p[k][0], base[1] + p[k][1], base[2] + p[k][2]};
            writeVertex(out.data() + offset + i * MCRT_VERTEX_STRIDE, world, color, uv[k]);
        }
    }
};

// UV of `face` at a point given in block-relative coordinates (projected onto the face plane).
void faceUv(const Face& face, int dir, const float p[3], float out[2]) {
    int a, b;
    planeAxes(axisOf(dir), a, b);
    const float s = std::fmin(std::fmax(p[a], 0.0f), 1.0f);
    const float t = std::fmin(std::fmax(p[b], 0.0f), 1.0f);
    for (int c = 0; c < 2; ++c) {
        const float bottom = face.uv[0][0][c] * (1 - s) + face.uv[1][0][c] * s;
        const float top = face.uv[0][1][c] * (1 - s) + face.uv[1][1][c] * s;
        out[c] = bottom * (1 - t) + top * t;
    }
}

void emitBeveledBlock(Emitter& emit, const BlockFaces& block) {
    const float w = kBevelWidth;
    auto exposed = [&](int dir) { return block.faces[dir].quad >= 0; };

    // Inset faces: pull in each edge whose neighbouring face of this block is also exposed.
    for (int dir = 0; dir < 6; ++dir) {
        if (!exposed(dir))
            continue;
        const Face& face = block.faces[dir];
        const int axis = axisOf(dir);
        int a, b;
        planeAxes(axis, a, b);
        const float lo[2] = {exposed(a * 2 + 1) ? w : 0.0f, exposed(b * 2 + 1) ? w : 0.0f};
        const float hi[2] = {exposed(a * 2) ? 1 - w : 1.0f, exposed(b * 2) ? 1 - w : 1.0f};
        const float plane = signOf(dir) > 0 ? 1.0f : 0.0f;
        float p[4][3], uv[4][2];
        const float st[4][2] = {{lo[0], lo[1]}, {hi[0], lo[1]}, {hi[0], hi[1]}, {lo[0], hi[1]}};
        for (int i = 0; i < 4; ++i) {
            p[i][axis] = plane;
            p[i][a] = st[i][0];
            p[i][b] = st[i][1];
            faceUv(face, dir, p[i], uv[i]);
        }
        emit.quad(p, face.color, uv);
    }

    // Bevel strips along convex edges (pairs of exposed faces on different axes).
    for (int d0 = 0; d0 < 6; ++d0) {
        for (int d1 = d0 + 1; d1 < 6; ++d1) {
            if (axisOf(d0) == axisOf(d1) || !exposed(d0) || !exposed(d1))
                continue;
            const int a0 = axisOf(d0), a1 = axisOf(d1);
            const int along = 3 - a0 - a1;
            const float e0 = signOf(d0) > 0 ? 1.0f : 0.0f; // edge coordinate on axis a0
            const float e1 = signOf(d1) > 0 ? 1.0f : 0.0f; // edge coordinate on axis a1
            const float in0 = signOf(d0) > 0 ? 1 - w : w;  // inset coordinate on axis a0
            const float in1 = signOf(d1) > 0 ? 1 - w : w;
            const float start = exposed(along * 2 + 1) ? w : 0.0f;
            const float end = exposed(along * 2) ? 1 - w : 1.0f;
            // Strip from face d0's inset edge to face d1's inset edge.
            float p[4][3], uv[4][2];
            const float alongValues[4] = {start, end, end, start};
            for (int i = 0; i < 4; ++i) {
                const bool onD0 = i < 2;
                p[i][along] = alongValues[i];
                p[i][a0] = onD0 ? e0 : in0;
                p[i][a1] = onD0 ? in1 : e1;
                faceUv(block.faces[d0], d0, p[i], uv[i]);
            }
            emit.quad(p, block.faces[d0].color, uv);
        }
    }

    // Corner triangles where three exposed faces meet (emitted as a quad with a repeated vertex).
    for (int sx = 0; sx < 2; ++sx)
        for (int sy = 0; sy < 2; ++sy)
            for (int sz = 0; sz < 2; ++sz) {
                const int dx = sx ? 0 : 1, dy = sy ? 2 : 3, dz = sz ? 4 : 5;
                if (!exposed(dx) || !exposed(dy) || !exposed(dz))
                    continue;
                const float c[3] = {sx ? 1.0f : 0.0f, sy ? 1.0f : 0.0f, sz ? 1.0f : 0.0f};
                const float in[3] = {sx ? 1 - w : w, sy ? 1 - w : w, sz ? 1 - w : w};
                float p[4][3] = {{c[0], in[1], in[2]}, {in[0], c[1], in[2]}, {in[0], in[1], c[2]}, {in[0], in[1], c[2]}};
                float uv[4][2];
                for (int i = 0; i < 4; ++i)
                    faceUv(block.faces[dy], dy, p[i], uv[i]);
                emit.quad(p, block.faces[dy].color, uv);
            }
}

} // namespace

void bevelFullCubes(std::vector<uint8_t>& layer) {
    const size_t quadCount = layer.size() / kQuadBytes;
    if (quadCount == 0)
        return;
    std::vector<BlockFaces> blocks(4096);
    std::vector<int> owner(quadCount, -1);

    for (size_t q = 0; q < quadCount; ++q) {
        const uint8_t* quad = layer.data() + q * kQuadBytes;
        float normal[3];
        const int local = terrain::quadBlock(quad, normal);
        if (local < 0)
            continue;
        owner[q] = local;
        BlockFaces& block = blocks[local];
        if (block.invalid)
            continue;
        Vertex v[4];
        for (int i = 0; i < 4; ++i)
            v[i] = readVertex(quad + i * MCRT_VERTEX_STRIDE);
        const int base[3] = {local & 15, local >> 8, (local >> 4) & 15};
        const int dir = fullFaceDirection(v, base, normal);
        if (dir < 0 || block.faces[dir].quad >= 0) {
            block.invalid = true; // not a plain cube (or two quads on one face, e.g. overlays)
            continue;
        }
        Face& face = block.faces[dir];
        face.quad = int(q);
        face.color = v[0].color;
        int a, b;
        planeAxes(axisOf(dir), a, b);
        for (const Vertex& vertex : v) {
            const int s = vertex.p[a] - float(base[a]) > 0.5f ? 1 : 0;
            const int t = vertex.p[b] - float(base[b]) > 0.5f ? 1 : 0;
            face.uv[s][t][0] = vertex.uv[0];
            face.uv[s][t][1] = vertex.uv[1];
        }
        ++block.count;
    }

    std::vector<uint8_t> out;
    out.reserve(layer.size() + layer.size() / 2);
    // Untouched quads first, in their original order.
    for (size_t q = 0; q < quadCount; ++q) {
        const int local = owner[q];
        if (local >= 0 && !blocks[local].invalid && blocks[local].count > 0)
            continue;
        out.insert(out.end(), layer.begin() + q * kQuadBytes, layer.begin() + (q + 1) * kQuadBytes);
    }
    for (int local = 0; local < 4096; ++local) {
        const BlockFaces& block = blocks[local];
        if (block.invalid || block.count == 0)
            continue;
        const int base[3] = {local & 15, local >> 8, (local >> 4) & 15};
        Emitter emit{out, base};
        emitBeveledBlock(emit, block);
    }
    layer.swap(out);
}

} // namespace mcrt::bevel
