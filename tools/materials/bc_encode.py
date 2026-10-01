"""Small vectorized BC1/BC3/BC4/BC5 encoders (and decoders, for checking) using numpy.

Quality target is "good texture compression", not a reference encoder: BC1 endpoints come from the
principal axis of each block, BC4 uses the block's min/max with 8 interpolated levels.
"""

import numpy as np


def _blocks(channel_data):
    """(H, W, C) -> (H/4 * W/4, 16, C) in row-major block order, texels row-major within a block."""
    h, w, c = channel_data.shape
    b = channel_data.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4)
    return b.reshape(-1, 16, c)


def encode_bc4(channel):
    """channel: (H, W) uint8 -> bytes (8 per block)."""
    v = _blocks(channel[..., None].astype(np.float32))[..., 0]          # (N, 16)
    hi = v.max(axis=1)
    lo = v.min(axis=1)
    flat = hi <= lo
    hi = np.where(flat, np.minimum(lo + 1, 255), hi)
    # 8-level mode (a0 > a1): palette = a0, a1, then 6 interpolated levels.
    t = np.array([0, 7, 1, 2, 3, 4, 5, 6], dtype=np.float32) / 7.0        # palette index -> weight of a1
    palette = hi[:, None] * (1 - t[None, :]) + lo[:, None] * t[None, :]  # (N, 8)
    idx = np.abs(v[:, :, None] - palette[:, None, :]).argmin(axis=2).astype(np.uint64)  # (N, 16)
    bits = np.zeros(v.shape[0], dtype=np.uint64)
    for i in range(16):
        bits |= idx[:, i] << np.uint64(3 * i)
    out = np.zeros((v.shape[0], 8), dtype=np.uint8)
    out[:, 0] = hi.astype(np.uint8)
    out[:, 1] = lo.astype(np.uint8)
    for i in range(6):
        out[:, 2 + i] = ((bits >> np.uint64(8 * i)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def _to565(c):
    r = np.clip(np.round(c[:, 0] * 31 / 255), 0, 31).astype(np.uint32)
    g = np.clip(np.round(c[:, 1] * 63 / 255), 0, 63).astype(np.uint32)
    b = np.clip(np.round(c[:, 2] * 31 / 255), 0, 31).astype(np.uint32)
    return (r << 11) | (g << 5) | b


def _from565(v):
    v = v.astype(np.uint32)
    r = ((v >> 11) & 31) * 255 / 31
    g = ((v >> 5) & 63) * 255 / 63
    b = (v & 31) * 255 / 31
    return np.stack([r, g, b], axis=1).astype(np.float32)


def encode_bc1(rgb):
    """rgb: (H, W, 3) uint8 -> (N, 8) uint8, always 4-color mode (as BC3 requires)."""
    px = _blocks(rgb.astype(np.float32))                                 # (N, 16, 3)
    mean = px.mean(axis=1, keepdims=True)
    centered = px - mean
    cov = np.einsum('nki,nkj->nij', centered, centered)
    # Principal axis via a few power iterations (robust and vectorized).
    axis = np.ones((px.shape[0], 3), dtype=np.float32)
    for _ in range(4):
        axis = np.einsum('nij,nj->ni', cov, axis)
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-6)
    proj = np.einsum('nki,ni->nk', centered, axis)
    c0 = mean[:, 0] + axis * proj.max(axis=1, keepdims=True)
    c1 = mean[:, 0] + axis * proj.min(axis=1, keepdims=True)
    e0 = _to565(np.clip(c0, 0, 255))
    e1 = _to565(np.clip(c1, 0, 255))
    swap = e0 < e1
    e0, e1 = np.where(swap, e1, e0), np.where(swap, e0, e1)
    same = e0 == e1
    # Keep 4-color mode valid: nudge equal endpoints apart.
    e0 = np.where(same & (e0 < 0xFFFF), e0 + 1, e0)
    e1 = np.where(same & (e0 == 0xFFFF), e1 - 1, e1)
    p0 = _from565(e0)
    p1 = _from565(e1)
    palette = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], axis=1)  # (N, 4, 3)
    d = ((px[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(axis=3)        # (N, 16, 4)
    idx = d.argmin(axis=2).astype(np.uint32)
    bits = np.zeros(px.shape[0], dtype=np.uint32)
    for i in range(16):
        bits |= idx[:, i] << np.uint32(2 * i)
    out = np.zeros((px.shape[0], 8), dtype=np.uint8)
    out[:, 0] = e0 & 0xFF
    out[:, 1] = e0 >> 8
    out[:, 2] = e1 & 0xFF
    out[:, 3] = e1 >> 8
    for i in range(4):
        out[:, 4 + i] = (bits >> np.uint32(8 * i)) & 0xFF
    return out


def encode_bc3(rgba):
    return np.concatenate([encode_bc4(rgba[..., 3]), encode_bc1(rgba[..., :3])], axis=1).tobytes()


def encode_bc5(rg):
    return np.concatenate([encode_bc4(rg[..., 0]), encode_bc4(rg[..., 1])], axis=1).tobytes()


# ---- decoders (verification only) -------------------------------------------------------------

def _unblock(blocks, h, w, c):
    return blocks.reshape(h // 4, w // 4, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(h, w, c)


def decode_bc4(data, h, w):
    b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 8).astype(np.float32)
    a0, a1 = b[:, 0], b[:, 1]
    t = np.array([0, 7, 1, 2, 3, 4, 5, 6], dtype=np.float32) / 7.0
    palette = a0[:, None] * (1 - t) + a1[:, None] * t
    bits = np.zeros(b.shape[0], dtype=np.uint64)
    for i in range(6):
        bits |= b[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    idx = np.stack([(bits >> np.uint64(3 * i)) & np.uint64(7) for i in range(16)], axis=1).astype(np.int64)
    vals = np.take_along_axis(palette, idx, axis=1)
    return _unblock(vals[..., None], h, w, 1)[..., 0]


def decode_bc1(blocks, h, w):
    b = blocks.astype(np.uint32)
    e0 = b[:, 0] | (b[:, 1] << 8)
    e1 = b[:, 2] | (b[:, 3] << 8)
    p0, p1 = _from565(e0), _from565(e1)
    palette = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], axis=1)
    bits = b[:, 4] | (b[:, 5] << 8) | (b[:, 6] << 16) | (b[:, 7] << 24)
    idx = np.stack([(bits >> (2 * i)) & 3 for i in range(16)], axis=1).astype(np.int64)
    vals = np.take_along_axis(palette, idx[..., None].repeat(3, axis=2), axis=1)
    return _unblock(vals, h, w, 3)
