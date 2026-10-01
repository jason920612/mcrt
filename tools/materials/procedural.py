"""Procedurally generated materials (no external source, so no license question).

Each generator returns (albedo RGBA, normal RGB, surface RGB) PIL images of the requested size,
laid out like the ambientCG-derived materials (see build_materials.py).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def grass_blades(size: int, seed: int = 7):
    """A strip of grass blades for alpha-tested grass cards: blades rise from the bottom edge.

    rgb = blade color (authored around plains grass, the shader applies the biome tint),
    a = coverage. The texture tiles horizontally; cards pick a random horizontal window.
    """
    rng = np.random.default_rng(seed)
    scale = 2  # supersample for smooth blade edges
    w = h = size * scale
    color = Image.new("RGB", (w, h), (0, 0, 0))
    coverage = Image.new("L", (w, h), 0)
    shade = Image.new("L", (w, h), 0)  # ambient occlusion: dark at the roots
    draw_color = ImageDraw.Draw(color)
    draw_cov = ImageDraw.Draw(coverage)
    draw_shade = ImageDraw.Draw(shade)

    blades = 520
    for i in range(blades):
        base_x = rng.uniform(0, w)
        height = h * rng.uniform(0.35, 0.97) ** 1.5
        base_w = w * rng.uniform(0.006, 0.014)
        lean = rng.normal(0, 0.18) * height  # tip offset
        bend = rng.uniform(0.3, 1.0)
        # Plains-grass greens with a few dry and a few dark blades.
        kind = rng.random()
        if kind < 0.12:
            tip = np.array([190, 175, 95]) * rng.uniform(0.8, 1.05)
            root = np.array([110, 115, 55]) * rng.uniform(0.8, 1.0)
        elif kind < 0.3:
            tip = np.array([95, 140, 55]) * rng.uniform(0.75, 1.0)
            root = np.array([45, 75, 25]) * rng.uniform(0.8, 1.0)
        else:
            tip = np.array([150, 190, 85]) * rng.uniform(0.85, 1.08)
            root = np.array([70, 110, 35]) * rng.uniform(0.8, 1.05)
        segments = 10
        left, right = [], []
        for s in range(segments + 1):
            t = s / segments
            x = base_x + lean * (t ** (1.0 + bend))
            y = h - height * t
            half = base_w * (1.0 - t) ** 0.8 * 0.5
            left.append((x - half, y))
            right.append((x + half, y))
        # Draw in segments so color runs from root to tip; repeat across the tiling seam.
        for offset in (-w, 0, w):
            for s in range(segments):
                t = (s + 0.5) / segments
                c = tuple(int(v) for v in np.clip(root + (tip - root) * t, 0, 255))
                quad = [left[s], left[s + 1], right[s + 1], right[s]]
                quad = [(x + offset, y) for x, y in quad]
                draw_color.polygon(quad, fill=c)
                draw_cov.polygon(quad, fill=255)
                draw_shade.polygon(quad, fill=int(255 * min(1.0, 0.35 + 0.65 * t ** 0.6)))

    color = color.resize((size, size), Image.LANCZOS)
    coverage = coverage.resize((size, size), Image.LANCZOS)
    shade = shade.resize((size, size), Image.LANCZOS)

    # Bleed blade colors into the transparent gaps so filtering never pulls in black fringes.
    rgb = np.asarray(color).astype(np.float32)
    a = np.asarray(coverage).astype(np.float32) / 255.0
    filled = rgb.copy()
    weight = a.copy()
    for radius in (2, 6, 16, 48):
        blur_rgb = np.asarray(Image.fromarray((rgb * a[..., None]).clip(0, 255).astype(np.uint8))
                              .filter(ImageFilter.BoxBlur(radius))).astype(np.float32)
        blur_a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(radius))) \
            .astype(np.float32) / 255.0
        estimate = blur_rgb / np.maximum(blur_a[..., None], 1e-3)
        take = (weight < 0.5) & (blur_a > 0.01)
        filled[take] = estimate[take]
        weight = np.maximum(weight, np.where(blur_a > 0.01, 1.0, 0.0))
    rgb = np.where(a[..., None] > 0.5, rgb, filled)

    albedo = Image.fromarray(np.dstack([rgb.clip(0, 255), a * 255]).astype(np.uint8), "RGBA")
    normal = Image.new("RGB", (size, size), (128, 128, 0))
    surface = Image.merge("RGB", (Image.new("L", (size, size), 165), shade, Image.new("L", (size, size), 0)))
    return albedo, normal, surface


def _bleed(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Fills transparent texels with nearby opaque colors so filtering never pulls in black."""
    filled = rgb.copy()
    known = a > 0.5
    for radius in (2, 6, 16, 48):
        blur_rgb = np.asarray(Image.fromarray((rgb * a[..., None]).clip(0, 255).astype(np.uint8))
                              .filter(ImageFilter.BoxBlur(radius))).astype(np.float32)
        blur_a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(radius)))             .astype(np.float32) / 255.0
        estimate = blur_rgb / np.maximum(blur_a[..., None], 1e-3)
        take = ~known & (blur_a > 0.01)
        filled[take] = estimate[take]
        known = known | (blur_a > 0.01)
    return np.where(a[..., None] > 0.5, rgb, filled)


def _foliage(size: int, seed: int, draw_twig):
    """Shared scaffolding for leaf-cluster cards: twigs radiate in a disc, density falls off
    towards the rim so a card reads as a rounded clump, not a square."""
    rng = np.random.default_rng(seed)
    scale = 2
    w = size * scale
    color = Image.new("RGB", (w, w), (0, 0, 0))
    coverage = Image.new("L", (w, w), 0)
    shade = Image.new("L", (w, w), 0)
    draws = (ImageDraw.Draw(color), ImageDraw.Draw(coverage), ImageDraw.Draw(shade))
    center = np.array([w / 2, w / 2])
    for _ in range(int(rng.integers(24, 30))):
        angle = rng.uniform(0, 2 * np.pi)
        start = center + rng.normal(0, w * 0.06, 2)
        length = w * rng.uniform(0.3, 0.46)
        draw_twig(rng, draws, start, angle, length, w)
    rgb = np.asarray(color.resize((size, size), Image.LANCZOS)).astype(np.float32)
    a = np.asarray(coverage.resize((size, size), Image.LANCZOS)).astype(np.float32) / 255.0
    ao = shade.resize((size, size), Image.LANCZOS)
    rgb = _bleed(rgb, a)
    albedo = Image.fromarray(np.dstack([rgb.clip(0, 255), a * 255]).astype(np.uint8), "RGBA")
    normal = Image.new("RGB", (size, size), (128, 128, 0))
    surface = Image.merge("RGB", (Image.new("L", (size, size), 150), ao, Image.new("L", (size, size), 0)))
    return albedo, normal, surface


def _leaf_polygon(tip, base, width):
    """A pointed leaf outline from base to tip."""
    d = tip - base
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-6)
    points = []
    for t in np.linspace(0, 1, 8):
        half = width * np.sin(np.pi * t) ** 0.8 * 0.5
        points.append(base + d * t + n * half)
    for t in np.linspace(1, 0, 8)[1:-1]:
        half = width * np.sin(np.pi * t) ** 0.8 * 0.5
        points.append(base + d * t - n * half)
    return [tuple(p) for p in points]


def broad_leaves(size: int, seed: int = 11):
    """Broadleaf clumps (oak, birch, jungle...): rgb authored around a natural leaf green, the
    shader scales it by biome foliage color relative to plains foliage."""
    def twig(rng, draws, start, angle, length, w):
        draw_color, draw_cov, draw_shade = draws
        direction = np.array([np.cos(angle), np.sin(angle)])
        end = start + direction * length
        draw_color.line([tuple(start), tuple(end)], fill=(70, 55, 35), width=max(2, int(w * 0.004)))
        draw_cov.line([tuple(start), tuple(end)], fill=255, width=max(2, int(w * 0.004)))
        draw_shade.line([tuple(start), tuple(end)], fill=110, width=max(2, int(w * 0.004)))
        leaves = int(rng.integers(55, 75))
        for i in range(leaves):
            t = rng.uniform(0.1, 1.0)
            base = start + direction * length * t
            # Fewer leaves far from the clump's center.
            r = np.linalg.norm(base - np.array([w / 2, w / 2])) / (w / 2)
            if rng.random() < r ** 3.5:
                continue
            side = 1 if rng.random() < 0.5 else -1
            leaf_angle = angle + side * rng.uniform(0.5, 1.3) + rng.normal(0, 0.2)
            leaf_len = w * rng.uniform(0.05, 0.085)
            tip = base + np.array([np.cos(leaf_angle), np.sin(leaf_angle)]) * leaf_len
            kind = rng.random()
            if kind < 0.08:
                c = np.array([150, 150, 60])          # yellowing
            elif kind < 0.35:
                c = np.array([45, 85, 30])            # shaded, dark
            else:
                c = np.array([75, 125, 40])           # healthy
            c = c * rng.uniform(0.85, 1.15)
            polygon = _leaf_polygon(tip, base, leaf_len * rng.uniform(0.38, 0.5))
            draw_color.polygon(polygon, fill=tuple(int(v) for v in np.clip(c, 0, 255)))
            draw_cov.polygon(polygon, fill=255)
            # Leaves deeper in the clump are darker (self-shadowing).
            draw_shade.polygon(polygon, fill=int(255 * np.clip(0.55 + 0.45 * r, 0, 1)))
            # Midrib.
            mid = tuple(int(v) for v in np.clip(c * 0.75, 0, 255))
            draw_color.line([tuple(base), tuple(tip)], fill=mid, width=1)
    return _foliage(size, seed, twig)


def needle_leaves(size: int, seed: int = 13):
    """Conifer sprigs (spruce): dense short needles along twigs, dark blue-green."""
    def twig(rng, draws, start, angle, length, w):
        draw_color, draw_cov, draw_shade = draws
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        end = start + direction * length
        draw_color.line([tuple(start), tuple(end)], fill=(60, 45, 30), width=max(3, int(w * 0.006)))
        draw_cov.line([tuple(start), tuple(end)], fill=255, width=max(3, int(w * 0.006)))
        steps = int(length / (w * 0.006))
        for i in range(steps):
            t = i / steps
            p = start + direction * length * t
            r = np.linalg.norm(p - np.array([w / 2, w / 2])) / (w / 2)
            needle = w * rng.uniform(0.04, 0.065) * (1.0 - 0.4 * t)
            for side in (-1, 1):
                if rng.random() < r ** 3:
                    continue
                d = direction * 0.55 + normal * side * rng.uniform(0.7, 1.0)
                d /= np.linalg.norm(d)
                tip = p + d * needle
                c = np.array([58, 98, 62]) * rng.uniform(0.75, 1.2)
                draw_color.line([tuple(p), tuple(tip)], fill=tuple(int(v) for v in np.clip(c, 0, 255)),
                                width=max(2, int(w * 0.0055)))
                draw_cov.line([tuple(p), tuple(tip)], fill=255, width=max(2, int(w * 0.0055)))
                draw_shade.line([tuple(p), tuple(tip)], fill=int(255 * np.clip(0.5 + 0.5 * r, 0, 1)),
                                width=max(2, int(w * 0.0035)))
    return _foliage(size, seed, twig)


GENERATORS = {"grass_blades": grass_blades, "broad_leaves": broad_leaves, "needle_leaves": needle_leaves}
