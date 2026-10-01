"""Builds MCRT's material textures from CC0 sources.

Reads tools/materials/materials.json, downloads each source once into a cache, and writes per
material two square PNGs plus a runtime table into the output directory:

  <name>_albedo.png  RGB = base color (sRGB), A = height
  <name>_data.png    R,G = tangent-space normal (OpenGL convention), B = roughness, A = ambient occlusion
  materials.json     material list (order = material id - 1) and block face mapping
  CREDITS.md         where every texture came from

Usage: python build_materials.py <output dir> [--cache <dir>] [--size 1024]
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

AMBIENTCG_API = "https://ambientcg.com/api/v2/full_json?id={id}&include=downloadData"
RESOLUTION = "1K-JPG"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mcrt-material-builder"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def ambientcg_zip(asset_id: str, cache: Path) -> Path:
    target = cache / f"{asset_id}_{RESOLUTION}.zip"
    if target.exists():
        return target
    info = json.loads(fetch(AMBIENTCG_API.format(id=asset_id)))
    assets = info.get("foundAssets", [])
    if not assets:
        raise RuntimeError(f"ambientCG asset {asset_id} not found")
    downloads = assets[0]["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
    match = next((d for d in downloads if d["attribute"] == RESOLUTION), None)
    if match is None:
        raise RuntimeError(f"{asset_id} has no {RESOLUTION} download")
    print(f"  downloading {asset_id} ({RESOLUTION})")
    target.write_bytes(fetch(match["downloadLink"]))
    return target


def map_from_zip(archive: zipfile.ZipFile, suffix: str):
    for name in archive.namelist():
        if name.lower().endswith(suffix.lower()):
            return Image.open(io.BytesIO(archive.read(name)))
    return None


def build_material(name: str, spec: dict, cache: Path, out: Path, size: int) -> str:
    kind, _, asset_id = spec["source"].partition(":")
    if kind != "ambientcg":
        raise RuntimeError(f"{name}: unsupported source {spec['source']}")
    with zipfile.ZipFile(ambientcg_zip(asset_id, cache)) as archive:
        color = map_from_zip(archive, "_Color.jpg")
        normal = map_from_zip(archive, "_NormalGL.jpg")
        roughness = map_from_zip(archive, "_Roughness.jpg")
        ao = map_from_zip(archive, "_AmbientOcclusion.jpg")
        height = map_from_zip(archive, "_Displacement.jpg")
    if color is None or normal is None or roughness is None:
        raise RuntimeError(f"{name}: {asset_id} is missing color, normal or roughness maps")

    def channel(image, mode="L", fill=255):
        if image is None:
            return Image.new(mode, (size, size), fill)
        return image.convert(mode).resize((size, size), Image.LANCZOS)

    r, g, b = color.convert("RGB").resize((size, size), Image.LANCZOS).split()
    Image.merge("RGBA", (r, g, b, channel(height, fill=128))).save(out / f"{name}_albedo.png", optimize=True)
    nx, ny, _ = normal.convert("RGB").resize((size, size), Image.LANCZOS).split()
    Image.merge("RGBA", (nx, ny, channel(roughness), channel(ao))).save(out / f"{name}_data.png", optimize=True)
    return asset_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(__file__).parent / ".cache")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    config = json.loads((Path(__file__).parent / "materials.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    materials = []
    credits = ["# Material credits", "", "All textures are CC0 (public domain).", ""]
    for name, spec in config["materials"].items():
        print(f"material {name}")
        asset_id = build_material(name, spec, args.cache, args.output, args.size)
        materials.append({"name": name, "scale": spec.get("scale", 1), "tinted": bool(spec.get("tinted", False))})
        credits.append(f"- `{name}`: ambientCG [{asset_id}](https://ambientcg.com/view?id={asset_id}), CC0")

    names = {m["name"] for m in materials}
    for block, faces in config["blocks"].items():
        for face, material in faces.items():
            if face == "smooth":
                continue
            if material not in names:
                raise RuntimeError(f"{block}.{face} refers to unknown material {material}")

    runtime = {"size": args.size, "materials": materials, "blocks": config["blocks"]}
    (args.output / "materials.json").write_text(json.dumps(runtime, indent=1), encoding="utf-8")
    (args.output / "CREDITS.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(f"wrote {len(materials)} materials to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
