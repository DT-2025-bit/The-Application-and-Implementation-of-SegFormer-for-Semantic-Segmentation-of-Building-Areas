"""检查切片划分是否存在同一原图泄漏到多个 split。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def source_from_tile(name: str) -> str:
    # tile 名: {source_stem}_{idx}_{y}_{x}.png
    m = re.match(r"(.+)_\d{3}_\d+_\d+\.png$", name)
    return m.group(1) if m else Path(name).stem


def collect_sources(split_dir: Path) -> set[str]:
    img_dir = split_dir / "images"
    if not img_dir.exists():
        return set()
    return {source_from_tile(p.name) for p in img_dir.glob("*.png")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/processed/massachusetts")
    args = parser.parse_args()
    root = Path(args.root)

    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = collect_sources(root / split)
        print(f"{split}: {len(splits[split])} source images, "
              f"{len(list((root / split / 'images').glob('*.png'))) if (root / split / 'images').exists() else 0} tiles")

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    ok = True
    for a, b in pairs:
        inter = splits[a] & splits[b]
        if inter:
            ok = False
            print(f"[FAIL] leakage {a}∩{b}: {sorted(list(inter))[:10]} ...")
        else:
            print(f"[OK] no leakage between {a} and {b}")

    report = {k: sorted(v) for k, v in splits.items()}
    with open(root / "split_sources.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
