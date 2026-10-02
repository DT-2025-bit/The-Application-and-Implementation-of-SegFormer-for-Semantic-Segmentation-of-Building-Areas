"""将 1500x1500 遥感图切为 512x512，默认 overlap=64，并按原图划分防止泄漏。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def sliding_windows(h: int, w: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
    ys = list(range(0, max(h - size, 0) + 1, stride))
    xs = list(range(0, max(w - size, 0) + 1, stride))
    if not ys or ys[-1] != h - size:
        ys.append(max(h - size, 0))
    if not xs or xs[-1] != w - size:
        xs.append(max(w - size, 0))
    boxes = []
    for y in ys:
        for x in xs:
            boxes.append((y, x, y + size, x + size))
    return boxes


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def process_split(
    raw_split_dir: Path,
    out_split_dir: Path,
    tile_size: int,
    overlap: int,
    min_building_ratio: float,
) -> dict:
    img_dir = raw_split_dir / "images"
    mask_dir = raw_split_dir / "masks"
    out_img = out_split_dir / "images"
    out_mask = out_split_dir / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    stride = tile_size - overlap
    stats = {"source_images": 0, "tiles_kept": 0, "tiles_dropped": 0}
    index = []

    images = list_images(img_dir)
    for img_path in tqdm(images, desc=f"slice {raw_split_dir.name}"):
        mask_path = find_mask(mask_dir, img_path.stem)
        if mask_path is None:
            print(f"[WARN] missing mask for {img_path.name}, skip")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            print(f"[WARN] failed to read {img_path.name}")
            continue

        mask = binarize_mask(mask)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        h, w = image.shape[:2]
        # 不足 tile_size 时右侧/底部 pad
        pad_h = max(tile_size - h, 0)
        pad_w = max(tile_size - w, 0)
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = image.shape[:2]

        stats["source_images"] += 1
        for i, (y0, x0, y1, x1) in enumerate(sliding_windows(h, w, tile_size, stride)):
            tile_img = image[y0:y1, x0:x1]
            tile_mask = mask[y0:y1, x0:x1]
            ratio = float((tile_mask > 0).mean())
            if ratio < min_building_ratio:
                stats["tiles_dropped"] += 1
                continue

            name = f"{img_path.stem}_{i:03d}_{y0}_{x0}.png"
            cv2.imwrite(str(out_img / name), tile_img)
            cv2.imwrite(str(out_mask / name), tile_mask)
            index.append(
                {
                    "tile": name,
                    "source": img_path.name,
                    "y0": y0,
                    "x0": x0,
                    "building_ratio": round(ratio, 6),
                }
            )
            stats["tiles_kept"] += 1

    with open(out_split_dir / "tiles_index.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "tiles": index}, f, ensure_ascii=False, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Slice Massachusetts buildings images")
    parser.add_argument("--raw-root", type=str, default="data/raw/massachusetts")
    parser.add_argument("--out-root", type=str, default="data/processed/massachusetts")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--min-building-ratio",
        type=float,
        default=0.01,
        help="丢弃建筑像素占比过低的切片；设为 0 则全部保留",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    all_stats = {}
    for split in args.splits:
        raw_split = raw_root / split
        if not (raw_split / "images").exists():
            print(f"[SKIP] {raw_split} not found")
            continue
        stats = process_split(
            raw_split,
            out_root / split,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_building_ratio=args.min_building_ratio,
        )
        all_stats[split] = stats
        print(split, stats)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "slice_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print("Done. Summary ->", out_root / "slice_summary.json")


if __name__ == "__main__":
    main()
