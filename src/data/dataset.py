from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_transforms(img_size: int = 512, train: bool = True):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class BuildingSegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 512, train: bool = True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.train = train
        self.transforms = build_transforms(img_size, train)
        self.images = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            # 尝试同 stem 不同后缀
            candidates = list(self.mask_dir.glob(img_path.stem + ".*"))
            if not candidates:
                raise FileNotFoundError(f"Mask not found for {img_path.name}")
            mask_path = candidates[0]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)
        out = self.transforms(image=image, mask=mask)
        return {
            "image": out["image"].float(),
            "mask": out["mask"].long(),
            "name": img_path.name,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    names = [b["name"] for b in batch]
    return {"image": images, "mask": masks, "name": names}
