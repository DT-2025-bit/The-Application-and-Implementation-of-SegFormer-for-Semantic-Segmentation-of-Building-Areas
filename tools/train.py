"""训练入口：python tools/train.py --config configs/segformer_b0_edge.yaml"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import BuildingSegDataset, collate_fn
from src.losses import CombinedLoss
from src.metrics import SegmentationMeter
from src.models import build_model
from src.utils import load_config, save_checkpoint, set_seed


def build_loaders(cfg: dict):
    train_ds = BuildingSegDataset(
        cfg["train_image_dir"], cfg["train_mask_dir"], cfg["img_size"], train=True
    )
    val_ds = BuildingSegDataset(
        cfg["val_image_dir"], cfg["val_mask_dir"], cfg["img_size"], train=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int):
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes, boundary_bandwidth=2)
    loss_sum, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        losses = criterion(logits, masks)
        loss_sum += float(losses["loss"].item())
        n += 1
        meter.update(logits, masks)
    metrics = meter.compute()
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(save_dir / "tb")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(
        cfg["model_name"], cfg["encoder"], cfg.get("num_classes", 2), cfg.get("pretrained", True)
    ).to(device)

    criterion = CombinedLoss(
        ce_weight=cfg.get("ce_weight", 1.0),
        dice_weight=cfg.get("dice_weight", 1.0),
        edge_weight=cfg.get("edge_weight", 0.0),
        edge_kernel=cfg.get("edge_kernel", 3),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01)
    )
    epochs = cfg["epochs"]
    warmup = cfg.get("warmup_epochs", 0)

    def lr_at(epoch: int) -> float:
        if epoch < warmup:
            return cfg["lr"] * float(epoch + 1) / float(max(warmup, 1))
        # cosine
        import math

        t = (epoch - warmup) / max(epochs - warmup, 1)
        return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * t))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")

    best_miou = -1.0
    patience = cfg.get("early_stop_patience", 15)
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        lr = lr_at(epoch)
        for g in optimizer.param_groups:
            g["lr"] = lr

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        running = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "edge": 0.0}
        steps = 0
        t0 = time.time()

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                losses = criterion(logits, masks)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            for k in running:
                running[k] += float(losses[k].item() if torch.is_tensor(losses[k]) else losses[k])
            if steps % cfg.get("log_interval", 20) == 0:
                pbar.set_postfix(
                    loss=running["loss"] / steps,
                    lr=f"{lr:.2e}",
                )

        train_loss = running["loss"] / max(steps, 1)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        if (epoch + 1) % cfg.get("val_interval", 1) == 0:
            metrics = validate(model, val_loader, criterion, device, cfg.get("num_classes", 2))
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} mIoU={metrics['mIoU']:.4f} "
                f"Dice={metrics['Dice']:.4f} B-IoU={metrics['BoundaryIoU']:.4f} "
                f"time={time.time()-t0:.1f}s"
            )
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "config": cfg,
            }
            save_checkpoint(ckpt, save_dir / "last.pth")
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                bad_epochs = 0
                save_checkpoint(ckpt, save_dir / "best.pth")
                print(f"  -> new best mIoU={best_miou:.4f}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stop at epoch {epoch+1}, best mIoU={best_miou:.4f}")
                    break

    writer.close()
    print("Training done. Best mIoU=", best_miou)


if __name__ == "__main__":
    main()
