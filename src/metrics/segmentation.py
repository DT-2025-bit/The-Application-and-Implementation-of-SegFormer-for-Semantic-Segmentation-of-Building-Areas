from __future__ import annotations

import torch
import torch.nn.functional as F


def _fast_hist(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    mask = (target >= 0) & (target < num_classes)
    hist = torch.bincount(
        num_classes * target[mask].view(-1) + pred[mask].view(-1),
        minlength=num_classes ** 2,
    )
    return hist.reshape(num_classes, num_classes)


def boundary_mask(mask: torch.Tensor, bandwidth: int = 2) -> torch.Tensor:
    """mask: B,H,W long/bool -> 边界带宽区域 bool."""
    m = (mask > 0).float().unsqueeze(1)
    k = 2 * bandwidth + 1
    dil = F.max_pool2d(m, kernel_size=k, stride=1, padding=bandwidth)
    ero = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=bandwidth)
    edge = (dil - ero) > 0
    return edge.squeeze(1)


class SegmentationMeter:
    def __init__(self, num_classes: int = 2, boundary_bandwidth: int = 2):
        self.num_classes = num_classes
        self.boundary_bandwidth = boundary_bandwidth
        self.reset()

    def reset(self):
        device = torch.device("cpu")
        self.hist = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64, device=device)
        self.inter = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.union_d = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.b_inter = 0.0
        self.b_union = 0.0
        self.total_correct = 0.0
        self.total_pixels = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = logits.argmax(dim=1)
        targets = targets.detach().cpu()
        preds = preds.detach().cpu()

        self.hist += _fast_hist(preds, targets, self.num_classes).double()
        self.total_correct += (preds == targets).sum().item()
        self.total_pixels += targets.numel()

        for c in range(self.num_classes):
            p = preds == c
            t = targets == c
            self.inter[c] += (p & t).sum().item()
            self.union_d[c] += (p | t).sum().item()  # for IoU via hist better; dice uses inter

        # Boundary IoU on building class (1)
        if self.num_classes >= 2:
            pb = boundary_mask(preds, self.boundary_bandwidth)
            tb = boundary_mask(targets, self.boundary_bandwidth)
            self.b_inter += (pb & tb).sum().item()
            self.b_union += (pb | tb).sum().item()

    def compute(self) -> dict[str, float]:
        hist = self.hist
        iu = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + 1e-6)
        miou = torch.nanmean(iu).item()
        # building class IoU if binary
        iou_building = iu[1].item() if self.num_classes > 1 else iu[0].item()

        # dice_c = 2*TP / (2*TP + FP + FN) = 2*diag / (sum_row + sum_col)
        dice_c = (2 * torch.diag(hist)) / (hist.sum(1) + hist.sum(0) + 1e-6)
        mdice = torch.nanmean(dice_c).item()

        pa = self.total_correct / max(self.total_pixels, 1.0)

        # precision / recall for building
        if self.num_classes > 1:
            tp = hist[1, 1].item()
            fp = hist[:, 1].sum().item() - tp
            fn = hist[1, :].sum().item() - tp
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
        else:
            precision = recall = 0.0

        biou = self.b_inter / (self.b_union + 1e-6)

        return {
            "mIoU": float(miou),
            "IoU_building": float(iou_building),
            "Dice": float(mdice),
            "PA": float(pa),
            "Precision": float(precision),
            "Recall": float(recall),
            "BoundaryIoU": float(biou),
        }
