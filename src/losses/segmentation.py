from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: B,C,H,W ; targets: B,H,W
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        cardinality = torch.sum(probs + targets_oh, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def morphological_edge(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """从二值/类别掩膜提取边缘。mask: B,H,W (0/1 或 类别)."""
    if mask.dtype != torch.float32:
        m = (mask > 0).float().unsqueeze(1)
    else:
        m = mask.unsqueeze(1)
    pad = kernel_size // 2
    dilated = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-m, kernel_size=kernel_size, stride=1, padding=pad)
    edge = (dilated - eroded).clamp(0, 1)
    return edge.squeeze(1)


class EdgeLoss(nn.Module):
    """对预测建筑概率图与 GT 边缘做 BCE，增强轮廓连续性。"""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 建筑类默认 channel=1（二类：背景0 / 建筑1）
        if logits.shape[1] == 1:
            prob_building = torch.sigmoid(logits[:, 0])
        else:
            prob_building = F.softmax(logits, dim=1)[:, 1]

        gt_edge = morphological_edge(targets, self.kernel_size)
        pred_edge = morphological_edge(prob_building, self.kernel_size)
        # BCELoss 在 AMP autocast 下不安全，强制 float32 计算
        pred = pred_edge.float().clamp(1e-6, 1 - 1e-6)
        gt = gt_edge.float()
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            return F.binary_cross_entropy(pred, gt)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        edge_weight: float = 0.5,
        edge_kernel: int = 3,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss()
        self.edge = EdgeLoss(kernel_size=edge_kernel)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        total = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        out = {"loss": total, "ce": loss_ce.detach(), "dice": loss_dice.detach()}
        if self.edge_weight > 0:
            loss_edge = self.edge(logits, targets)
            total = total + self.edge_weight * loss_edge
            out["loss"] = total
            out["edge"] = loss_edge.detach()
        else:
            out["edge"] = torch.tensor(0.0, device=logits.device)
        return out
