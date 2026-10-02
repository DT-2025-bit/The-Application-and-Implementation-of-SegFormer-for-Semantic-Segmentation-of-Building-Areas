from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHF(nn.Module):
    """HuggingFace SegFormer 封装，输出 B,C,H,W logits。"""

    def __init__(self, encoder: str = "mit-b0", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        model_id = f"nvidia/{encoder}"
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits  # B,C,h,w (下采样)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class SMPModel(nn.Module):
    def __init__(self, arch: str, encoder: str, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        encoder_weights = "imagenet" if pretrained else None
        if arch == "unet":
            self.model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        elif arch == "deeplabv3plus":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(model_name: str, encoder: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = model_name.lower()
    if "segformer" in name:
        return SegFormerHF(encoder=encoder, num_classes=num_classes, pretrained=pretrained)
    if "unet" in name:
        return SMPModel("unet", encoder=encoder, num_classes=num_classes, pretrained=pretrained)
    if "deeplab" in name:
        return SMPModel("deeplabv3plus", encoder=encoder, num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {model_name}")
