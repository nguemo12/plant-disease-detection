"""
Shared-Backbone Multi-Head CNN.

A single custom conv backbone extracts general leaf texture/lesion features.
Each crop gets its own small classification head on top of the shared
feature vector. Routing at train time uses the known crop label; at
inference time (Streamlit) the user-selected crop picks the head directly.
"""

import json

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SharedBackbone(nn.Module):
    """4-block conv backbone. 224x224 input -> pooled feature vector.

    Baseline depth -- see project notes: start here, only add a block if
    train/val curves show underfitting (both plateau low with a small gap).
    If train pulls away from val, add regularization instead of depth.
    """

    def __init__(self, in_ch: int = 3, widths=(32, 64, 128, 256), dropouts=(0.1, 0.2, 0.3, 0.4)):
        super().__init__()
        assert len(widths) == len(dropouts)
        blocks = []
        ch = in_ch
        for w, d in zip(widths, dropouts):
            blocks.append(ConvBlock(ch, w, d))
            ch = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return x


class CropHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SharedBackboneMultiHead(nn.Module):
    def __init__(self, crop_to_idx: dict, disease_to_idx_per_crop: dict, backbone_kwargs: dict = None):
        super().__init__()
        self.crop_to_idx = crop_to_idx
        self.idx_to_crop = {v: k for k, v in crop_to_idx.items()}
        self.disease_to_idx_per_crop = disease_to_idx_per_crop
        self.idx_to_disease_per_crop = {
            crop: {v: k for k, v in mapping.items()}
            for crop, mapping in disease_to_idx_per_crop.items()
        }

        self.backbone = SharedBackbone(**(backbone_kwargs or {}))
        self.heads = nn.ModuleDict({
            crop: CropHead(self.backbone.out_dim, len(diseases))
            for crop, diseases in disease_to_idx_per_crop.items()
        })

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward_head(self, features: torch.Tensor, crop_name: str) -> torch.Tensor:
        return self.heads[crop_name](features)

    def forward(self, images: torch.Tensor, crop_idx: torch.Tensor):
        """
        Training-time forward pass over a batch that may mix crops.

        images: (B, C, H, W)
        crop_idx: (B,) long tensor, values index into self.idx_to_crop

        Returns a list of (batch_positions, crop_name, logits) -- one entry
        per distinct crop present in the batch. batch_positions lets the
        caller line logits back up with the right rows of the batch to
        compute a masked multi-task loss.
        """
        features = self.forward_features(images)
        outputs = []
        crop_idx_cpu = crop_idx.detach().cpu()
        for cidx in crop_idx_cpu.unique():
            crop_name = self.idx_to_crop[int(cidx)]
            positions = (crop_idx_cpu == cidx).nonzero(as_tuple=True)[0]
            logits = self.forward_head(features[positions], crop_name)
            outputs.append((positions, crop_name, logits))
        return outputs

    @torch.no_grad()
    def predict(self, image: torch.Tensor, crop_name: str, topk: int = 3):
        """
        Inference for a single image with a known crop (Streamlit use case).
        image: (1, C, H, W) or (C, H, W)
        Returns list of (disease_name, probability), sorted descending.
        """
        self.eval()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        features = self.forward_features(image)
        logits = self.forward_head(features, crop_name)
        probs = torch.softmax(logits, dim=1)[0]
        idx_to_disease = self.idx_to_disease_per_crop[crop_name]
        k = min(topk, probs.shape[0])
        top_probs, top_idx = probs.topk(k)
        return [(idx_to_disease[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]

    @classmethod
    def from_label_maps(cls, crop_to_idx_path: str, disease_to_idx_per_crop_path: str, backbone_kwargs: dict = None):
        with open(crop_to_idx_path) as f:
            crop_to_idx = json.load(f)
        with open(disease_to_idx_per_crop_path) as f:
            disease_to_idx_per_crop = json.load(f)
        return cls(crop_to_idx, disease_to_idx_per_crop, backbone_kwargs)