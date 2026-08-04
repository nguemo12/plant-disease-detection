"""
PyTorch Dataset for the Shared-Backbone Multi-Head CNN project.

Reads dataset_index.csv (with a 'split' column added by make_splits.py)
and serves (image_tensor, crop_idx, disease_idx) triples.
"""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train: bool, img_size: int = 224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # mild color jitter only -- color/stripe pattern is the diagnostic
            # signal for many of these diseases, so we don't want to distort it
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class PlantDiseaseDataset(Dataset):
    def __init__(self, index_csv: str, split: str, img_size: int = 224, transform=None):
        df = pd.read_csv(index_csv)
        if "split" not in df.columns:
            raise ValueError(
                f"{index_csv} has no 'split' column -- run make_splits.py first"
            )
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {index_csv}")
        self.transform = transform or build_transform(train=(split == "train"), img_size=img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        crop_idx = torch.tensor(int(row["crop_idx"]), dtype=torch.long)
        disease_idx = torch.tensor(int(row["disease_idx"]), dtype=torch.long)
        return image, crop_idx, disease_idx