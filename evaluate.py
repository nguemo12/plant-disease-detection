"""
Per-crop evaluation on the test split.

Deliberately does NOT report one blended accuracy number -- a strong head
can hide a failing one. Instead: classification_report and confusion
matrix per crop head, plus a summary table across crops.
"""

import argparse
import json
from collections import defaultdict

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset
from model import SharedBackboneMultiHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", default="dataset_index.csv")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--crop_map", default="crop_to_idx.json")
    parser.add_argument("--disease_map", default="disease_to_idx_per_crop.json")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dataset_root", default=None,
                         help="Local root replacing the CSV's Kaggle path prefix, "
                              "e.g. 'kaggle/input/datasets/vipoooool/new-plant-diseases-dataset'")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: this checkpoint is a raw state_dict (just weights, no metadata
    # wrapper) -- so crop/disease maps and img_size come from the JSON
    # files and --img_size instead of from the checkpoint itself.
    with open(args.crop_map) as f:
        crop_to_idx = json.load(f)
    with open(args.disease_map) as f:
        disease_to_idx_per_crop = json.load(f)

    state_dict = torch.load(args.checkpoint, map_location=device)

    model = SharedBackboneMultiHead(crop_to_idx, disease_to_idx_per_crop).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = PlantDiseaseDataset(args.index_csv, split=args.split, img_size=args.img_size,
                              dataset_root=args.dataset_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    true_by_crop = defaultdict(list)
    pred_by_crop = defaultdict(list)

    with torch.no_grad():
        for images, crop_idx, disease_idx in loader:
            images = images.to(device)
            outputs = model(images, crop_idx.to(device))
            for positions, crop_name, logits in outputs:
                preds = logits.argmax(dim=1).cpu().tolist()
                targets = disease_idx[positions].tolist()
                pred_by_crop[crop_name].extend(preds)
                true_by_crop[crop_name].extend(targets)

    summary_rows = []
    for crop in sorted(true_by_crop):
        y_true, y_pred = true_by_crop[crop], pred_by_crop[crop]
        idx_to_disease = model.idx_to_disease_per_crop[crop]
        labels = sorted(idx_to_disease)
        target_names = [idx_to_disease[i] for i in labels]

        print(f"\n=== {crop} ===")
        print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(f"  labels: {target_names}")
        print(confusion_matrix(y_true, y_pred, labels=labels))

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
        summary_rows.append((crop, len(y_true), acc, macro_f1))

    print(f"\n=== Summary ({args.split} split) ===")
    print(f"{'crop':25s} {'n':>6s} {'accuracy':>10s} {'macro-F1':>10s}")
    for crop, n, acc, macro_f1 in summary_rows:
        flag = "  <- weak head" if macro_f1 < 0.6 else ""
        print(f"{crop:25s} {n:6d} {acc:10.3f} {macro_f1:10.3f}{flag}")


if __name__ == "__main__":
    main()