"""
Training loop for the Shared-Backbone Multi-Head CNN.

Masked multi-task loss: every batch can mix crops. The model groups the
batch by crop, runs only the matching head on each group, and we sum
per-sample cross-entropy across all groups then divide by batch size --
so every image contributes exactly once, through its own crop's head,
and the shared backbone gets gradient signal from every crop each step.
"""

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import PlantDiseaseDataset
from model import SharedBackboneMultiHead


def masked_multitask_loss(model_outputs, disease_idx: torch.Tensor, device):
    """model_outputs: list of (positions, crop_name, logits) from model.forward()."""
    total_loss = images_seen = 0
    for positions, _crop_name, logits in model_outputs:
        targets = disease_idx[positions].to(device)
        total_loss = total_loss + F.cross_entropy(logits, targets, reduction="sum")
        images_seen += len(positions)
    return total_loss / images_seen


def run_epoch(model, loader, device, optimizer=None, log_prefix="train"):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_batches = 0.0, 0
    all_true, all_pred = [], []
    n_total_batches = len(loader)
    t0 = time.time()

    for batch_idx, (images, crop_idx, disease_idx) in enumerate(loader, start=1):
        images = images.to(device)
        crop_idx_dev = crop_idx.to(device)

        with torch.set_grad_enabled(train_mode):
            outputs = model(images, crop_idx_dev)
            loss = masked_multitask_loss(outputs, disease_idx, device)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        for positions, _crop_name, logits in outputs:
            preds = logits.argmax(dim=1).cpu()
            all_pred.extend(preds.tolist())
            all_true.extend(disease_idx[positions].tolist())

        # Print progress every batch for the first 5, then every 10th --
        # a silent script for a whole epoch is indistinguishable from a
        # hung one, so this is here specifically to make progress visible.
        if batch_idx <= 5 or batch_idx % 10 == 0 or batch_idx == n_total_batches:
            elapsed = time.time() - t0
            rate = batch_idx / elapsed if elapsed > 0 else 0
            print(f"    [{log_prefix}] batch {batch_idx}/{n_total_batches} "
                  f"| running loss {total_loss / n_batches:.4f} "
                  f"| {rate:.2f} batch/s", flush=True)

    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / n_batches, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", default="dataset_index.csv")
    parser.add_argument("--crop_map", default="crop_to_idx.json")
    parser.add_argument("--disease_map", default="disease_to_idx_per_crop.json")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6, help="early stop if val macro-F1 doesn't improve")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: training on CPU -- a full epoch over a real PlantVillage-sized "
              "dataset can take a long time. Watch for per-batch progress below; if you "
              "see none for several minutes even on batch 1, see the num_workers note in "
              "the README/troubleshooting.", flush=True)

    train_ds = PlantDiseaseDataset(args.index_csv, split="train", img_size=args.img_size)
    val_ds = PlantDiseaseDataset(args.index_csv, split="val", img_size=args.img_size)
    print(f"train: {len(train_ds)} images, val: {len(val_ds)} images, "
          f"{len(train_ds) // args.batch_size + 1} batches/epoch", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SharedBackboneMultiHead.from_label_maps(args.crop_map, args.disease_map).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1, epochs_no_improve = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}", flush=True)
        train_loss, train_f1 = run_epoch(model, train_loader, device, optimizer, log_prefix="train")
        val_loss, val_f1 = run_epoch(model, val_loader, device, optimizer=None, log_prefix="val")
        scheduler.step(val_f1)

        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} f1 {train_f1:.4f} "
              f"| val loss {val_loss:.4f} f1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, epochs_no_improve = val_f1, 0
            torch.save({
                "model_state": model.state_dict(),
                "crop_to_idx": model.crop_to_idx,
                "disease_to_idx_per_crop": model.disease_to_idx_per_crop,
                "img_size": args.img_size,
                "val_macro_f1": val_f1,
            }, args.out)
            print(f"  -> saved new best model (val macro-F1 {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"No val improvement for {args.patience} epochs, stopping early.")
                break

    print(f"Best val macro-F1: {best_f1:.4f} -- checkpoint at {args.out}")


if __name__ == "__main__":
    main()