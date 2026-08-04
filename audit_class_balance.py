"""
Class balance audit for the Shared-Backbone Multi-Head CNN project.

Reads dataset_index.csv (produced by build_dataset_index.py) and reports,
per crop and per class within each crop's head:
  - image counts
  - imbalance ratio (largest class / smallest class) within each head
  - flags for classes/crops that are too thin to train reliably

Use this BEFORE picking backbone depth or head design — it tells you
whether you need heavier augmentation, oversampling, or to prune a crop.
"""

import pandas as pd

INDEX_CSV = "dataset_index.csv"
MIN_CLASS_IMAGES = 200   # below this, a class is high risk for overfitting
MIN_CROP_IMAGES = 500    # below this, consider dropping the crop entirely


def main():
    df = pd.read_csv(INDEX_CSV)

    print("=== Per-crop totals ===")
    crop_totals = df.groupby("crop").size().sort_values(ascending=False)
    for crop, count in crop_totals.items():
        flag = "  <- LOW: consider pruning or heavy augmentation" if count < MIN_CROP_IMAGES else ""
        print(f"  {crop:25s} {count:6d} images, {df[df.crop == crop]['disease'].nunique()} classes{flag}")

    print("\n=== Per-class imbalance within each head ===")
    for crop, group in df.groupby("crop"):
        counts = group.groupby("disease").size().sort_values(ascending=False)
        ratio = counts.max() / counts.min()
        print(f"\n  {crop}  (imbalance ratio {ratio:.1f}x)")
        for disease, count in counts.items():
            flag = "  <- LOW: augment or oversample" if count < MIN_CLASS_IMAGES else ""
            print(f"    {disease:35s} {count:6d}{flag}")

    print("\n=== Summary ===")
    thin_crops = crop_totals[crop_totals < MIN_CROP_IMAGES].index.tolist()
    thin_classes = (
        df.groupby(["crop", "disease"]).size()
        .reset_index(name="count")
        .query("count < @MIN_CLASS_IMAGES")
    )
    print(f"Crops below {MIN_CROP_IMAGES} images: {thin_crops or 'none'}")
    print(f"Classes below {MIN_CLASS_IMAGES} images: {len(thin_classes)}")
    if len(thin_classes):
        print(thin_classes.to_string(index=False))


if __name__ == "__main__":
    main()