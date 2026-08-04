"""
Dataset indexer for the Shared-Backbone Multi-Head CNN project.

Expects folders named "<Crop>___<Disease>" (PlantVillage-style), e.g.:
    Corn_(maize)___Northern_Leaf_Blight
    Corn_(maize)___healthy
    Apple___Apple_scab

Produces:
  - dataset_index.csv             : path, crop, disease, crop_idx, disease_idx
  - crop_to_idx.json               : crop name -> crop id (selects which head to use)
  - disease_to_idx_per_crop.json   : {crop: {disease: local class id within that crop's head}}

crop_idx and disease_idx are exactly what you feed into the masked
multi-task loss: crop_idx picks the active head, disease_idx is the
target for that head's softmax.
"""

import json
import re
from pathlib import Path

import pandas as pd

DATASET_ROOT = Path("data")  # <-- set this to your dataset folder
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_folder_name(folder_name: str):
    """Split 'Corn_(maize)___Northern_Leaf_Blight' into (crop, disease)."""
    crop_raw, _, disease_raw = folder_name.partition("___")
    # drop parenthetical aside e.g. "(maize)", underscores -> spaces, lowercase
    crop = re.sub(r"\(.*?\)", "", crop_raw).strip("_ ").replace("_", " ").strip().lower()
    disease = disease_raw.replace("_", " ").strip().lower()
    return crop, disease


def build_index(dataset_root: Path) -> pd.DataFrame:
    records = []
    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        crop, disease = parse_folder_name(folder.name)
        for img_path in folder.iterdir():
            if img_path.suffix.lower() in IMG_EXTS:
                records.append({"path": str(img_path), "crop": crop, "disease": disease})
    return pd.DataFrame(records)


def build_label_maps(df: pd.DataFrame):
    crop_to_idx = {crop: i for i, crop in enumerate(sorted(df["crop"].unique()))}

    disease_to_idx_per_crop = {}
    for crop, group in df.groupby("crop"):
        diseases = sorted(group["disease"].unique())
        disease_to_idx_per_crop[crop] = {d: i for i, d in enumerate(diseases)}

    return crop_to_idx, disease_to_idx_per_crop


def main():
    df = build_index(DATASET_ROOT)
    if df.empty:
        raise SystemExit(f"No images found under {DATASET_ROOT} — check the path.")

    crop_to_idx, disease_to_idx_per_crop = build_label_maps(df)

    df["crop_idx"] = df["crop"].map(crop_to_idx)
    df["disease_idx"] = df.apply(
        lambda r: disease_to_idx_per_crop[r["crop"]][r["disease"]], axis=1
    )

    df.to_csv("dataset_index.csv", index=False)
    with open("crop_to_idx.json", "w") as f:
        json.dump(crop_to_idx, f, indent=2, ensure_ascii=False)
    with open("disease_to_idx_per_crop.json", "w") as f:
        json.dump(disease_to_idx_per_crop, f, indent=2, ensure_ascii=False)

    print(f"{len(df)} images across {len(crop_to_idx)} crops\n")
    for crop, diseases in disease_to_idx_per_crop.items():
        print(f"  {crop}: {len(diseases)} classes -> {list(diseases)}")


if __name__ == "__main__":
    main()