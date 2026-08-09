import json
import torch

state_dict = torch.load("best_model.pt", map_location="cpu", weights_only=False)

with open("crop_to_idx.json") as f:
    crop_to_idx = json.load(f)
with open("disease_to_idx_per_crop.json") as f:
    disease_to_idx_per_crop = json.load(f)

torch.save({
    "model_state": state_dict,
    "crop_to_idx": crop_to_idx,
    "disease_to_idx_per_crop": disease_to_idx_per_crop,
    "img_size": 224,       # whatever --img_size you actually trained with
    "val_macro_f1": None,  # unknown for this run — fine, app.py handles None
}, "best_model.pt")