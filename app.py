"""
Streamlit deployment for the Shared-Backbone Multi-Head CNN.

The user picks the crop explicitly (no crop auto-detection), so
inference always routes through exactly the head that was trained
and evaluated for that crop.
"""

import torch
import streamlit as st
from PIL import Image

from dataset import build_transform
from model import SharedBackboneMultiHead

CHECKPOINT_PATH = "best_model.pt"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model = SharedBackboneMultiHead(ckpt["crop_to_idx"], ckpt["disease_to_idx_per_crop"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device, ckpt["img_size"], ckpt.get("val_macro_f1")


st.set_page_config(page_title="Crop disease diagnosis", page_icon=":herb:")
st.title("Crop-specific plant disease diagnosis")
st.write(
    "Select the crop, upload a leaf photo, and get a prediction from that "
    "crop's dedicated classification head."
)

try:
    model, device, img_size, val_f1 = load_model()
except FileNotFoundError:
    st.error(
        f"No checkpoint found at '{CHECKPOINT_PATH}'. Run train.py first, "
        f"then re-launch this app from the same directory as the checkpoint."
    )
    st.stop()

crop_names = sorted(model.crop_to_idx.keys())

with st.sidebar:
    st.subheader("Model")
    if val_f1 is not None:
        st.caption(f"Validation macro-F1 at training time: {val_f1:.3f}")
    st.caption(f"{len(crop_names)} crops, {sum(len(d) for d in model.disease_to_idx_per_crop.values())} disease classes total")

crop = st.selectbox("Crop", crop_names, format_func=str.title)
uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    transform = build_transform(train=False, img_size=img_size)
    tensor = transform(image).to(device)

    n_classes = len(model.disease_to_idx_per_crop[crop])
    with st.spinner("Running diagnosis..."):
        predictions = model.predict(tensor, crop, topk=min(5, n_classes))

    st.subheader("Prediction")
    top_disease, top_prob = predictions[0]
    st.metric(
        label=f"Most likely ({crop.title()})",
        value=top_disease.title(),
        delta=f"{top_prob:.1%} confidence",
    )

    st.write("All candidates for this crop:")
    for disease, prob in predictions:
        st.progress(prob, text=f"{disease.title()} — {prob:.1%}")

    st.caption(
        "This model only sees the head for the crop you selected above -- "
        "it does not attempt to identify the crop itself. Prediction "
        "quality depends on how much training data that crop's head had; "
        "check the per-crop evaluation report before trusting a thin head."
    )
else:
    st.info("Upload a leaf image to get a prediction.")