# ======================================================
# Cat Breed Detector 🐱
# Streamlit App with Enhanced UI/UX + Image URL Upload
# ======================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import urllib.request
import requests
from io import BytesIO

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="Cat Breed Detector 🐱",
    page_icon="🐱",
    layout="centered"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_URL = (
    "https://huggingface.co/nabielherdiana/"
    "cat-breed-efficientnetv2/resolve/main/"
    "efficientnet_v2s_aug-model.pth"
)
MODEL_PATH = "efficientnet_v2s_aug-model.pth"
NUM_CLASSES = 20

# ------------------------------------------------------
# Class Labels
# ------------------------------------------------------
IDX_TO_CLASS = {
    0: "Abyssinian",
    1: "American Shorthair",
    2: "Bengal",
    3: "Birman",
    4: "British Shorthair",
    5: "Domestic Long Hair",
    6: "Domestic Shorthair",
    7: "Exotic Shorthair",
    8: "Himalayan",
    9: "Maine Coon",
    10: "Norwegian Forest",
    11: "Oriental Short Hair",
    12: "Persian",
    13: "Ragdoll",
    14: "Russian Blue",
    15: "Scottish Fold",
    16: "Siamese",
    17: "Sphynx",
    18: "Turkish Angora",
    19: "Turkish Van",
}

# ------------------------------------------------------
# Image Transform
# ------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ------------------------------------------------------
# Load Model
# ------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Hugging Face..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    model = nn.Sequential(
        model,
        nn.LogSoftmax(dim=1)
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ------------------------------------------------------
# Helper: Load image from URL
# ------------------------------------------------------
def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.title("🐱 Cat Breed Detector")
st.markdown(
    """
A deep learning–based application that predicts **cat breeds** using a  
**fine-tuned EfficientNetV2-S** model.

Choose **one input method** below:
"""
)

st.divider()

# ------------------------------------------------------
# Input Section
# ------------------------------------------------------
input_method = st.radio(
    "Select image input method:",
    ["Upload Image File", "Paste Image URL"],
    horizontal=True
)

image = None

if input_method == "Upload Image File":
    uploaded_file = st.file_uploader(
        "Upload a cat image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

else:
    image_url = st.text_input(
        "Paste image URL",
        placeholder="https://example.com/cat.jpg"
    )
    if image_url:
        try:
            image = load_image_from_url(image_url)
        except Exception:
            st.error("❌ Failed to load image from URL. Please check the link.")

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------
if image is not None:
    st.subheader("🖼️ Image Preview")
    st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with st.spinner("🔍 Analyzing image..."):
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.exp(output)[0]
            top_probs, top_idxs = torch.topk(probs, 3)

    st.subheader("🔮 Prediction Result")

    for rank, (idx, prob) in enumerate(zip(top_idxs, top_probs), start=1):
        st.markdown(
            f"**{rank}. {IDX_TO_CLASS[idx.item()]}** — "
            f"`{prob.item() * 100:.2f}%`"
        )

    st.progress(float(top_probs[0]))

    st.caption(
        "⚠️ Predictions are based on the visual features of the uploaded image "
        "and may vary depending on image quality."
    )
