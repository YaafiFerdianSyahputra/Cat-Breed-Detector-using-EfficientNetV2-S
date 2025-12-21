# ======================================================
# Cat Breed Detector 🐱
# Streamlit App with Enhanced UI/UX + Image URL Upload
# ======================================================

import streamlit as st
import torch
import torch.nn as nn
# --- [BARU] Import untuk Faster R-CNN ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# ----------------------------------------
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
# Image Transform (EfficientNet)
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
# Load Model (Classifier)
# ------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading Classifier model..."):
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
# [BARU] Load Detector (Faster R-CNN) & Crop Logic
# ------------------------------------------------------
@st.cache_resource
def load_detector():
    # Menggunakan Faster R-CNN pre-trained (COCO weights)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    det_model = fasterrcnn_resnet50_fpn(weights=weights)
    det_model.to(DEVICE)
    det_model.eval()
    return det_model

def crop_cat_from_image(det_model, original_image):
    # Transform sederhana untuk detector (hanya ubah ke Tensor)
    det_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = det_transform(original_image).to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        predictions = det_model(img_tensor)[0]

    # COCO Class ID untuk 'Cat' adalah 17
    # Kita cari label 17 dengan confidence score > 0.5
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    cat_indices = [i for i, label in enumerate(labels) if label == 17 and scores[i] > 0.5]

    if len(cat_indices) > 0:
        # Ambil kucing dengan score tertinggi
        best_idx = cat_indices[0]
        box = boxes[best_idx].cpu().numpy()
        
        # Koordinat box: x_min, y_min, x_max, y_max
        # Crop image menggunakan PIL
        cropped_img = original_image.crop((box[0], box[1], box[2], box[3]))
        return cropped_img, True # Return True jika berhasil crop
    
    return original_image, False # Return False jika tidak ketemu kucing

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
"""
)

# --- [BARU] Checkbox Auto-Crop ---
use_autocrop = st.checkbox("✂️ Gunakan Auto-Crop (Faster R-CNN)", value=True, help="Otomatis memotong background agar fokus ke kucing.")

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
    # Simpan gambar asli untuk preview awal
    original_display = image.copy()
    image_to_process = image

    # --- [BARU] Proses Auto-Crop jika dicentang ---
    if use_autocrop:
        with st.spinner("✂️ Detecting & Cropping cat area..."):
            detector = load_detector() # Load model detector (cached)
            image_to_process, was_cropped = crop_cat_from_image(detector, image)
            
            if was_cropped:
                st.success("✅ Cat detected & cropped!")
            else:
                st.warning("⚠️ No cat detected clearly. Using full image.")

    st.subheader("🖼️ Image Preview")
    
    # Tampilkan gambar berdampingan (Original vs Cropped) jika di-crop
    if use_autocrop and 'was_cropped' in locals() and was_cropped:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Original")
            st.image(original_display, use_container_width=True)
        with col2:
            st.caption("Auto-Cropped Input")
            st.image(image_to_process, use_container_width=True)
    else:
        st.image(image_to_process, use_container_width=True)

    # Lanjut ke proses normal (EfficientNet)
    input_tensor = transform(image_to_process).unsqueeze(0).to(DEVICE)

    with st.spinner("🔍 Analyzing breed..."):
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
