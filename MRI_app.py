"""Streamlit demo: brain tumor classification on MRI with YOLOv8 + Grad-CAM.

Run:  streamlit run MRI_app.py
"""
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.yolo_cam import load_yolo_cls

MODEL_PATH = "best.pt"
IMG_SIZE = 224

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")
st.title("Brain Tumor Classification with YOLOv8 + Grad-CAM")
st.caption("Research demo, not a medical device. Classes: glioma, meningioma, pituitary, no tumor.")


@st.cache_resource(show_spinner=True)
def get_model():
    return load_yolo_cls(MODEL_PATH)


model, target_layer, class_names, device = get_model()

uploaded_file = st.file_uploader("Upload a brain MRI image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(input_tensor)[0]
    # YOLOv8-cls already returns probabilities in eval mode; normalise just in case
    probs = scores if torch.isclose(scores.sum(), torch.tensor(1.0, device=scores.device), atol=1e-3) else scores.softmax(0)
    pred_class = int(torch.argmax(probs))

    st.subheader(f"Prediction: {class_names[pred_class].upper()} ({float(probs[pred_class]):.1%})")

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]

    rgb_img = np.array(img).astype(np.float32) / 255.0
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Input MRI", use_container_width=True)
    col2.image(heatmap, caption="Grad-CAM: regions driving the prediction", use_container_width=True)

    with st.expander("Class probabilities"):
        for name, p in zip(class_names, probs.tolist()):
            st.write(f"{name}: {p:.1%}")
