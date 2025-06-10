import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
import cv2
import os 

st.title("Brain Tumor Classification with YOLOv8 + Grad-CAM")

@st.cache_resource(show_spinner=True)
def load_model(path):
    model = YOLO(path)
    torch_model = model.model
    torch_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_model.to(device)
    return torch_model, device

model_path = os.path.join(os.getcwd(), "models", "best.pt")
torch_model, device = load_model(model_path)

uploaded_file = st.file_uploader("Upload MRI Image (jpg, png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))

    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    with torch.no_grad():
        output = torch_model(input_tensor)[0]
    pred_class = int(torch.argmax(output, dim=1))

    class_names = ['GLIOMA', 'MENINGIOMA', 'notumor', 'PITUITARY']
    predicted_class_name = class_names[pred_class]

    st.write(f"**Predicted Class:** {predicted_class_name}")

    target_layer = torch_model.model[-2]
    cam = GradCAM(model=torch_model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(img).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5)


    st.image(visualization, caption=f'Grad-CAM Heatmap for {predicted_class_name}', use_column_width=True)
