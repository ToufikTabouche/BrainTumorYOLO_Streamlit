"""Grad-CAM explanation for a YOLOv8 classification model on one MRI image.

Usage (from the repository root):
    python -m src.gradcam --model best.pt --image path/to/mri.jpg [--out gradcam.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.yolo_cam import load_yolo_cls


def explain(model_path, image_path, img_size=224):
    model, target_layer, class_names, device = load_yolo_cls(model_path)

    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_class = int(torch.argmax(model(input_tensor)[0]))

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]

    rgb_img = np.array(img).astype(np.float32) / 255.0
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return img, heatmap, class_names[pred_class]


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for a YOLOv8 classification model")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None, help="Save the figure instead of showing it")
    args = parser.parse_args()

    img, heatmap, label = explain(args.model, args.image)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    ax[0].imshow(img)
    ax[0].set_title("Input MRI")
    ax[1].imshow(heatmap)
    ax[1].set_title(f"Grad-CAM\nPredicted: {label}")
    for a in ax:
        a.axis("off")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
        print(f"Saved {args.out} (predicted: {label})")
    else:
        plt.show()


if __name__ == "__main__":
    main()
