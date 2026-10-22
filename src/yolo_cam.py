"""Helpers to run Grad-CAM on an Ultralytics YOLOv8 classification model."""
import torch
from torch import nn
from ultralytics import YOLO


class LogitsOnly(nn.Module):
    """Some Ultralytics versions return a tuple from the classification head.
    Grad-CAM needs a plain tensor, so keep only the first output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def load_yolo_cls(path, device=None):
    """Return (wrapped torch model, target layer for Grad-CAM, class names, device)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(path)
    core = yolo.model.to(device).eval()
    # Last feature layer before the classification head (YOLOv8-cls)
    target_layer = core.model[-2]
    class_names = [yolo.names[i] for i in range(len(yolo.names))]
    return LogitsOnly(core).eval(), target_layer, class_names, device
