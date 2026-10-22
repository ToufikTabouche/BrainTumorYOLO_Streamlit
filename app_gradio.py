"""Gradio demo: brain tumor classification on MRI with a YOLOv8 classification model.

Run:  python app_gradio.py
"""
import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")


def predict(image):
    result = model(image, imgsz=224, verbose=False)[0]
    probs = result.probs.data.tolist()
    return {model.names[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Brain MRI"),
    outputs=gr.Label(num_top_classes=4, label="Prediction"),
    title="Brain Tumor Classification with YOLOv8",
    description="Upload a brain MRI image. Classes: glioma, meningioma, pituitary, no tumor. Research demo only.",
)

if __name__ == "__main__":
    demo.launch()
