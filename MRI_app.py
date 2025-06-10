import streamlit as st
import torch

st.title("MRI Brain Tumor Detection Demo")

st.write("Torch version:", torch.__version__)

# Dummy tensor computation to test torch is working
x = torch.tensor([1.0, 2.0, 3.0])
st.write("Tensor example:", x)
