import streamlit as st
from PIL import Image

from utils.gemini_utils import validate_image
from utils.displays import model_results
from utils.models import download_models

download_models()
model_names = ["Transfer Learning - Xception", "Mini Xception"]

st.title("Brain Tumor Classification")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((120,120))
        st.image(image, caption="Uploaded File")
    
    selected_models = st.multiselect("Classification Models", options=model_names, default=[model_names[0]], key="model_selection")
    
    submit = st.button("Submit")

if submit:
    if not uploaded_file:
        st.error("Please upload a human brain MRI scan.")
    else:
        image = Image.open(uploaded_file)
        file_name = uploaded_file.name

        with st.spinner("Validating image..."):
            imageIsBrainScan = validate_image(image, file_name)
            
        if not imageIsBrainScan:
            st.error("The uploaded image is not a brain MRI scan, please only upload human brain MRI scans.")
        else:
            model_results(selected_models, image, file_name)