import streamlit as st
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os

def download_models():
    # Create a models directory in the Streamlit environment
    os.makedirs('models', exist_ok=True)

    # Write in your Google Drive File IDs

    # https://drive.google.com/file/d/1pRMGpPRSQawcdvCku--KEtGo7uc0cHiF/view?usp=drive_link
    mini_xception_id ="1pRMGpPRSQawcdvCku--KEtGo7uc0cHiF"

    # https://drive.google.com/file/d/1GFgp7REhNLhlrT5gHrZRHj8f4qPFLpNj/view?usp=drive_link
    xception_id="1GFgp7REhNLhlrT5gHrZRHj8f4qPFLpNj"

    # Create path vars for each new streamlit file
    mini_xception_path = 'models/mini_xception_model.h5'
    xception_path = 'models/xception_model.h5'

    #Safely download each to the Streamlit Server
    if not os.path.exists(mini_xception_path):
        with st.spinner('Downloading CNN model ... '):
            url = f"https://drive.google.com/uc?id={mini_xception_id}"
            gdown. download(url, mini_xception_path, quiet=False)

    if not os.path.exists(xception_path):
        with st.spinner('Downloading Xception model ... '):
            url = f"https://drive.google.com/uc?id={xception_id}"
            gdown. download(url, xception_path, quiet=False)

@st.cache_resource
def load_xception():
    return load_model("models/xception_model.h5")

@st.cache_resource
def load_mini_xception():
    return load_model("models/mini_xception_model.h5")

def load_models(selected_models):
    models = []
    image_sizes = []
    for model_name in selected_models:
        if model_name == "Transfer Learning - Xception":
            models.append(load_xception())
            image_sizes.append((299,299))
        elif model_name == "Mini Xception":
            models.append(load_mini_xception())
            image_sizes.append((224,224))
    
    return models, image_sizes

def preprocess_image(image, image_shape):
    img = image.convert('RGB')
    img = img.resize(image_shape, Image.Resampling.NEAREST)  # Resize image
    img_array = img_to_array(img) # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0
    return img_array

def generate_saliency_map(model, image, class_index, img_size, input_shape):
    img_array = preprocess_image(image, input_shape)
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]
        gradients = tape.gradient(target_class, img_tensor)
        gradients = tf.math.abs(gradients)
        gradients = tf.reduce_max(gradients, axis=-1)
        gradients = gradients.numpy().squeeze()

    # Resize gradients to match original image size
    gradients = cv2.resize(gradients, img_size)

    # Create a circular mask for the brain area
    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

    # Apply mask to gradients
    gradients = gradients * mask

    # Normalize only the brain area
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
        brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
    gradients[mask] = brain_gradients

    # Apply a higher threshold
    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0

    # Apply more aggressive smoothing
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    # Create a heatmap overlay with enhanced contrast
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, img_size)

    # Superimpose the heatmap on original image with increased opacity
    original_img = img_to_array(image)
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    # Convert superimposed image to PIL Image end encode as a Base64
    pil_image = Image.fromarray(superimposed_img)

    return pil_image 