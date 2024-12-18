import google.generativeai as genai
import streamlit as st
import base64
import os
from PIL import Image
from io import BytesIO

from utils.aisuite.client import Client

client = Client()

MODELS = ["google:gemini-1.5-flash", "openai:gpt-4o-mini", "groq:mixtral-8x7b-32768"]

RESULT_INTERPRETATION_PROMPT = """
    You are an expert neurologist. You are tasked with explaining a sliency map of a brain
    tumor MRI scan. The saliency map was generated by a deep learning model that was trained
    to classify brain tumors as either glioma, meningioma, pituitary, or no tumor. 

    The saliency map highlights the regions of the image that the machine learning model
    is focusing on to make the prediction.

    The deep learning model predicted the image to be of class '{model_prediction}' with a 
    confidence of {confidence}.
    

    In your response:
    - Explain what regions of the brain the model is focusing on, based on the saliency map.
    Refer to the regions highlighted in light cyan, those are the regions where the model is 
    focusing on.
    - Explain possibly reasons why the model made the prediction it did.
    - !IMPORTANT! DO NOT MENTION ANYTHING LIKE 'The saliency map highlights the regions
    the model is focusing on, which are in light cyan'
    - !IMPORTANT! Keep your explanations to 4 sentences max.
"""

IMAGE_VALIDATION_PROMPT = """
Analyze the provided image and determine if it is a brain MRI scan. Return true if the image is a brain MRI scan and false if it is not. It is very important that you return only 'true' or 'false' as the response.
"""

def _encode_image(_image):
    if isinstance(_image, Image.Image):
        buffered = BytesIO()
        _image.save(buffered, format="JPEG")
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_string
    else:
        raise ValueError("Input must be a PIL Image")

# Function to validate whether the image is a human brain MRI scan
@st.cache_data
def validate_image(_image, file_name):
    base64_image = _encode_image(_image)
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": IMAGE_VALIDATION_PROMPT}, 
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.0
            )

            response_text = response.choices[0].message.content

            return True if response_text.strip().lower() == "true" else False
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    return False

# Function to interpret model results
@st.cache_data
def interpret_results(model_name, file_name, _saliency_map, _model_prediction, _confidence):
    prompt = RESULT_INTERPRETATION_PROMPT.format(
        model_prediction=_model_prediction,
        confidence=_confidence*100
    )
    base64_image = _encode_image(_saliency_map)

    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt}, 
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    return ""