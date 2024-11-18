import streamlit as st
import numpy as np
import pandas as pd
from utils.models import load_models, generate_saliency_map, preprocess_image
from utils.gemini_utils import interpret_results

labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

def model_results(selected_models, image, file_name):
    with st.spinner("Loading models..."):
        models, image_sizes = load_models(selected_models)
    
    cols = st.columns(len(selected_models))

    for i in range(len(cols)):

        col = cols[i]
        model = models[i]
        img_size = image_sizes[i]
        model_name = selected_models[i]

        with col:                
            try:
                img_array = preprocess_image(image, img_size)

                predictions = model.predict(img_array)[0]

                class_index = np.argmax(predictions)
                predicted_class = labels[class_index]
                prediction_confidence = predictions[class_index]

                # Display predictions
                st.subheader(f"Prediction Result for {model_name}")
                st.write(f"**Class**: {predicted_class}")
                st.write(f"**Confidence**: {prediction_confidence*100}%")

                confidences = pd.DataFrame({"Class": labels, "Confidence": predictions})
                st.bar_chart(confidences.set_index("Class"))
                saliency_map = generate_saliency_map(model, image, class_index, (image.width, image.height), img_size)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                with col2:
                    st.image(saliency_map, caption="Saliency Map", use_container_width=True)

                with st.spinner("Interpreting saliency map..."):
                    explanation = interpret_results(model_name, file_name, saliency_map, predicted_class, prediction_confidence)

                st.write("## Explanation")
                st.write(explanation)
            except Exception as e:
                        st.error(f"An error occurred: {e}")