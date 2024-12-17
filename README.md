# Brain-Tumor-Classification

## Check out the project
https://brain-tumor-classification-syw4urqyf2jue5aftr43eg.streamlit.app/

## Running Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/aaespino238/Brain-Tumor-Classification.git
   cd Brain-Tumor-Classification
2. **Install Dependencies:**
   ```bash
   pip install requirements.txt
3. **Set Up Environment:**
   Create a .env file in the root directory and configure one or more of the following variables
   ```bash
   GEMINI_API_KEY=your-gemini-api-key
   OPENAI_API_KEY=your-openai-api-key
   GROQ_API_KEY=your-grok-api-key
   ```
   Support for additional models can be added by creating a proper provider file in utils/aisuite/providers/ and adding it to the MODELS list with the appropriate name in utils/llm_utils.py. Refer to https://github.com/andrewyng/aisuite for more information.
5. **Run the Application Locally:**
   ```bash
   streamlit run app.py

## Additional Information
- All model training related code is located is src
- Folder structure and files were heavily inspired by https://github.com/oarriaga/face_classification/tree/master
- Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
   - Dataset contains a number of duplicates, duplicate removal code is located in src/utils/remove_duplicates.ipynb

- Trained models can be downloaded using the download_models function located in utils/models.py.
     - Code for download_models was referenced from https://medium.com/@bridog314/deploying-large-ml-model-files-to-streamlit-using-google-drive-0818b0d416c9.

