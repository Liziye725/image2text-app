import easyocr as ocr  #OCR
import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 

# title
st.title("Easy OCR - Extract Text from Images")
st.markdown("## Optical Character Recognition - Using `easyocr`, `streamlit` -  hosted on 🤗 Spaces")
st.markdown("Link to the app - [image-to-text-app on 🤗 Spaces](https://huggingface.co/spaces/Amrrs/image-to-text-app)")

# Upload image
image = st.sidebar.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

# Cashe loading - OCR model
@st.cache
def load_model(): 
    reader = ocr.Reader(['en', 'ch'],model_storage_directory='.')
    return reader 

reader = load_model()

if image is not None:

    input_image = Image.open(image) # read image
    st.image(input_image, caption="Uploaded Image", use_column_width=True) # display image

    with st.spinner("🤖 AI is processing... "):
        result = reader.readtext(np.array(input_image))
    
    # OCR results
    extracted_text = []
    for text in result:
        extracted_text.append(f"📌 {text[1]} (Confidence: {text[2]:.2f})")

    st.write(extracted_text)

    # show results
    st.success("✅ OCR Extraction Complete!")
    st.text_area("Extracted Text:", "\n".join(extracted_text), height=200)
    
    st.balloons()
else:
    st.warning("⚠ Please upload an image to proceed.")

st.caption("Reference: @1littlecoder. Credits to 🤗 Spaces for Hosting this ")
