
import streamlit as st
import requests
from PIL import Image
import io

# FastAPI URL
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

# Streamlit App Configuration
st.set_page_config(
    page_title="Polyp Detection from Endoscopy Images",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and Introduction
st.title("🔬 Polyp Detection from Endoscopy Images")
st.markdown("""
    <div style="text-align: center; font-size: 1.2em;">
        Upload an endoscopy image to detect polyps using a state-of-the-art pre-trained model.
        <br><br>
    </div>
    """, unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader(
    "📤 Upload your endoscopy image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# If an image is uploaded
if uploaded_file is not None:
    # Convert the image to bytes
    image = Image.open(uploaded_file)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes = img_bytes.getvalue()

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        if st.button("🔍 Detect Polyps"):
            with st.spinner('Analyzing the image...'):
                try:
                    # Send the image to the FastAPI prediction endpoint
                    files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                    response = requests.post(FASTAPI_URL, files=files)

                    # Handle the response
                    if response.status_code == 200:
                        result = response.json()
                        predicted_class = result.get("predicted_class", "No prediction")

                        # Display the prediction
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background-color: #F0F2F6; border-radius: 10px; margin-top: 20px;">
                            <h2 style="color: #4CAF50;">✅ Detected Category:</h2>
                            <h1 style="font-size: 2.5em; color: #FF6347;">{predicted_class}</h1>
                        </div>
                        """, unsafe_allow_html=True)

                    else:
                        st.error(f"Error {response.status_code}: Unable to get the prediction.")
                        st.write(response.json())

                except Exception as e:
                    st.error(f"An error occurred while processing the image: {e}")

# Footer with Information
st.markdown("""
    <hr>
    <div style="text-align: center;">
        <p style="font-size: 1em;">This tool uses machine learning to detect potential polyps in endoscopy images. Please consult a healthcare professional for accurate diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
