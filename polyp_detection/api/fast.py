import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from polyp_detection.ml_logic.model import prediction  # Import your prediction function
from polyp_detection.params import image_size, batch_size, chunk_num
import cv2
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
# Ensure that you use the path where the model is saved
model_path = "models/vgg19_model_20240905_140342.h5"
if os.path.exists(model_path):
    trained_model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}")

# Define categories for predictions
categories = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'ulcerative-colitis',
    'polyps'
]

# Preprocess the uploaded image to match the model's input size and format
def preprocess_uploaded_image(image_bytes):
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image to the expected size (e.g., 100x100)
    img_resized = cv2.resize(img, image_size[:2])

    # Normalize the image
    img_normalized = img_resized / 255.0

    # Return the preprocessed image as a NumPy array
    return np.expand_dims(img_normalized, axis=0)  # Add batch dimension

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image bytes from the uploaded file
        image_bytes = await file.read()

        # Preprocess the image
        preprocessed_image = preprocess_uploaded_image(image_bytes)

        # Use the loaded model to predict
        y_pred = trained_model.predict(preprocessed_image)

        # Get the predicted class index and label
        predicted_class_index = np.argmax(y_pred, axis=1)[0]
        predicted_class_label = categories[predicted_class_index]

        # Return the prediction as a JSON response
        return JSONResponse(content={"predicted_class": predicted_class_label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional: Add a root endpoint to check the API status
@app.get("/")
def read_root():
    return {"message": "Polyp Detection API is running"}
