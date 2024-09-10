# 🔬 Polyp Detection from Endoscopy Images

This repository contains a deep learning-based web application for detecting polyps in endoscopy images. The app leverages a pre-trained VGG19 model and is deployed using **FastAPI** for backend inference and **Streamlit** for the frontend interface.


## Dataset
The dataset used for training the model belongs to Kaggle and can be found here: [Kvasir Dataset](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset). The dataset includes labeled endoscopy images from various categories related to gastrointestinal conditions. The data is preprocessed using image augmentation techniques and split into training and validation sets for fine-tuning the VGG19 model.

## 🌟 Features
- **Polyp Detection**: Automatically detects polyps from endoscopy images using a trained deep learning model.
- **Real-Time Inference**: Upload an image and receive a category prediction in real time.
- **User-Friendly Interface**: Built using **Streamlit** for easy interaction and visualization.
- **State-of-the-art Model**: Based on the VGG19 architecture, fine-tuned for medical image analysis.

## 🚀 Demo
Upload an endoscopy image in `JPG`, `JPEG`, or `PNG` format, and the model will classify the image into one of the following categories:
- Dyed-lifted polyps
- Dyed resection margins
- Esophagitis
- Normal cecum
- Normal pylorus
- Normal Z-line
- Ulcerative colitis
- Polyps

## 🔧 Setup and Installation

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **TensorFlow 2.0+**
- **Streamlit**
- **FastAPI**
- **OpenCV**

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/polyp-detection-endoscopy.git
cd polyp-detection-endoscopy
```

### 2. Install Dependencies

Install the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI Backend
Start the FastAPI server for model inference:

```bash
uvicorn fastapi_app:app --reload
```

### 3. Run the Streamlit Frontend
In a separate terminal window, start the Streamlit frontend:

```bash
streamlit run app.py
```
