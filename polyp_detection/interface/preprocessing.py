import cv2
from polyp_detection.params import *
import numpy as np


def enhance_image(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian Blur for noise reduction
    enhanced_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)

    return enhanced_img



def apply_threshold(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold
    _, thresh_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

    return thresh_img

def preprocess_image(file_path, image_size):

    img = cv2.imread(file_path).astype(np.uint8)
    img_enhanced = enhance_image(img)
    img_resized = cv2.resize(img_enhanced,  image_size[:2])
    img_normalized = img_resized / 255.0
    thresh_img = apply_threshold(img_resized)
    return img_normalized, thresh_img


def preprocess_labels(labels, image_size):
    masks = np.zeros((len(labels), image_size[0], image_size[1], 1), dtype=np.uint8)
    for i, label in enumerate(labels):
        masks[i] = np.expand_dims(np.where(label == 1, 1, 0), axis=-1)
    return masks
