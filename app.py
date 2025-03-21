import os
import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image

from remote_infer_rest import ort_v5

# Constants
MODEL_NAME = 'yolo'
REST_URL = 'http://modelmesh-serving.nerc-demo-5b7ce1:8008'
INFER_URL = f'{REST_URL}/v2/models/{MODEL_NAME}/infer'
IMAGES_FOLDER = './images'  # Folder for storing user-uploaded images
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6
INPUT_SIZE = 640
CLASSES_FILE = 'coco.yaml'

# Ensure the "images" folder exists
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

def draw_predictions(image_path, img, predictions):
    """Draw bounding boxes on the image and display it."""
    if predictions:
        for det in predictions:
            x1, y1, x2, y2, conf, class_idx = det
            label = f"Class {int(class_idx)}: {conf:.2f}"
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# Streamlit: File upload section
st.title('YOLO Object Detection App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image to the "images" folder
    image_path = os.path.join(IMAGES_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prepare and run the inference
    infer = ort_v5(image_path, INFER_URL, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, INPUT_SIZE, CLASSES_FILE)
    img, predictions, result = infer()

    if result:
        st.write(f"Inference Result: {result}")
        st.write('Predictions:')
        st.write(predictions)
        draw_predictions(image_path, img, predictions)
    else:
        st.error("Error: No predictions received.")
