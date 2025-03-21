import os
import cv2
import base64
import requests
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image

# Constants
MODEL_NAME = 'yolo'
REST_URL = 'http://modelmesh-serving.nerc-demo-5b7ce1:8008'
INFER_URL = f'{REST_URL}/v2/models/{MODEL_NAME}/infer'
IMAGES_FOLDER = './images'  # Folder for storing user-uploaded images
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6
INPUT_SIZE = 640

# Ensure the "images" folder exists
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

def encode_image(image_path):
    """Encode image as base64 for API request."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def send_inference_request(image_path):
    """Send image to REST inference API and return the response."""
    headers = {"Content-Type": "application/json"}

    # Read the image as bytes
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    # Send the image as a hex-encoded binary in the request
    payload = {
        "inputs": [
            {
                "name": "images",  # Updated name as expected by the API
                "shape": [1],  # Batch size
                "datatype": "BYTES",
                "parameters": {"binary_data": True},  # Ensure binary data
                "data": [image_bytes.hex()]  # Convert binary to hex string
            }
        ],
        "parameters": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD
        }
    }

    response = requests.post(INFER_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def draw_predictions(image_path, predictions):
    """Draw bounding boxes on the image and display it."""
    img = cv2.imread(image_path)

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

    # Encode and send for inference
    response = send_inference_request(image_path)

    if response and "outputs" in response:
        results = response["outputs"][0]["data"]
        predictions = np.array(results).reshape(-1, 6)  # Reshape as needed
        print("Predictions:", predictions)
        draw_predictions(image_path, predictions)
    else:
        st.error("Error: No predictions received.")
