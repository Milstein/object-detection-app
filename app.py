# app.py
import streamlit as st
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from remote_infer_grpc import ort_v5

# Configuration
model_name = 'yolo'
grpc_url = 'grpc://modelmesh-serving.nerc-demo-5b7ce1:8033'  # Update with actual gRPC URL
classes_file = 'coco.yaml'
conf = 0.4  # Confidence threshold
iou = 0.6  # IoU threshold

# Extract host and port
pattern1 = r"\\/\\/(.+?):\\d+"
match1 = re.search(pattern1, grpc_url)
grpc_host = match1.group(1) if match1 else ''

pattern2 = r"(\\d+)$"
match2 = re.search(pattern2, grpc_url)
grpc_port = match2.group(1) if match2 else ''

# Initialize inference model
infer = ort_v5(grpc_host, grpc_port, model_name, 640, classes_file)

st.title("YOLO Object Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    img, out, result = infer(image_path, conf, iou)
    
    st.write("Predictions:")
    st.write(result)
    st.write("Each detection is a float64 array shaped as [x1, y1, x2, y2, confidence, class_index]")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    ax.axis("off")
    st.pyplot(fig)
