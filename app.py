import cv2
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from remote_infer_rest import ort_v5
import os
from PIL import Image

# Parameters
model_name = 'yolo'
rest_url = 'http://modelmesh-serving.nerc-demo-5b7ce1:8008'
infer_url = f'{rest_url}/v2/models/{model_name}/infer'
classes_file = 'coco.yaml'
image_path = './images'

# Ensure the "images" folder exists
if not os.path.exists(image_path):
    os.makedirs(image_path)

# 1. Confidence threshold, between 0 and 1 (detections with less score won't be retained)
conf = 0.4

# 2. Intersection over Union Threshold, between 0 and 1 (cleanup overlapping boxes)
iou = 0.6

# Streamlit UI for image upload
st.title("YOLO Object Detection")
st.write("Upload an image to perform object detection using YOLO.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to the images directory
    uploaded_image_path = os.path.join(image_path, uploaded_file.name)
    with open(uploaded_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    infer = ort_v5(uploaded_image_path, infer_url, conf, iou, 640, classes_file)
    img, out, result = infer()

    # Check if the predictions tensor is empty
    if out.size(0) == 0:
        st.write("No objects detected in the image.")
    else:
        st.write(f'{result}')
        st.write('Predictions:')
        st.write(out)
        st.write('Format: each detection is a float64 array shaped as [top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y, confidence, class_index]')
        st.write('The coordinates are relative to a letterboxed representation of the image of size 640x640')

        # Process image and display it
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig = plt.gcf()
        fig.set_size_inches(24, 12)
        plt.axis('off')
        plt.imshow(img)
        st.pyplot(fig)

