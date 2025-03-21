import re
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from remote_infer_rest import ort_v5

model_name = 'yolo'
rest_url = 'http://modelmesh-serving.nerc-demo-5b7ce1:8008'
infer_url = f'{rest_url}/v2/models/{model_name}/infer'
classes_file = './coco.yaml'

print(infer_url)

# 1. The image you want to analyze
#image_path='./images/car.jpg' # You can replace this with an image you upload
image_path='images/zidane.jpg' 

# 2. Confidence threshold, between 0 and 1 (detections with less score won't be retained)
conf = 0.4

# 3. Intersection over Union Threshold, between 0 and 1 (cleanup overlapping boxes)
iou = 0.6

infer=ort_v5(image_path, infer_url, conf, iou, 640, classes_file)
img, out, result = infer()
print(f'{result}')
print('Predictions:')
print(out)
print('Format: each detection is a float64 array shaped as [top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y, confidence, class_index]')
print('The coordinates are relative to a letterboxed representation of the image of size 640x640')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.gcf()
fig.set_size_inches(24, 12)
plt.axis('off')
plt.imshow(img)