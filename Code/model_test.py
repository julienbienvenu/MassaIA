import cv2
import torch
from PIL import Image


path_model = 'Yolo/runs/train/exp12/weights/best.pt'
path_to_yolo = 'Yolo/'
path_image = 'data/coco128/test/images/'
model = torch.hub.load(path_to_yolo, 'custom', path=path_model, source='local')  # local repo

# Images
img1 = Image.open(path_image + 'aerial_image_00009304D1.png')  # PIL image

imgs = [img1]  # batch of images

# Inference
results = model(imgs, size=512)  # includes NMS

# Results
results.print()  
results.show()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

