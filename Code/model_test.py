import torch
from PIL import Image
import csv


path_model = 'Yolo/runs/train/exp13/weights/best.pt'
path_to_yolo = 'Yolo/'
path_image = 'data/coco128/test/images/'


model = torch.hub.load(path_to_yolo, 'custom', path=path_model, source='local')  # local repo

def extract_donnees():
    
	path_csv = 'data/coco128/test/name_images.csv'
	data = []
	reader = csv.DictReader(open(path_csv))
	
	with open(path_csv, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append(row['filename'])
		
	return data

# Images
imgs = []
data = extract_donnees()
print('Test images :', len(data))

for i in range(20):
    img1=Image.open(path_image + data[i] + '.png')  # PIL image
    imgs.append(img1)

# Inference
results = model(imgs, size=512)  # includes NMS

# Results
results.print()  
#results.save()  # .save() or .show()

print(results.xyxy[0])  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)

