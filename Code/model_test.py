import torch
import csv
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple

def extract_donnees():
		    
	path_csv = 'data/coco128/test/name_images.csv'
	data = []
	reader = csv.DictReader(open(path_csv))
	
	with open(path_csv, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append(row['filename'])
		
	return data

# imgs : création de la liste des images
path_image = 'data/coco128/test/images/'
imgs = []
data = []
data = extract_donnees()
print('Test images :', len(data))

for i in range(len(data)):
	img1=path_image + data[i] + '.png' # PIL image
	imgs.append(img1)

# bounding_box_real : création de la liste des bouding box
bounding_box_real = []
path_csv = 'data/coco128/test/images/test_data.csv'
reader = csv.DictReader(open(path_csv))
	
with open(path_csv, newline='') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		bounding_box_real.append([int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax'])])

# bounding_box_predict : création de la liste des bouding box
bounding_box_predict = []
path_csv = 'results_model.csv'
reader = csv.DictReader(open(path_csv))
	
with open(path_csv, newline='') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		bounding_box_predict.append([int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax'])])

def bb_intersection_over_union(boxA, boxB):

	'''
	boxA/B= [xmin, ymin, xmax, ymax]
	'''
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
	iou = interArea / int(boxAArea + boxBArea - interArea)
	
	return abs(iou)


def test_sample():

	predic_ma = []
	L = []
	for i in range(len(imgs)):
	
		print(i, len(imgs))
		path_model = 'Yolo/runs/train/'+'test_henri'+'/weights/best.pt'
		path_to_yolo = 'Yolo/'
		
		model = torch.hub.load(path_to_yolo, 'custom', path=path_model, source='local')  # local repo

		# Inference
		results = model(imgs[i], size=512)  # includes NMS

		# Results
		#results.print()  
		#results.save()  # .save() or .show()
		
		try :
			L.append([int(results.pandas().xyxy[0]['xmin']),
			int(results.pandas().xyxy[0]['ymin']),
			int(results.pandas().xyxy[0]['xmax']),
			int(results.pandas().xyxy[0]['ymax'])])
		except:
			L.append([0,0,0,0])

	df = pd.DataFrame({
	 "xmin": [L[i][0] for i in range(len(L))],
	 "ymin": [L[i][1] for i in range(len(L))],
	 "xmax" : [L[i][2] for i in range(len(L))],
	 "ymax" : [L[i][3] for i in range(len(L))]})
	
	df.to_csv('results_model.csv')


def test_iou():
	iou_list = []
	Detection = namedtuple("Detection", ["image_path", "gt", "pred"])
	examples = []
	predict = [] #[xmin, ymin, xmax, ymax]
	
	for i in range(len(data)):
		examples.append(Detection(imgs[i], bounding_box_real[i], bounding_box_predict[i]))
		
	# loop over the example detections*
	cpt = 0
	for detection in examples:
		cpt=cpt+1
		# load the image
		image = cv2.imread(detection.image_path)
		# draw the ground-truth bounding box along with the predicted
		# bounding box
		cv2.rectangle(image, tuple(detection.gt[:2]), tuple(detection.gt[2:]), (0, 255, 0), 2)
		cv2.rectangle(image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2)
		# compute the intersection over union and display it
		iou = bb_intersection_over_union(detection.gt, detection.pred)
		iou_list.append(iou)
		cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		#print("{}: {:.4f}".format(detection.image_path, iou))
		# show the output image
		#cv2.imshow("Image", image)
		#cv2.imwrite("runs/images/"+str(cpt)+".jpg", image)
		#cv2.waitKey(0)

	#print(sum(iou_list)/len(iou_list)) iou average
	return (iou_list)

def indicators(iou_list):
	detected = 0
	porcentage = []
	for iou in iou_list:
		
		if iou!=0:
			detected += 1
			porcentage.append(iou)

	fig = plt.figure()

	pie = [detected/len(iou_list),1-detected/len(iou_list)]
	plt.pie(pie, labels=['Detected','Not detected'], autopct = lambda pie: str(round(pie, 2)) + '%')
	plt.savefig("runs/analysis/pie_detected.png")
	plt.show()
	plt.hist(porcentage)
	plt.title('Repartition of detected iou')
	#plt.plot([i for i in range(len(porcentage))],porcentage)
	plt.show()

iou_list= test_iou()
indicators(iou_list)