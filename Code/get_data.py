import csv  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random
from PIL import Image
import numpy as np
import math

def extract_donnees(): #ok
    
	path_csv = '../dataset_massalia/labels/csv/train_boxes.csv'
	data = []
	reader = csv.DictReader(open(path_csv))
	
	with open(path_csv, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append([row['filename'],float(row['xmin']),float(row['xmax']),float(row['ymin']),float(row['ymax'])])
		
	return data

def chgmt_de_base(img, rotation):
	rotation = rotation*np.pi/360
	xmin = img[1] - (512/2)
	ymin = img[3] - (512/2)
	xmax = img[2] - (512/2)
	ymax = img[4] - (512/2)

	#p1, p2, p3, p4 les 4 points
	p1 = [xmax, ymax]
	p2 = [xmax, ymin]
	p3 = [xmin, ymin]
	p4 = [xmin, ymax]

	rec = [p1,p2,p3,p4]
	rec_prime = rec

	for i in range(len(rec)):
		#x' = xcos() + ysin()
		#y' = -xsin() + ycos()
		rec_prime[i][0] = math.ceil(rec[i][0]*np.cos(rotation) + rec[i][1]*np.sin(rotation))
		rec_prime[i][1] = math.ceil(-rec[i][0]*np.sin(rotation) + rec[i][1]*np.cos(rotation))

	img[1] = (rec_prime[2][0] + (512/2)) 
	img[2] = (rec_prime[0][0] + (512/2)) 
	img[3] = (rec_prime[2][1] + (512/2)) 
	img[4] = (rec_prime[0][1] + (512/2)) 
	
	return img

data = extract_donnees()



def test_bis():
	plt.figure(figsize=(10, 10))
	for i in range(9):
		img = data[random.randint(0,100)]
		image = Image.open('../dataset_massalia/images/train_images/'+img[0])
		
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(image)
		
		plt.plot([img[1],img[1],img[2],img[2],img[1]],[img[3],img[4],img[4],img[3],img[3]],'r')
		plt.axis("off")
	plt.show()



def test():
	plt.figure(figsize=(10, 10))
	for i in range(9):
		image = image.rotate(90)
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(image)
		img = chgmt_de_base(img,90)
		plt.plot([img[1],img[1],img[2],img[2],img[1]],[img[3],img[4],img[4],img[3],img[3]],'r')
		plt.axis("off")
	plt.show()


img = data[random.randint(0,100)]
image = Image.open('../dataset_massalia/images/train_images/'+img[0])
plt.imshow(image)
imm = np.array(image)
print(type(imm))
plt.imshow(imm)
plt.plot()
plt.show()