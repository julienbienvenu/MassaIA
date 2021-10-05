import csv  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random

def extract_donnees(): #ok
    
	path_csv = '../dataset_massalia/labels/csv/train_boxes.csv'
	data = []
	reader = csv.DictReader(open(path_csv))
	
	with open(path_csv, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append([row['filename'],float(row['xmin']),float(row['xmax']),float(row['ymin']),float(row['ymax'])])
		
	return data

data = extract_donnees()

'''
img = data[random.randint(0,100)]
plt.imshow(mpimg.imread('../dataset_massalia/images/train_images/'+img[0]))
plt.plot([img[1],img[1],img[2],img[2],img[1]],[img[3],img[4],img[4],img[3],img[3]],'r')
plt.show()
'''

import tensorflow as tf
from tensorflow.keras import layers

def data_augmentation(image, rotation):
	return tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])




#plt.figure(figsize=(10, 10))
img = data[random.randint(0,100)]
image = mpimg.imread('../dataset_massalia/images/train_images/'+img[0])
plt.imshow(image)
plt.plot([img[1],img[1],img[2],img[2],img[1]],[img[3],img[4],img[4],img[3],img[3]],'r')
plt.show()
for i in range(9):
	
	rotation = random.randint(0,360)
	augmented_image = data_augmentation(image, rotation)
	#ax = plt.subplot(3, 3, i + 1)
	plt.imshow(augmented_image[0])
	plt.axis("off")

	plt.show()
