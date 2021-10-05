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


img = data[random.randint(0,100)]
plt.imshow(mpimg.imread('../dataset_massalia/images/train_images/'+img[0]))
plt.plot([img[1],img[1],img[2],img[2],img[1]],[img[3],img[4],img[4],img[3],img[3]],'r')
plt.show()