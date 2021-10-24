import csv

def extract_donnees():
    
	path_csv = "test_data.csv"
	data = []
	reader = csv.DictReader(open(path_csv))
	
	with open(path_csv, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append([row['filename'][:-4],float(row['xmin'])/512,float(row['xmax'])/512,float(row['ymin'])/512,float(row['ymax'])/512])
		
	return data

      
def save_list_name():
	data = extract_donnees()
	with open('../name_images.csv','w') as f:
		for i in range(len(data)):
		    name = data[i][0]
		    f.write(str(name)+'\n')

	f.close()
	
save_list_name()
