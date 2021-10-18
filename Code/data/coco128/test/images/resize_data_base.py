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

data = extract_donnees()
for i in range(len(data)):
    dx = (data[i][2] - data[i][1]) / 2
    dy = (data[i][4] - data[i][3]) / 2
    x = data[i][1] + dx
    y = data[i][3] + dy
    with open('../labels/' + data[i][0]+'.txt','w') as f:
        f.write('0\t')
        f.write(str(x)+'\t')
        f.write(str(y)+'\t')
        f.write(str(dx)+'\t')
        f.write(str(dy)+'\t')

        f.close()
