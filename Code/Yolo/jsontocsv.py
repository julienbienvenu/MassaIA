import pandas as pd
import json
import csv

path = 'runs/val/exp2/best_predictions.json'

f = open(path)
data = json.load(f)
f.close()
name = data[0]['image_id']
score = data[0]['score']

out = [['filename','xmin','ymin','xmax','ymax','score']]
list_append = [data[0]['image_id']+'.png',int(data[0]['bbox'][0]),int(data[0]['bbox'][1]),int(data[0]['bbox'][2]),int(data[0]['bbox'][3]),round(float(data[0]['score']),2)]
for val in data:
    if val['image_id']==name:
        if val['score']>score:
            list_append = [val['image_id']+'.png',int(val['bbox'][0]),int(val['bbox'][1]),int(val['bbox'][2]),int(val['bbox'][3]),round(float(val['score']),2)]
            score = val['score']
    else:        
        out.append(list_append)
        list_append = [val['image_id']+'.png',int(val['bbox'][0]),int(val['bbox'][1]),int(val['bbox'][2]),int(val['bbox'][3]),round(float(val['score']),2)]
        name = val['image_id']
        score = val['score']


print(len(out))

file = open('result.csv','w')
with file:    
    write = csv.writer(file)
    write.writerows(out)
file.close()
