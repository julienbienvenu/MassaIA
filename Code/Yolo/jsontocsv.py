import pandas as pd
import json
import csv

path = 'runs/val/exp2/best_predictions.json'

f = open(path)
data = json.load(f)
f.close()
print(len(data))
print(data[0]['score'])

out = [['filename','xmin','ymin','xmax','ymax','score']]
for val in data:
    if (val['score']) > 0.2:
        out.append([val['image_id']+'.png',int(val['bbox'][0]),int(val['bbox'][1]),int(val['bbox'][2]),int(val['bbox'][3]),val['score']])

print(len(out))

file = open('result.csv','w')
with file:    
    write = csv.writer(file)
    write.writerows(out)

file.close()