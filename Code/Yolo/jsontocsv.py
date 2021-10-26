import pandas as pd
df = pd.read_json ('runs/val/exp2/best_predictions.json')
df.to_csv ('result.csv', index = ['filename','xmin','ymin','xmax','ymax','score'])
