pip install -r requirements.txt
python3 train.py --img 512 --batch 6 --epochs 250 --data ../data/coco128.yaml --weights best.pt
python3 val.py --data ../data/coco128.yaml --weights runs/train/exp/weights/best.pt --img 512 --save-json
python3 jsontocsv.py
