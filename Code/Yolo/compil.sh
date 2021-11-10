pip install -r requirements.txt
python3 train.py --img 512 --batch 6 --epochs 250 --data ../data/coco128.yaml --weights best.pt
python3 detect.py --source ../data/coco128/submit --weights runs/train/exp3/weights/best.pt --img 512 --save-txt --save-conf --nosave