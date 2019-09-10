# TODO: create shell script for running your improved model
wget "https://www.dropbox.com/s/qdow9smhm72u7fg/yolo-4000.pth?dl=1" -O yolo-4000.pth
python3 yolo_best.py $1 $2
