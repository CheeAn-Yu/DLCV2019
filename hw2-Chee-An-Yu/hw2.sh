# TODO: create shell script for running your YoloV1-vgg16bn model
wget 'https://www.dropbox.com/s/n5e9tq6tec778v6/yolo-19500.pth?dl=1' -O yolo-19500.pth

python3.7 yolo_base.py $1 $2



