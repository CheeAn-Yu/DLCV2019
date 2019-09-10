# TODO: create shell script for Problem 1
wget https://www.dropbox.com/s/rs8ou20vekobw3g/cnn0.436931.pth?dl=0 -O cnn0.436931.pth
python3 predict_cnn.py $1 $2 $3
