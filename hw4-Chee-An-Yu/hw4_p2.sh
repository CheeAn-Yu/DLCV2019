# TODO: create shell script for Problem 2
wget https://www.dropbox.com/s/c4a6u8mt6kpow4q/rnn0.469.pth?dl=0 -O rnn0.469.pth
python3 predict_rnn.py $1 $2 $3
