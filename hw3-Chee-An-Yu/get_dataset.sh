# Download dataset from Dropbox
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gbnGEMyLIsYdIoyUyVZjYK8MzQZs4e_V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gbnGEMyLIsYdIoyUyVZjYK8MzQZs4e_V" -O hw3_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./hw3_data.zip -d hw3_data

# Remove the downloaded zip file
rm ./hw3_data.zip
