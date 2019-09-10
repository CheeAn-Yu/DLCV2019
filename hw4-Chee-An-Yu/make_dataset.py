from reader import readShortVideo
from reader import getVideoList
import os 
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import pickle

all_video_path = []
all_video_frames = []
video_root = "./hw4_data/TrimmedVideos/video/train"
val_video_root = "./hw4_data/TrimmedVideos/video/valid"
category_list = sorted(listdir(video_root))

for category in category_list:
    video_in_folder = sorted(listdir(os.path.join(video_root,category)))
    video_path = ['-'.join(file_name.split('-')[:5]) for file_name in video_in_folder]
    all_video_path += video_path
    for video in video_in_folder:
        frames = readShortVideo(video_root,category,video, downsample_factor=12, rescale_factor=1)
        frames = transforms.Compose([
        	transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        	transform.ToTensor(),
        	# transform.Resize(224)
        	])
        all_video_frames.append(frames)

train_X = []
for i in range(len(all_video_frames)):
    train_X.append(all_video_frames[i])
with open("train_X_d12.pkl", "wb") as f:
    pickle.dump(train_X, f)

all_video_path = []
all_video_frames = []
val_video_root = "./hw4_data/TrimmedVideos/video/valid"
category_list = sorted(listdir(val_video_root))


for category in category_list:
    val_video_in_folder = sorted(listdir(os.path.join(val_video_root,category)))
    val_video_path = ['-'.join(file_name.split('-')[:5]) for file_name in val_video_in_folder]
    all_video_path += val_video_path
    for video in val_video_in_folder:
        frames = readShortVideo(val_video_root,category,video, downsample_factor=12, rescale_factor=1)
        all_video_frames.append(torch.tensor(frames))

val_X = []
for i in range(len(all_video_frames)):
    val_X.append(all_video_frames[i])
with open("val_X_d12.pkl", "wb") as f:
    pickle.dump(val_X, f)

#### deal with label
train_df = pd.read_csv('./hw4_data/TrimmedVideos/label/gt_train.csv')
val_df = pd.read_csv('./hw4_data/TrimmedVideos/label/gt_valid.csv')

train_df = train_df.sort_values(['Video_name']).reset_index(drop=True)
video_name = train_df['Video_name'].tolist()
action_labels = train_df['Action_labels'].tolist()
with open('train_y.pkl', 'wb') as f:
    pickle.dump(action_labels,f)

val_df = val_df.sort_values(['Video_name']).reset_index(drop=True)
action_labels = val_df['Action_labels'].tolist()
with open('valid_y.pkl', 'wb') as f:
    pickle.dump(action_labels,f)