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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

class VideoLoader(Dataset):
    def __init__(self, video_root, csv_root, transform=None):
        """ Intialize the  dataset """
        self.all_video_path = []
        self.all_video_frames = []
        self.csv_root = csv_root
        self.video_root = video_root
        self.category_list = sorted(listdir(video_root))
        for category in self.category_list:
            video_in_folder = sorted(listdir(os.path.join(video_root,category)))
            video_path = ['-'.join(file_name.split('-')[:5]) for file_name in video_in_folder]
            self.all_video_path += video_path
            for video in video_in_folder:
                self.all_video_frames.append([category,video]) 
            
        df = pd.read_csv(self.csv_root)
        self.df = df.sort_values(['Video_name']).reset_index(drop=True)
        self.label = self.df['Action_labels'].tolist()
        self.transform = transform
#         self.file = sorted(listdir(root_dir))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        frames = readShortVideo(self.video_root, self.all_video_frames[index][0],self.all_video_frames[index][1],
                                downsample_factor=12, rescale_factor=1)
        
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # label = self.landmarks_frame['label'][index]
        # label = torch.FloatTensor([label])
        frames = torch.stack(frames)
        # print(frames.size())
        # output = torch.mean(frames,0)
#         if self.transform:
#             frames = self.transform(frames)
        # return output, self.label[index]
        return frames, self.label[index]
    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.all_video_path)

val_video_root = "./hw4_data/TrimmedVideos/video/valid"
video_root = "./hw4_data/TrimmedVideos/video/train"
train_csv = './hw4_data/TrimmedVideos/label/gt_train.csv'
csv_root = './hw4_data/TrimmedVideos/label/gt_valid.csv'
val_dataset = VideoLoader(val_video_root, csv_root,transform=None)
train_dataset = VideoLoader(video_root, train_csv,transform=None)
image,label = val_dataset[0]


import torch.nn as nn
cnn_feature = torchvision.models.resnet50(pretrained=True)
res50_conv = nn.Sequential(*list(cnn_feature.children())[:-1])
res50_conv.cuda()

vgg_feature = torchvision.models.vgg16(pretrained=True)
vgg_feature = vgg_feature.features.cuda()


batchSize = 1
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False)

# for batch,(image, label) in enumerate(tqdm(val_dataloader)):
#     img =[]
#     image = image.cuda()
#     output = vgg_feature(image.squeeze()).view(-1,512*7*7)
#     print(output.size())

print("make validation feature")
for batch,(image, label) in enumerate(tqdm(val_dataloader)):
    with torch.no_grad():
        image = image.cuda()

        output = vgg_feature(image.squeeze(dim=0)).view(-1,512*7*7)
        #output = res50_conv(image.squeeze(dim=0)).view(-1,2048)
        cnn_output = torch.mean(output,0)
        # print("output",output.size())
        # print("cnn_output",cnn_output.size())
        # output = vgg_feature(image.squeeze(dim=0))
        # print("feature_size",output.size())
    #     img.append(output)
    #     img = np.array(img)
        np.save('./cnn_vgg2_val_feature/'+str(batch).zfill(3),cnn_output.detach().cpu())
        np.save('./rnn_vgg2_val_feature/'+str(batch).zfill(3),output.detach().cpu())
    # np.save('./val_label/'+str(batch).zfill(3),label.detach().cpu())
    # print(output.size())
    # print(label.size())

print("make training feature")
for batch,(image, _) in enumerate(tqdm(train_dataloader)):
    with torch.no_grad():
        
        image = image.cuda()
        output = vgg_feature(image.squeeze(dim=0)).view(-1, 512*7*7)
        #output = res50_conv(image.squeeze(dim=0)).view(-1,2048)
        cnn_output = torch.mean(output,0)
    #     img.append(output)
    #     img = np.array(img)
        np.save('./cnn_vgg2_train_feature/'+str(batch).zfill(4),cnn_output.detach().cpu())
        np.save('./rnn_vgg2_train_feature/'+str(batch).zfill(4),output.detach().cpu())
        # torch.cuda.empty_cache()
    # label = label.detach().cpu()
    # np.save('./train_label/'+str(batch).zfill(4),label.detach().cpu())
    # print(output.size())