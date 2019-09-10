from reader import readShortVideo
from reader import getVideoList
import matplotlib.pyplot as plt
from os import listdir
import os
import pandas as pd
import numpy as np
import pickle
from skimage import io, transform
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import VideoLoader
from dataset import FeatureLoader
from dataset import *
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import sys

vgg_feature = torchvision.models.vgg16(pretrained=True)
vgg_feature = vgg_feature.features.cuda()

def transform(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_output = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad((0,40), fill=0, padding_mode='constant'),
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
        ])
    return transform_output(image)


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
        global arg
        # print(df['Video_name'].tolist())
        order = df['Video_name'].tolist()
        arg = np.argsort(order)
        self.df = df.sort_values(['Video_name']).reset_index(drop=True)
        # self.label = self.df['Action_labels'].tolist()
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
        frames = vgg_feature(frames.cuda())
        # print(frames.size())
        frames = torch.mean(frames,0)
        frames = frames.view(-1)
        # print(frames.size())
        # output = torch.mean(frames,0)
#         if self.transform:
#             frames = self.transform(frames)
        # return output, self.label[index]
        return frames
    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.all_video_path)


# val_video_root = "./hw4_data/TrimmedVideos/video/valid"
# csv_root = './hw4_data/TrimmedVideos/label/gt_valid.csv'
val_video_root = sys.argv[1]
csv_root = sys.argv[2]
val_dataset = VideoLoader(val_video_root, csv_root,transform=None)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64,shuffle=False)



# batchSize = 64
# val_feature_root = "./cnn_vgg2_val_feature"

# val_csv = "./val_label"

# val_dataset = FeatureLoader(val_feature_root,val_csv)


# val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False)

feature_size = 512*7*7
class Net(nn.Module):
    def __init__(self, feature_size):
        super(Net, self).__init__()
        self.linear = nn.Linear(feature_size,4096)
        self.linear2 = nn.Linear(4096,1024)
        self.linear3 = nn.Linear(1024,11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.linear(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)
        # y_pred = self.softmax(x)
        return x

def to_txt(save_root,output, category):
    print("save_root", os.path.join(save_root,category+'.txt'))
    with open(os.path.join(save_root,category+'.txt'), 'w') as f:
        for i in range(len(output)):
            if i == len(output)-1:
                f.write(str(output[i]))
            else:
                f.write(str(output[i])+'\n')

validation_acc = []
model = Net(feature_size).cuda()
model.load_state_dict(torch.load("./cnn0.436931.pth"))
result = []
save_root = sys.argv[3]
with torch.no_grad():
    correct = 0
    total = 0
    model.eval()
    val_loss = 0.0
    for batch, (features) in enumerate(tqdm(val_dataloader)):
        # print("features",features.size())
        # image = image.cuda()
        # features = res50_conv(image).squeeze()
        output = model(features.cuda())
        output_label = torch.argmax(output,1).cpu().data
        # print("output_label",output_label.size())
        result += output_label.tolist()
        # print("label",label.squeeze(),label.size())
    # accuracy = np.mean((output_label == label).numpy())
        # _, predicted = torch.max(output.data, 1)

        # print("predicted",predicted)
        # print('label',label.size())
        
        # total += label.size(0)
        # correct += (output_label == label.squeeze()).sum().item()
        
       
        # print("total",total)
        # print("correct",correct)
    # accuracy = correct / total
    print("not_sorted", result)
    pre = np.array(result)
    zipped = list(zip(pre,arg))
    zipped.sort(key = lambda t:t[1])
    for i in range(len(zipped)):
        result[i] = zipped[i][0]
    to_txt(save_root, result,"p1_valid")

    print("result",result)


    # print("result",result)
    # to_txt(save_root, result,"p1_valid")
    # print("validation accuracy:", accuracy)
