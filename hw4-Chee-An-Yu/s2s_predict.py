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


class ValSeqenceLoader(Dataset):
    def __init__(self, video_root, transform=None):
        """ Intialize the  dataset """
        self.video_root = video_root
        self.num_video = sorted(listdir(video_root))
        self.transform = transform
        # self.csv_root = csv_root
        # self.num_csv = sorted(listdir(csv_root))
#         self.df = pd.read_csv(csv_root,header=None,squeeze=True)
       
    
    def __getitem__(self, index):
        
        video = os.path.join(self.video_root, self.num_video[index])
        frames_list = sorted(listdir(video))
        # print(len(frames_list))

        # df = pd.read_csv(os.path.join(self.csv_root, self.num_csv[index]),header=None,squeeze=True)

        # df = torch.tensor(df.values.tolist())

        frames = []
        labels = []
        for img in frames_list:

            # image = np.load(os.path.join(video,img))
            # image = torch.from_numpy(image)
            image = io.imread(os.path.join(video,img))
            image = transform(image)
            image = image.unsqueeze(dim=0)
            image = vgg_feature(image.cuda()).view(-1)
            # print(image.size())
            frames.append(image)
        
            
        frames = torch.stack(frames).cuda()
        # frames = vgg_feature(frames)
        # print("frames",frames.size())    
        print("num_video",self.num_video[index])
        return self.num_video[index], frames
    def __len__(self):
        
        return len(self.num_video)


# val_feature_root = "/home/robot/hw4-Chee-An-Yu/seq2seq/val_feature_1"
# val_feature_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/videos/valid"
# val_csv_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/labels/valid"

val_feature_root = sys.argv[1]
val_data = ValSeqenceLoader(val_feature_root)
val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=1, shuffle=False)


class LSTM(nn.Module):
    def __init__(self, input_size=512*7*7, hidden_size=512, n_layers=2, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
            dropout=dropout, bidirectional=True, batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        # self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc2 = nn.Linear(int(self.hidden_size), 11)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence,  hidden=None):
        
        outputs, (hn, cn) = self.lstm(padded_sequence, hidden)
        # outputs(23,512,512)
        # print("outputs",outputs.size())
        # print("outputs1",outputs.size(1))
     
        category = self.fc2(self.dropout(self.fc1(outputs)))
        return category

def to_txt(save_root,output, category):
    print("save_root", os.path.join(save_root,category+'.txt'))
    with open(os.path.join(save_root,category+'.txt'), 'w') as f:
        for i in range(len(output)):
            if i == len(output)-1:
                f.write(str(output[i]))
            else:
                f.write(str(output[i])+'\n')

save_root = sys.argv[2]
num_video = sorted(listdir(val_feature_root))
feature_size = 512*7*7
model = LSTM(feature_size,hidden_size=512, n_layers=2, dropout=0.5).cuda()
model.load_state_dict(torch.load("./seq004.pth"))
model.eval()
validation_acc = {k:[] for k in num_video}
print("dic",validation_acc)

with torch.no_grad():
    print("evaulation!")
    correct = 0
    total = 0
    model.eval()
    val_loss = 0.0
    for batch, (category, padded) in enumerate(val_dataloader):
        print("category", category[0])
        batch_total = 0
        batch_correct = 0
        batch_acc = 0
        print("padded",padded.size())
        # padded = vgg_feature(padded.squeeze(dim=0))
        # image = image.cuda()
        # features = res50_conv(image).squeeze()
        output = model(padded.cuda(), hidden=None)
        # print("output",output.size())
        output_label = torch.argmax(output,2).cpu().data
        output_txt = output_label.squeeze().tolist()
        # print("output_label",output_txt)
        to_txt(save_root, output_txt,category[0])
        
        # print("label",label.squeeze(),label.size())
    
        # _, predicted = torch.max(output.data, 1)

        # print("predicted",predicted)
        # print('label',label.size())
        # batch_total += label.size(1)
        # batch_correct += (output_label == label.squeeze().cpu()).sum().item()
        # batch_acc = batch_correct / batch_total
        # validation_acc[category[0]].append(batch_acc)
# print(validation_acc)