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



class VideoLoader(Dataset):
    def __init__(self, video_root, csv_root, transform=None):
        """ Intialize the  dataset """
        global all_video_path
        all_video_path = []
        self.all_video_frames = []
        self.csv_root = csv_root
        self.video_root = video_root
        self.category_list = sorted(listdir(video_root))
        for category in self.category_list:
            video_in_folder = sorted(listdir(os.path.join(video_root,category)))
            video_path = ['-'.join(file_name.split('-')[:5]) for file_name in video_in_folder]
            all_video_path += video_path
            for video in video_in_folder:
                self.all_video_frames.append([category,video]) 
            
        df = pd.read_csv(self.csv_root)
        global arg
        # print(df['Video_name'].tolist())
        order = df['Video_name'].tolist()
        arg = np.argsort(order)
        # order = df['Action_labels'].tolist()
        # print(arg)
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
        frames = frames.view(frames.size(0),-1)
        print("before",frames.size())
        # print("after",frames.view(frames.size(0),-1).size())
        # frames = torch.mean(frames,0)
        # frames = frames.view(-1)
        # print(frames.size())
        # output = torch.mean(frames,0)
#         if self.transform:
#             frames = self.transform(frames)
        # return output, self.label[index]
        return frames

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(all_video_path)

batchSize = 1
# val_video_root = "./hw4_data/TrimmedVideos/video/valid"
# csv_root = './hw4_data/TrimmedVideos/label/gt_valid.csv'
val_video_root = sys.argv[1]
csv_root = sys.argv[2]

val_dataset = VideoLoader(val_video_root, csv_root,transform=None)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False)





# val_feature_root = "./rnn_vgg2_val_feature"

# val_csv = "./val_label"

# val_video = FeatureLoader(val_feature_root,val_csv)
# # image, label = val_video[0]


# val_dataloader = torch.utils.data.DataLoader(dataset=val_video, batch_size=batchSize,shuffle=False,collate_fn=collate_fn)

feature_size = 512*7*7
class LSTM(nn.Module):
    def __init__(self, input_size=512*7*7, hidden_size=512, n_layers=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
            dropout=dropout, bidirectional=False, batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence, hidden=None):
        # packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_length, batch_first=True)
        # print("packed",padded_sequence.size())
        outputs, (hn, cn) = self.lstm(padded_sequence, hidden)
        # outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print("outputs",outputs.size())
        # print("hn",hn.size())
        # print("cn",cn.size())
        hidden_output = hn[-1]
        # hidden_output = torch.mean(outputs,1)
        outputs = self.fc_1(hidden_output)
        outputs = self.bn_0(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        
        # print(hidden_output.size())
        # outputs = self.bn_0(hidden_output)
        # outputs = self.relu(hidden_output)
        outputs = self.fc_2(outputs)
        return outputs

def to_txt(save_root,output, category):
    print("save_root", os.path.join(save_root,category+'.txt'))
    with open(os.path.join(save_root,category+'.txt'), 'w') as f:
        for i in range(len(output)):
            if i == len(output)-1:
                f.write(str(output[i]))
            else:
                f.write(str(output[i])+'\n')

validation_acc = []
model = LSTM(feature_size,hidden_size=512, n_layers=2, dropout=0.5).cuda()
model.load_state_dict(torch.load("./rnn0.469.pth"))
result = []

correct = 0
total = 0
model.eval()
val_loss = 0.0



save_root = sys.argv[3]
with torch.no_grad():
    for batch, (padded) in enumerate(tqdm(val_dataloader)):
        print("padded",padded.size())
    # image = image.cuda()
    # features = res50_conv(image).squeeze()
        output = model(padded.cuda())
        output_label = torch.argmax(output,1).cpu().data
        print("output_label",output_label)
        # print("label",label.squeeze(),label.size())
        result += output_label.tolist()
        # _, predicted = torch.max(output.data, 1)
        print("result",result)
        # print("predicted",predicted)
        # print('label',label.size())


        # total += label.size(0)
        # correct += (output_label == label.squeeze()).sum().item()
        

        # print("total",total)
        # print("correct",correct)
        # accuracy = correct / total
        # print("accuracy", accuracy)

print("not_sorted", result)
pre = np.array(result)
zipped = list(zip(pre,arg))
zipped.sort(key = lambda t:t[1])
for i in range(len(zipped)):
    result[i] = zipped[i][0]
to_txt(save_root, result,"p2_result")

print("result",result)

