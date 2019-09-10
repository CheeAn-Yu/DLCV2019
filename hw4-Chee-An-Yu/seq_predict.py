from reader import readShortVideo
from reader import getVideoList
import matplotlib.pyplot as plt
from os import listdir
import os
import pandas as pd
import numpy as np
import pickle

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

batchSize = 256
train_feature_root = "/home/robot/hw4-Chee-An-Yu/seq2seq/train_feature"
val_feature_root = "/home/robot/hw4-Chee-An-Yu/seq2seq/val_feature"
train_csv_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/labels/train"
val_csv_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/labels/valid"
num_video = sorted(listdir(train_feature_root))
num_csv = sorted(listdir(train_csv_root))
train_video_csv = list(zip(num_video,num_csv))
val_num_video = sorted(listdir(val_feature_root))
val_num_csv = sorted(listdir(val_csv_root))
val_video_csv = list(zip(val_num_video, val_num_csv))

# print("num_video",num_video)
# print("train_video_csv",train_video_csv)




# for tensor, length, label in val_dataloader:
#   print(tensor.size())
#   print(length.size())
#   print(label.size())

# class LSTM(nn.Module):
#     def __init__(self, input_size=512*7*7, hidden_size=512, n_layers=2, dropout=0.5):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
#             dropout=dropout, bidirectional=False, batch_first=True)
#         self.bn_0 = nn.BatchNorm1d(self.hidden_size)
#         self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
#         # self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
#         self.fc2 = nn.Linear(int(self.hidden_size), 11)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#     def forward(self, padded_sequence,  hidden=None):
#         out_seq = []
#         # packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_length, batch_first=True)
        
#         outputs, (hn, cn) = self.lstm(padded_sequence, hidden)
#         outputs = outputs.squeeze()
#         # print("outputs",outputs.size())
#         # print("outputs1",outputs.size(1))
#         for idx in range(outputs.size(0)):
#             # print("in",self.fc2(outputs[idx]).size())
#             category = F.softmax(self.fc2(self.dropout(self.fc1(self.relu(outputs[idx])))),0)
#             out_seq.append(category)
#         category = torch.stack(out_seq)
#         return outputs

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

    def forward(self, padded_sequence, input_length, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_length, batch_first=True)
        
        outputs, (hn, cn) = self.lstm(packed, hidden)
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


batchSize = 64
feature_size = 512*7*7
model = LSTM(feature_size,hidden_size=512, n_layers=2, dropout=0.5).cuda()
model.load_state_dict(torch.load("/home/robot/hw4-Chee-An-Yu/seq_model/seq0.575.pth"))
model.eval()
# model.eval()
pred = {k:[] for k in val_num_video}
with torch.no_grad():
    print("evaulation!")

    for video, csv in val_video_csv:

        # val_dataset = SeqFeatureLoader(os.path.join(val_feature_root,video),os.path.join(val_csv_root,csv))
        # val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False)
        val_dataset = SeqFeatureLoader(os.path.join(val_feature_root,video),os.path.join(val_csv_root,csv))
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False,collate_fn=collate_fn)
        for batch, (padded,length, label) in enumerate(val_dataloader):
            # print("padded",padded.size())
            # image = image.cuda()
            # features = res50_conv(image).squeeze()
            # output = model(padded.unsqueeze(dim=0).cuda(), hidden=None)
            output = model(padded.cuda(), length)
            # print("output",output)
            output_label = torch.argmax(output,1).cpu().data
            pred[video] += output_label.tolist()
            # print("output_label",output_label)
            # print("label",label.squeeze(),label.size())
        # accuracy = np.mean((output_label == label).numpy())
            # _, predicted = torch.max(output.data, 1)

            # print("predicted",predicted)
            # print('label',label.size())
            # loss = loss_function(output, label.cuda())

            # total += label.size(0)
            # correct += (output_label == label.squeeze()).sum().item()
            
            # val_loss += loss
            # print("total",total)
            # print("correct",correct)
        # accuracy = correct / total
        # print("validation loss:",val_loss.item() / (batch + 1)) 
        # print("accuracy:" + video, accuracy)
        # validation_acc[video].append(accuracy)
print(pred)