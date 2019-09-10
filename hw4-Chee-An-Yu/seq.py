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

batchSize = 64
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

train_loss = []
validation_acc = {k:[] for k in val_num_video}

feature_size = 512*7*7
# feature_size = 2048
model = LSTM(feature_size,hidden_size=512, n_layers=2, dropout=0.5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()
max_accuracy = 0
model.train()
for epoch in tqdm(range(100)):
    print("Epoch", epoch+1)
    CE_loss = 0.0
    for video, csv in train_video_csv:
        # print("'video",video)
        # print("csv",csv)
        train_dataset = SeqFeatureLoader(os.path.join(train_feature_root,video),os.path.join(train_csv_root,csv))
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
        for batch, (padded, length, label) in enumerate(tqdm(train_dataloader)):
            # print("label",label)
            optimizer.zero_grad()
            # image = image.cuda()
            # label = label.cuda()
            # features = res50_conv(image).squeeze()
            # features.cuda()
            # print("feature",features.size())
            # print("label",label.size())
            padded = padded.cuda()
            output = model(padded, length)
            # print("output",output.size())
            # print("max",torch.max(output,1)[1])
            # print("label",label.cuda().size())
            # print("output",output.size())
            loss = loss_function(output, label.cuda())
            loss.backward()
            optimizer.step()
            CE_loss += loss.cpu().data.numpy()
        print("training loss",CE_loss/(batch+1))
        train_loss.append(CE_loss/(batch+1))
    # torch.save(model.state_dict(),'./cnn.pth')
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        val_loss = 0.0
        for video, csv in val_video_csv:

            val_dataset = SeqFeatureLoader(os.path.join(val_feature_root,video),os.path.join(val_csv_root,csv))
            val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False,collate_fn=collate_fn)
            for batch, (padded, length, label) in enumerate(tqdm(val_dataloader)):

                # image = image.cuda()
                # features = res50_conv(image).squeeze()
                output = model(padded.cuda(), length)
                output_label = torch.argmax(output,1).cpu().data
                # print("output_label",output_label)
                # print("label",label.squeeze(),label.size())
            # accuracy = np.mean((output_label == label).numpy())
                # _, predicted = torch.max(output.data, 1)

                # print("predicted",predicted)
                # print('label',label.size())
                loss = loss_function(output, label.cuda())

                total += label.size(0)
                correct += (output_label == label.squeeze()).sum().item()
                
                val_loss += loss
                # print("total",total)
                # print("correct",correct)
            accuracy = correct / total
            print("validation loss:",val_loss.item() / (batch + 1)) 
            print("accuracy:" + video, accuracy)
            validation_acc[video].append(accuracy)
        # if accuracy > max_accuracy:
            # max_accuracy = accuracy
            # print("save model"+str(accuracy))
        torch.save(model.state_dict(),'./seq_model/seq%.3f.pth'%(accuracy))
    print("validation",validation_acc)
    # plt.figure(figsize=(15,5))
    # plt.subplot(1,2,1)
    # plt.plot(train_loss)
    # plt.title("training loss")
    # plt.ylabel("cross entropy")
    # plt.xlabel("epoch")
    # plt.subplot(1,2,2)
    # plt.plot(validation_acc)
    # plt.title("validation accuracy")
    # plt.ylabel("accuracy")
    # plt.xlabel("epoch")
    # plt.savefig("res50_p2_curve12.png")
        # plt.show()

    # torch.save(model.state_dict(),'./rnn_model/rnn.pth')
    model.train()