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

batchSize = 2
train_feature_root = "./rnn_vgg2_train_feature"
val_feature_root = "./rnn_vgg2_val_feature"
train_csv = "./train_label"
val_csv = "./val_label"
train_dataset = FeatureLoader(train_feature_root,train_csv)
val_video = FeatureLoader(val_feature_root,val_csv)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(dataset=val_video, batch_size=batchSize,shuffle=False,collate_fn=collate_fn)

# for tensor, length, label in val_dataloader:
#   print(tensor.size())
#   print(length.size())
#   print(label.size())

class LSTM(nn.Module):
    def __init__(self, input_size=512*7*7, hidden_size=512, n_layers=1, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
            dropout=dropout, bidirectional=True, batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, padded_sequence, input_length, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_length, batch_first=True)
        print("packed",padded_sequence.size())
        outputs, (hn, cn) = self.lstm(packed, hidden)
        # outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print("outputs",outputs.size())
        print("hn",hn.size())
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
validation_acc = []

feature_size = 512*7*7
# feature_size = 2048
model = LSTM(feature_size,hidden_size=512, n_layers=2, dropout=0.5).cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()
max_accuracy = 0
model.train()
for epoch in tqdm(range(100)):
    print("Epoch", epoch+1)
    CE_loss = 0.0
    for batch, (padded, length, label) in enumerate(tqdm(train_dataloader)):
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
        loss = loss_function(output, label.cuda().squeeze(dim=1))
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
            loss = loss_function(output, label.cuda().squeeze(dim=1))

            total += label.size(0)
            correct += (output_label == label.squeeze()).sum().item()
            
            val_loss += loss
            # print("total",total)
            # print("correct",correct)
        accuracy = correct / total
        print("validation loss:",val_loss.item() / (batch + 1)) 
        print("validation accuracy:", accuracy)
        validation_acc.append(accuracy)
        if accuracy > max_accuracy and accuracy > 0.4:
            max_accuracy = accuracy
            print("save model"+str(accuracy))
            # torch.save(model.state_dict(),'./rnn_model/rnn%.3f.pth'%(accuracy))

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