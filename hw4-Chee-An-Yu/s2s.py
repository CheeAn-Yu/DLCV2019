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

batchSize = 1
train_feature_root = "/home/robot/hw4-Chee-An-Yu/seq2seq/train_feature_1"
val_feature_root = "/home/robot/hw4-Chee-An-Yu/seq2seq/val_feature_1"
train_csv_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/labels/train"
val_csv_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/labels/valid"

train_data = TrainSeqenceLoader(train_feature_root, train_csv_root, 512)
val_data = ValSeqenceLoader(val_feature_root, val_csv_root)
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True)
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
        out_seq = []
        # packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_length, batch_first=True)
        
        outputs, (hn, cn) = self.lstm(padded_sequence, hidden)
        # outputs(23,512,512)
        # print("outputs",outputs.size())
        # print("outputs1",outputs.size(1))
     
        category = self.fc2(self.dropout(self.fc1(outputs)))
        # print("category",category.size())
        
        return category

train_loss = []
validation_acc = {k:[] for k in range(len(val_data))}
average_acc = []
feature_size = 512*7*7
# feature_size = 2048
model = LSTM(feature_size,hidden_size=512, n_layers=2, dropout=0.5).cuda()
print(model)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# loss_function = nn.CrossEntropyLoss()
# max_accuracy = 0
# model.train()
# for epoch in tqdm(range(100)):
#     print("Epoch", epoch+1)
#     CE_loss = 0.0

#     for batch, (padded, label) in enumerate(tqdm(train_dataloader)):
#         # print("label",label)
#         optimizer.zero_grad()
#         # image = image.cuda()
#         # label = label.cuda()
#         # features = res50_conv(image).squeeze()
#         # features.cuda()
#         # print("feature",[padded.size(0)])
#         # print("label",label.size())
#         padded = padded.cuda()
#         output = model(padded,hidden=None)
#         # output = model(padded.unsqueeze(dim=0),hidden=None)
#         # print("output",output.size())
#         # print("max",torch.max(output,1)[1])
#         # print("label",label.cuda().size())
#         # print("output",output.size())
#         loss = loss_function(output.view(-1,11), label.view(-1))
#         loss.backward()
#         optimizer.step()
#         CE_loss += loss.cpu().data.numpy()
#     print("training loss",CE_loss/(batch+1))
#     train_loss.append(CE_loss/(batch+1))
#     # torch.save(model.state_dict(),'./cnn.pth')
#     with torch.no_grad():
#         print("evaulation!")
#         correct = 0
#         total = 0
#         model.eval()
#         val_loss = 0.0
#         for batch, (padded, label) in enumerate(val_dataloader):
#             batch_total = 0
#             batch_correct = 0
#             batch_acc = 0
#             # image = image.cuda()
#             # features = res50_conv(image).squeeze()
#             output = model(padded.cuda(), hidden=None)
#             # print("output",output.size())
#             output_label = torch.argmax(output,2).cpu().data
#             # print("output_label",output_label)
#             # print("label",label.squeeze(),label.size())
        
#             # _, predicted = torch.max(output.data, 1)

#             # print("predicted",predicted)
#             # print('label',label.size())
#             loss = loss_function(output.squeeze(), label.squeeze().cuda())
#             batch_total += label.size(1)
#             batch_correct += (output_label == label.squeeze().cpu()).sum().item()
#             batch_acc = batch_correct / batch_total
#             validation_acc[batch].append(batch_acc)


#             total += label.size(1)
#             correct += (output_label == label.squeeze().cpu()).sum().item()
            
#             val_loss += loss
#             # print("total",total)
#             # print("correct",correct)
#         accuracy = correct / total
#         print("validation loss:",val_loss.item() / (batch + 1)) 
#         print("accuracy:", accuracy)
#         average_acc.append(accuracy)
#         # if accuracy > max_accuracy:
#             # max_accuracy = accuracy
#             # print("save model"+str(accuracy))
#     torch.save(model.state_dict(),'./seq_model/seq%.3d.pth'%(epoch+1))
#     np.save("./validation.npy",validation_acc)
#     np.save("./average_acc.npy",average_acc)
#     print("validation",validation_acc)
#     print("average_acc ", average_acc)
#     # plt.figure(figsize=(15,5))
#     # plt.subplot(1,2,1)
#     # plt.plot(train_loss)
#     # plt.title("training loss")
#     # plt.ylabel("cross entropy")
#     # plt.xlabel("epoch")
#     # plt.subplot(1,2,2)
#     # plt.plot(validation_acc)
#     # plt.title("validation accuracy")
#     # plt.ylabel("accuracy")
#     # plt.xlabel("epoch")
#     # plt.savefig("res50_p2_curve12.png")
#         # plt.show()

#     # torch.save(model.state_dict(),'./rnn_model/rnn.pth')
#     model.train()