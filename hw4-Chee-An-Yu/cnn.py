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
import torch.nn as nn
from tqdm import tqdm
# val_video_root = "./hw4_data/TrimmedVideos/video/valid"
# video_root = "./hw4_data/TrimmedVideos/video/train"
# train_csv = './hw4_data/TrimmedVideos/label/gt_train.csv'
# csv_root = './hw4_data/TrimmedVideos/label/gt_valid.csv'
batchSize = 64

# cnn_feature = torchvision.models.resnet50(pretrained=True)
# res50_conv = nn.Sequential(*list(cnn_feature.children())[:-1]).cuda()

# val_dataset = VideoLoader(val_video_root, csv_root,transform=None)
# train_dataset = VideoLoader(video_root, train_csv,transform=None)
# train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False)

# for batch,(image, label) in enumerate(val_dataloader):
#     with torch.no_grad():
#       image = image.cuda()
#       output = res50_conv(image).view(batchSize,-1)
#       print(output.size())
#       print(label.size())

train_feature_root = "./cnn_vgg2_train_feature"
val_feature_root = "./cnn_vgg2_val_feature"
train_csv = "./train_label"
val_csv = "./val_label"
train_dataset = FeatureLoader(train_feature_root, train_csv)
val_dataset = FeatureLoader(val_feature_root,val_csv)
img,label = train_dataset[0]
print(img.size())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batchSize,shuffle=False)


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



train_loss = []
validation_acc = []
model = Net(feature_size).cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()
max_accuracy = 0
model.train()
for epoch in tqdm(range(100)):
    print("Epoch", epoch+1)
    CE_loss = 0.0
    for batch, (features, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        # image = image.cuda()
        # label = label.cuda()
        # features = res50_conv(image).squeeze()
        # features.cuda()
        # print("feature",features.size())
        # print("label",label.size())
        output = model(features.cuda())
        # print("output",output.size())
        # print("max",torch.max(output,1)[1])
        loss = loss_function(output, label.cuda().squeeze())
        loss.backward()
        optimizer.step()
        CE_loss += loss.cpu().data.numpy()
    print("training loss",CE_loss/(batch+1))
    train_loss.append(CE_loss)
    # torch.save(model.state_dict(),'./cnn.pth')
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        val_loss = 0.0
        for batch, (features, label) in enumerate(tqdm(val_dataloader)):

            # image = image.cuda()
            # features = res50_conv(image).squeeze()
            output = model(features.cuda())
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
        print("validation loss:",loss / (batch + 1)) 
        print("validation accuracy:", accuracy)
        validation_acc.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            print("save model"+str(accuracy))
            torch.save(model.state_dict(),'./cnn_model/cnn%3f.pth'%(accuracy))
    # torch.save(model.state_dict(),'./cnn_model/cnn.pth')

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss)
    plt.title("training loss")
    plt.ylabel("cross entropy")
    plt.xlabel("epoch")
    plt.subplot(1,2,2)
    plt.plot(validation_acc)
    plt.title("validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig("wvgg_p1_curve.png")
    model.train()
        
