#!/usr/bin/env python
# coding: utf-8

# In[10]:

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
# from dataloader_v4 import Load_Data
from dataloader_v5 import Load_Data
# from models import Yolov1_vgg16bn
from improve_model import Yolov1_vgg16bn
# from loss import Loss
from loss_improve import Loss
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# In[11]:




def train(model, epoch, log_interval=100):
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = Loss(7,2,5,0.5)
    model.train()
    
    iteration = 0
    for ep in range(epoch):
        model.train()

        if ep == 30:
            learning_rate = 0.0001
        if ep == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate



        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to("cuda"), target.to("cuda")
            # print("data: ", data.type(), "target: ", target.type())
            data.float()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                logging.info('training!')
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
            
        test(model)
            


# In[12]:


def test(model):
    criterion = Loss(14,2,5,0.5)
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(testset_loader) #         
    # test_loss /= testset_loader.batch_size
    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss))


# In[13]:


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def train_save(model, epoch, save_interval, log_interval=100):
    learning_rate = 0.001
    device = "cuda"
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = Loss(14,2,5,0.5)
    model.train()  # set training mode
    
    iteration = 0
    for ep in range(epoch):
        model.train()

        if ep == 30:
            learning_rate = 0.0001
        if ep == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate



        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            output = output.float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                logging.info('training!')
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('improve/yolo-%i.pth' % iteration, model, optimizer)
            iteration += 1
        test(model)
    
    # save the final model
    save_checkpoint('yolo-%i.pth' % iteration, model, optimizer)


# In[14]:





# In[15]:


train_fname = [str(i).zfill(5)+".txt" for i in range(15000)]
train_iname = [str(i).zfill(5)+".jpg" for i in range(15000)]
test_fname = [str(i).zfill(4)+".txt" for i in range(1500)]
test_iname = [str(i).zfill(4)+".jpg" for i in range(1500)]

train_label = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/train15000/labelTxt_hbb/"
train_img = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/train15000/images/"
test_img = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/val1500/images/"
test_label = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/val1500/labelTxt_hbb/"

train_data = Load_Data(train_label,train_fname, train_img, train_iname, train=True)
test_data = Load_Data(test_label, test_fname, test_img, test_iname,train=False)


# In[16]:


trainset_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
testset_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=1)


# In[17]:

model = Yolov1_vgg16bn(pretrained=True)
model.cuda()
# checkpoint_path="/home/robot/hw2-Chee-An-Yu/pass0.092/yolo-19500.pth"
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# load_checkpoint(checkpoint_path, model, optimizer)
train_save(model, 60, 500, 100)


# In[ ]:





# In[ ]:




