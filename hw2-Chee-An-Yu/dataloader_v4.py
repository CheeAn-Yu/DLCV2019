#!/usr/bin/env python
# coding: utf-8

# In[125]:


# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

ide = np.identity(16)
_class = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court',
'ground-track-field', 'harbor', 'bridge', 'small-vehicle', 'large-vehicle', 'helicopter',  
'roundabout', 'soccer-ball-field', 'swimming-pool', 'container-crane']
onehot_encode = dict(zip(_class, ide))

train_fname = [str(i).zfill(5)+".txt" for i in range(15000)]
train_iname = [str(i).zfill(5)+".jpg" for i in range(15000)]
test_fname = [str(i).zfill(4)+".txt" for i in range(1500)]
test_iname = [str(i).zfill(4)+".jpg" for i in range(1500)]

train_label = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/train15000/labelTxt_hbb/"
train_img = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/train15000/images/"
test_img = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/val1500/images/"
test_label = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/val1500/labelTxt_hbb/"

class Load_Data(data.Dataset):
    
    def __init__(self, label_path, label_name, img_path, img_name, train):
        
        self.label_path = label_path
        self.img_path = img_path
        self.label_name = label_name
        self.img_name = img_name
        self.data = self.read_label(label_path, label_name)
        self.train = train
#         self.normal = self.nor_data_transform(self.data)
#         self.trs_data = self.data_transform(self.data)
#         self.normal = self.nor_data_transform(self.data) # label with cell information
#         self.img = self.read_img(img_path)
#         self.tensor_img = self.to_tensor(self.img) # input to model
#         self.data = self.read_label(label_path)



    
    def __len__(self):
        
        return len(self.label_name)

    def __getitem__(self, index):
        
        fname = self.img_name[index]
        lname = self.label_name[index]
        
#         data = self.read_label(self.label_path)
#         self.normal = self.nor_data_transform(self.data) fixed for iter !!! 

        img = cv2.imread(self.img_path+fname)
        img = cv2.resize(img,(448,448))
        # if self.train:
            # img = self.randomBlur(img)
            # img = self.RandomBrightness(img)
            # img = self.RandomHue(img)
            # img = self.RandomSaturation(img)

        self.tensor_img = self.to_tensor(img)
        label = self.read_one_label(self.label_path+lname)
        normal_label = self.one_nor_data_transform(label)
        self.encoder = self.encode(normal_label)
        

#         return self.tensor_img, torch.Tensor( self.normal[index] )
        return self.tensor_img, self.encoder


    def read_label(self, file_location, file_name):
        data = []
        for name in file_name:
    #     print(file_location+name)
            fp = open(file_location+name,"r")
            line = fp.readline()
            file = []

            while line:
                line = line.strip()
                file.append(line.split(" "))
                line = fp.readline()
            data.append(file)
    #     fp.close()
        return data
    
    def read_one_label(self, file_location):
        fp = open(file_location,"r")
        line = fp.readline()
        file = []

        while line:
            line = line.strip()
            file.append(line.split(" "))
            line = fp.readline()
        
        return file 




    def transform(self, data):
        trans_data = []
    #     print(float(data[2]))
    #     print(float(data[2]) + float(data[0]))

        x = (float(data[2]) + float(data[0])) / 2 * 0.875
    #     x = x % 64
        y = (float(data[5]) + float(data[1])) / 2 * 0.875
    #     y = y % 64
        w = (float(data[2]) - float(data[0])) * 0.875
    #     w = w / 448
        h = (float(data[5]) - float(data[1])) * 0.875
    #     h = h / 448
        trans_data.append(x)
        trans_data.append(y)
        trans_data.append(w)
        trans_data.append(h)
        trans_data.append(onehot_encode[data[8]])
        trans_data.append(int(data[9].strip()))
        return trans_data

    def normal(self, label):
        nor = label.copy()
        nor[0] = label[0] % 64 / 64

        
        nor[1] = label[1] % 64  / 64
        nor[2] = label[2] / 448
        nor[3] = label[3] / 448
        grid_x = label[0] // 64
        grid_y = label[1] // 64
#         nor.append((grid_x, grid_y))
        nor.append(grid_x)
        nor.append(grid_y)
        return nor

    def data_transform(self, data):
        trans_data = []
        for file in data:
            trans_file = []
            for item in file:
                trans_item = self.transform(item)
                trans_file.append(trans_item)
            trans_data.append(trans_file)
        return trans_data
    
   
            
    
    
    def nor_data_transform(self, data):
        trans_data = []
        for file in data:
            trans_file = []
            for item in file:
                trans_item = self.transform(item)
                print(trans_item)
                nor_item = self.normal(trans_item)
#                 nor_item = torch.Tensor(nor_item)
                trans_file.append(nor_item)
            trans_data.append(trans_file)
        return trans_data
        
    def one_nor_data_transform(self, data):
        trans_file = []
        for item in data:
            trans_item = self.transform(item)
            nor_item = self.normal(trans_item)
            trans_file.append(nor_item)
        return trans_file

    
    def to_tensor(self, img):
        img = torch.from_numpy(img.transpose((2,0,1)))
        
        return img.float().div(255)


    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im
        
        
    def encode(self,normal_label):
        
        target = torch.zeros((7,7,26))
        for label in normal_label:
            x = int(label[6])
            y = int(label[7])
#             print(normal_label[:4])
#             target[x,y,:4] = torch.tensor(label[:4])
            target[y][x][:4] = torch.tensor(label[:4])
            
            
#             target[x,y,5:9] = torch.tensor(label[:4])
            target[y][x][5:9] = torch.tensor(label[:4])
            
            target[y][x][4] = 1
            target[y][x][9] = 1
            target[y][x][10:] = torch.tensor(label[4])
#             print(torch.tensor(label[4]))
#         print(target)
        return target
            
    



# train_data = Load_Data(train_label,train_fname, train_img, train_iname)