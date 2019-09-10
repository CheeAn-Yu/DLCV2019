from skimage import io, transform
from reader import readShortVideo
from reader import getVideoList
import os 
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from skimage import io, transform

vgg_feature = torchvision.models.vgg16(pretrained=True)
vgg_feature = vgg_feature.features.cuda()


vgg_feature = torchvision.models.vgg16(pretrained=True)
vgg_feature = vgg_feature.features.cuda()
# def padding_sequence()

class SeqenceLoader(Dataset):
    def __init__(self, video_root, csv_root, seq_len, transform=None):
        """ Intialize the  dataset """
        self.video_root = video_root
        self.all_video_frames = sorted(listdir(video_root))
        self.transform = transform
        self.df = pd.read_csv(csv_root,header=None,squeeze=True)
        self.seq_len = seq_len
    def __getitem__(self, index):
        
        if index < self.seq_len:
            seq = self.all_video_frames[:index+1]               
        else:
            seq = self.all_video_frames[index-self.seq_len+1:index+1]
#         print(seq)
        frames = []
        for img in seq:
            
            image = io.imread(os.path.join(self.video_root,img))
            if self.transform:
                image = self.transform(image)
                image = image.unsqueeze(dim=0)
    #             print(image.size())
                image = vgg_feature(image.cuda())
    #         print(seq,len(seq))
                frames.append(image.view(-1))
        
        frames = torch.stack(frames).squeeze()
            
            
        return frames, self.df[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.all_video_frames)


video_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/videos/train/"
num_video = sorted(listdir(video_root))
csv_root = "/home/robot/hw4-Chee-An-Yu/hw4_data/FullLengthVideos/labels/train/"
save_root = "/home/robot/hw4-Chee-An-Yu/seq2seq/train_feature"
num_csv = sorted(listdir(csv_root))
video_csv = list(zip(num_video,num_csv))

from tqdm import tqdm
for video,csv in video_csv:
    path = os.path.join(save_root,video)
    print(path)
    if not os.path.isdir(path):
#         print("fuck")
        os.mkdir(path)
    else:
        dataset = SeqenceLoader(video_root+video, csv_root+csv, seq_len=32,transform=transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.Pad((0,40), fill=0, padding_mode='constant'),
                                   transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                   
                               ]))
        # print("fuck")
        val_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,shuffle=False)
        for batch,(img, label) in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
#                 print("img_size",img.squeeze(dim=0).size())
                print(img.squeeze(dim=0).detach().cpu().size())
                print(os.path.join(save_root,video,str(batch).zfill(5)+".npy"))
                np.save(os.path.join(save_root,video,str(batch).zfill(5)+".npy"),img.squeeze(dim=0).detach().cpu())