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
import random

class VideoLoader(Dataset):
    def __init__(self, video_root, csv_root, transform=None):
        """ Intialize the  dataset """
        self.all_video_path = []
        self.all_video_frames = []
        self.csv_root = csv_root
        self.video_root = video_root
        self.category_list = sorted(listdir(video_root))
        for category in self.category_list:
            video_in_folder = sorted(listdir(os.path.join(video_root,category)))
            video_path = ['-'.join(file_name.split('-')[:5]) for file_name in video_in_folder]
            self.all_video_path += video_path
            for video in video_in_folder:
                self.all_video_frames.append([category,video]) 
            
        df = pd.read_csv(self.csv_root)
        self.df = df.sort_values(['Video_name']).reset_index(drop=True)
        self.label = self.df['Action_labels'].tolist()
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
        # output = torch.mean(frames,0)
        # if self.transform:
            # frames = self.transform(frames)
        return output, self.label[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.all_video_path)


class RNNVideoLoader(Dataset):
    def __init__(self, video_root, csv_root, transform=None):
        """ Intialize the  dataset """
        self.all_video_path = []
        self.all_video_frames = []
        self.csv_root = csv_root
        self.video_root = video_root
        self.category_list = sorted(listdir(video_root))
        for category in self.category_list:
            video_in_folder = sorted(listdir(os.path.join(video_root,category)))
            video_path = ['-'.join(file_name.split('-')[:5]) for file_name in video_in_folder]
            self.all_video_path += video_path
            for video in video_in_folder:
                self.all_video_frames.append([category,video]) 
            
        df = pd.read_csv(self.csv_root)
        self.df = df.sort_values(['Video_name']).reset_index(drop=True)
        self.label = self.df['Action_labels'].tolist()
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
        # output = torch.mean(frames,0)
        # if self.transform:
            # frames = self.transform(frames)
        return frames, self.label[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.all_video_path)

class FeatureLoader(Dataset):
    def __init__(self, feature_root, csv_root, transform=None):
        """ Intialize the  dataset """
        self.csv_root = csv_root
        self.feature_root = feature_root
        
        self.feature = sorted(listdir(feature_root))
        self.label = sorted(listdir(csv_root))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """

        feature = np.load(os.path.join(self.feature_root,self.feature[index]))
        feature = torch.from_numpy(feature)
        label = np.load(os.path.join(self.csv_root,self.label[index]))
        label = torch.from_numpy(label)
        return feature, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.feature)

class SeqFeatureLoader(Dataset):
    def __init__(self, feature_root, csv_root, transform=None):
        """ Intialize the  dataset """
        self.csv_root = csv_root
        self.feature_root = feature_root
        
        self.feature = sorted(listdir(feature_root))
        self.label = pd.read_csv(self.csv_root, header=None, squeeze=True)
                           
    def __getitem__(self, index):
        """ Get a sample from the dataset """

        feature = np.load(os.path.join(self.feature_root,self.feature[index]))
        feature = torch.from_numpy(feature)
        label = self.label[index]
        label = torch.from_numpy(np.array(label))
        return feature, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.feature)

def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x:x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequence_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.stack([x[1] for x in sorted_batch])
#     print(sorted_batch)
    return sequence_padded, lengths, labels


class TrainSeqenceLoader(Dataset):
    def __init__(self, video_root, csv_root, seq_len, transform=None):
        """ Intialize the  dataset """
        self.video_root = video_root
        self.num_video = sorted(listdir(video_root))
        self.transform = transform
        self.csv_root = csv_root
        self.num_csv = sorted(listdir(csv_root))
#         self.df = pd.read_csv(csv_root,header=None,squeeze=True)
        self.seq_len = seq_len
    
    def __getitem__(self, index):
        
        video = os.path.join(self.video_root, self.num_video[index])
        frames_list = sorted(listdir(video))
        sample_list = sorted(random.sample(frames_list, k=self.seq_len))
#         print(sample_list)
        df = pd.read_csv(os.path.join(self.csv_root, self.num_csv[index]),header=None,squeeze=True)
#         print("df",df.values.tolist())
        df = df.values.tolist()
#         print(df)
        sample_list = sorted(random.sample(list(zip(frames_list,df)), k=self.seq_len))
#         print(sample_list)
        frames = []
        labels = []
        for img,label in sample_list:
#             image = io.imread(os.path.join(video,img))
            image = np.load(os.path.join(video,img))
            image = torch.from_numpy(image)
            frames.append(image)
            labels.append(torch.tensor([label]))
            
        frames = torch.stack(frames).cuda()    
        labels = torch.stack(labels).cuda()
            
        return frames, labels

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.num_video)

class ValSeqenceLoader(Dataset):
    def __init__(self, video_root, csv_root, transform=None):
        """ Intialize the  dataset """
        self.video_root = video_root
        self.num_video = sorted(listdir(video_root))
        self.transform = transform
        self.csv_root = csv_root
        self.num_csv = sorted(listdir(csv_root))
#         self.df = pd.read_csv(csv_root,header=None,squeeze=True)
       
    
    def __getitem__(self, index):
        
        video = os.path.join(self.video_root, self.num_video[index])
        frames_list = sorted(listdir(video))
        # print(len(frames_list))

        df = pd.read_csv(os.path.join(self.csv_root, self.num_csv[index]),header=None,squeeze=True)

        df = torch.tensor(df.values.tolist())

        frames = []
        labels = []
        for img in frames_list:

            image = np.load(os.path.join(video,img))
            image = torch.from_numpy(image)
            frames.append(image)
        
            
        frames = torch.stack(frames).cuda()    
            
        return frames, df.cuda()

    def __len__(self):
        
        return len(self.num_video)



if __name__ =='__main__':
    val_video_root = "./hw4_data/TrimmedVideos/video/valid"
    video_root = "./hw4_data/TrimmedVideos/video/train"
    train_csv = './hw4_data/TrimmedVideos/label/gt_train.csv'
    csv_root = './hw4_data/TrimmedVideos/label/gt_valid.csv'
    val_dataset = VideoLoader(val_video_root, csv_root,transform=None)
    train_dataset = VideoLoader(video_root, train_csv,transform=None)
    # print(len(train_dataset))
    image,label = train_dataset[0]
    print("image_size",image.size())
