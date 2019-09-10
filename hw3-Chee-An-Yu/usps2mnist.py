import torch
import torch.cuda as tcuda
import torchvision.utils as tutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from skimage import io, transform
import math
import pandas as pd
import os
from tqdm import tqdm
import cv2

numEpochs = 25
batchSize = 128
learningRate = 0.0005
weightDecay = 2.5e-4
rgb2grayWeights = [0.2989, 0.5870, 0.1140]

class MNIST(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """ Intialize the MNIST dataset """
        self.root_dir = root_dir
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[index,0])
        image = io.imread(img_name)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        label = self.landmarks_frame['label'][index]
        label = torch.FloatTensor([label])

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.landmarks_frame)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20 , 5)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.bn3  = nn.BatchNorm1d(500)
		
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.bn3(self.fc1(out)))
        return out

class LeNetClassifier(nn.Module):
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        return self.fc2(x)




model = torch.load("./preTrainedCNNUSPS")
classifier = torch.load("./preTrainedClassiferUSPS")

train_dataset = MNIST(csv_file="./hw3_data/digits/mnistm/train.csv", root_dir="./hw3_data/digits/mnistm/train",transform=transforms.Compose([

                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)

model.eval() 
correct = 0
total = 0
correctClass = torch.zeros(10)
totalClass = torch.zeros(10)
for images, labels in test_loader:

    if tcuda.is_available():
        # print("fuckkkkk")
        images, labels = images.cuda(), labels.cuda()

    labels[torch.eq(labels, 10)] = 0
    labels = torch.squeeze(labels).long()
    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(model(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    for i in range(len(correctClass)):
        classInPrediction = predicted == i
        classInLabels = labels == i
        correctClass[i] += (classInPrediction * classInLabels).sum()
        totalClass[i] += (classInLabels).sum()

print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))
for i in range(len(correctClass)):
    print('\nTest Accuracy of the model on the Class %d : %d %%' % (i, 100 * correctClass[i] / totalClass[i]))