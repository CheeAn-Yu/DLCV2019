import torch
import torch.cuda as tcuda
import torchvision.utils as tutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import math
from torch.utils.data import DataLoader, Dataset
import pandas as pd
# from dataset_usps import *
import itertools
from tqdm import tqdm
import os
from skimage import io, transform
import cv2

batchSize = 256
learningRate = 2.5e-4
dSteps = 1  # To train D more
numIterations = 500
weightDecay = 2.5e-5
betas = (0.5, 0.999)
numberHiddenUnitsD = 500
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(500, numberHiddenUnitsD)
        self.fc2 = nn.Linear(numberHiddenUnitsD, numberHiddenUnitsD)
        self.fc3 = nn.Linear(numberHiddenUnitsD, 2)
        self.bn1 = nn.BatchNorm1d(numberHiddenUnitsD)
        self.bn2 = nn.BatchNorm1d(numberHiddenUnitsD)
        
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.fc1(x)))
        out = F.leaky_relu(self.bn2(self.fc2(out)))
        return self.fc3(out)

class LeNetClassifier(nn.Module):
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        return self.fc2(x)


sourceTrainDataset = MNIST(csv_file="./hw3_data/digits/mnistm/train.csv", root_dir="./hw3_data/digits/mnistm/train",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

sourceTestDataset =  MNIST(csv_file="./hw3_data/digits/mnistm/test.csv", root_dir="./hw3_data/digits/mnistm/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

sourceTrainLoader = torch.utils.data.DataLoader(dataset=sourceTrainDataset, batch_size=batchSize, shuffle=True)
sourceTestLoader = torch.utils.data.DataLoader(dataset=sourceTestDataset, batch_size=batchSize, shuffle=False)

targetTrainDataset = MNIST(csv_file="./hw3_data/digits/svhn/train.csv", root_dir="./hw3_data/digits/svhn/train",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
targetTestDataset = MNIST(csv_file="./hw3_data/digits/svhn/test.csv", root_dir="./hw3_data/digits/svhn/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
targetTrainLoader = torch.utils.data.DataLoader(dataset=targetTrainDataset, batch_size=batchSize, shuffle=True)
targetTestLoader = torch.utils.data.DataLoader(dataset=targetTestDataset, batch_size=batchSize, shuffle=False)

sourceCNN = LeNet()
if ~tcuda.is_available():
    sourceCNN = torch.load('preTrainedCNNMNIST',map_location=lambda storage, loc: storage)
else:
    sourceCNN = torch.load('preTrainedCNNMNIST')

sourceCNN.eval()

targetCNN = LeNet()
if ~tcuda.is_available():
    targetCNN = torch.load('preTrainedCNNMNIST',map_location=lambda storage, loc: storage)
else:
    targetCNN = torch.load('preTrainedCNNMNIST')

targetCNN.train()

classifier = LeNetClassifier()
if ~tcuda.is_available():
    classifier = torch.load('preTrainedClassiferMNIST',map_location=lambda storage, loc: storage)
else:
    classifier = torch.load('preTrainedClassiferMNIST')

# classifier.eval()

if tcuda.is_available():
    sourceCNN.cuda()
    targetCNN.cuda()
    classifier.cuda()

correct = 0
total = 0

for param in sourceCNN.parameters():
    param.requires_grad = False

sourceCNN.eval()
D = Discriminator()
D.train()
targetCNN.train()
classifier.train()

Doptimizor = optim.Adam(D.parameters(),  lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
TargetOptimizor = optim.Adam(targetCNN.parameters(),  lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
# optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()),  lr=learningRate, weight_decay = weightDecay)
# ClassOptimizor = optim.Adam(classifier.parameters(), lr=learningRate, betas = betas, eps=1e-09, weight_decay= weightDecay)
criteria = torch.nn.CrossEntropyLoss()

# Following Labels are in reference of D:
sourceLabels = torch.zeros(batchSize, 1).long().squeeze()
targetLabels = torch.ones(batchSize, 1).long().squeeze()

if tcuda.is_available():
    D.cuda()
    targetCNN.cuda()
    sourceCNN.cuda()
    targetLabels = targetLabels.cuda()
    sourceLabels = sourceLabels.cuda()
    criteria.cuda()

i = 0
maxTargetAcc = 60
numValidation = 500 # 500
numEpochs = int(math.ceil(float(numIterations) / float(min(len(sourceTrainLoader), len(targetTrainLoader)))))
for currentEpoch in tqdm(range(numEpochs)):
    targetError = 0
    DError = 0
    for it, ((sourceImages, source_label), (targetImages, target_label)) in enumerate(tqdm(zip(sourceTrainLoader, targetTrainLoader))):

        if sourceImages.size(0) != targetImages.size(0):
            continue

        if tcuda.is_available():
            sourceImages = sourceImages.cuda()
            targetImages = targetImages.cuda()

        if sourceImages.size(1) == 3:
            sourceImages = rgb2grayWeights[0] * sourceImages[:,0,:,:] + rgb2grayWeights[1] * sourceImages[:,1,:,:] + rgb2grayWeights[2] * sourceImages[:,2,:,:]
            sourceImages.unsqueeze_(1)

        if targetImages.size(1) == 3:
            targetImages = rgb2grayWeights[0] * targetImages[:,0,:,:] + rgb2grayWeights[1] * targetImages[:,1,:,:] + rgb2grayWeights[2] * targetImages[:,2,:,:]
            targetImages.unsqueeze_(1)

        # Training D:
        D.zero_grad()

        sourceFeaturesForD = sourceCNN(Variable(sourceImages))
        targetFeaturesForD = targetCNN(Variable(targetImages))

        predictionOnSourceImagesForD = D(sourceFeaturesForD.detach())
        predictionOnTargetImagesForD = D(targetFeaturesForD.detach())
        predictionOnD = torch.cat((predictionOnSourceImagesForD, predictionOnTargetImagesForD), 0)
        labelsForD = torch.cat((sourceLabels, targetLabels), 0)

        DError = criteria(predictionOnD, Variable(labelsForD))
        DError.backward()

        Doptimizor.step()

        D.zero_grad()

        # Training Target:
        targetCNN.zero_grad()

        targetFeatures = targetCNN(Variable(targetImages))
        predictionOnTargetImages = D(targetFeatures)

        targetLabelsT = Variable(1 - targetLabels)

        TargetTargetError = criteria(predictionOnTargetImages, targetLabelsT)
        TargetTargetError.backward()

        if (i > 5):
            TargetOptimizor.step()
        
        targetCNN.zero_grad()

        targetError = TargetTargetError
        i = i + 1

        # Training for classify:
        # classifier.zero_grad()
        # source_label[torch.eq(source_label, 10)] = 0
        # source_label = troch.squeeze(source_label).long()
        # pred = classifier(targetFeatures)

        # class_loss = F.cross_entropy(pred, source_label)



        if (i-1) % 100 == 0:
            print('Train Itr: {} \t D Loss: {:.6f} \t Target Loss: {:.6f} \n '.format(
            i, DError.data, targetError.data))

            if (i - 1) % 100 == 0:

                correctT = 0
                totalT = 0
                correctD = 0
                totalD = 0
                j = 0
                for images, labelsTest in targetTestLoader:
                    if tcuda.is_available():
                        images, labelsTest= images.cuda(), labelsTest.cuda()

                    labelsTest = labelsTest.long()
                    labelsTest[torch.eq(labelsTest, 10)] = 0
                    if images.size(1) == 3:
                        images = rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                                 rgb2grayWeights[2] * images[:, 2, :, :]
                        images.unsqueeze_(1)

                    images = Variable(images)
                    outputs = classifier(targetCNN(images))
                    _, predicted = torch.max(outputs.data, 1)

                    totalT += labelsTest.size(0)
                    
                    correctT += (predicted == labelsTest.squeeze(1)).sum()
                    # print("here",predicted.size(), labelsTest.squeeze(1).size())
                    # print("fuck", labelsTest.size(0), correctT)
                    _, predictedD = torch.max(outputs.data, 1)
                    totalD += predictedD.size(0)
                    labelsT = torch.ones(predictedD.size()).long()
                    if tcuda.is_available():
                        labelsT = labelsT.cuda()

                    correctD += (predictedD == labelsT).sum()
                    j += 1
                    if j > numValidation:
                        break;

                currentAcc = 100 * correctT / totalT

                if currentAcc > maxTargetAcc:
                    torch.save(targetCNN, 'targetTrainedModel')
                    maxTargetAcc = currentAcc

                print('\n\nAccuracy of target on target test images: %d %%' % (100 * correctT / totalT))
                
                j = 0
                for images, labelsTest in sourceTestLoader:
                    if tcuda.is_available():
                        images, labelsTest = images.cuda(), labelsTest.cuda()

                    labelsTest = labelsTest.long()
                    labelsTest[torch.eq(labelsTest, 10)] = 0

                    if images.size(1) == 3:
                        images = rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                                 rgb2grayWeights[2] * images[:, 2, :, :]
                        images.unsqueeze_(1)

                    labelsTest.squeeze_()
                    images = Variable(images)
                    outputsDFromSource = D(sourceCNN(images))

                    _, predictedD = torch.max(outputsDFromSource.data, 1)
                    totalD += predictedD.size(0)
                    labelsT = torch.zeros(predictedD.size()).long()
                    if tcuda.is_available():
                        labelsT = labelsT.cuda()

                    correctD += (predictedD == labelsT).sum()
                    j += 1
                    if j > numValidation:
                        break;

                print('Accuracy of D on validation images: %d %%' % (100 * correctD / totalD))

# Save the Trained Model
torch.save(targetCNN, 'targetTrainedModelMNISTtoSVHN')
print('Max target accuracy achieved is %d %%' %maxTargetAcc)
targetCNN.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in sourceTestLoader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels = labels.long()
    labels[torch.eq(labels, 10)] = 0

    if images.size(1) == 3:
        images = rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    labels.squeeze_()
    images = Variable(images)
    outputs = classifier(targetCNN(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the source test images: %d %%' % (100 * correct / total))

correct = 0
total = 0
for images, labels in targetTestLoader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels = labels.long()
    labels[torch.eq(labels, 10)] = 0

    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(targetCNN(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the target test images: %d %%' % (100 * correct / total))

correct = 0
total = 0
for images, labels in targetTrainLoader:
    if tcuda.is_available():
        images, labels = images.cuda(), labels.cuda()

    labels = labels.long()
    labels[torch.eq(labels, 10)] = 0

    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(targetCNN(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the target train images: %d %%' % (100 * correct / total))