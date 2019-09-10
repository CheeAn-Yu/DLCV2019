import torch
import numpy as np
import skimage.io
import skimage
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import random
import pickle
import os
from os import listdir
import pandas as pd
from tqdm import tqdm
import sys

class Generator(nn.Module):
    def __init__(self, figsize=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 101, figsize * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(figsize * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(figsize * 8, figsize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(figsize * 4, figsize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(figsize * 2, figsize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(figsize, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
            
    def forward(self, X):
        output = self.decoder(X)/2.0+0.5
        return output
    
class Discriminator(nn.Module):
    def __init__(self, figsize=64):
        super(Discriminator, self).__init__()
        self.decoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, figsize, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(figsize, figsize * 2, 4, 2, 1),
            nn.BatchNorm2d(figsize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(figsize * 2, figsize * 4, 4, 2, 1),
            nn.BatchNorm2d(figsize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(figsize * 4, figsize * 8, 4, 2, 1),
            nn.BatchNorm2d(figsize * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(figsize * 8, figsize *1, 4, 1, 0),
        )
        self.fc_dis = nn.Linear(figsize *1, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(figsize *1, 1) # one class
        
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        decode_output = self.decoder(X)

        flat = decode_output.view(-1,64)
        fc_dis = self.fc_dis(flat)
        fc_aux = self.fc_aux(flat)
        
        realfake = self.sigmoid(fc_dis)
        classes = self.sigmoid(fc_aux)
        
        return realfake, classes

random.seed(122)
torch.manual_seed(122)
# use for random generation
up = np.ones(10)
down = np.zeros(10)
fixed_class = np.hstack((up,down))
fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
fixed_noise = torch.randn(10, 100, 1, 1)
fixed_noise = torch.cat((fixed_noise,fixed_noise))
fixed_input = Variable(torch.cat((fixed_noise, fixed_class),1))
latent_size = 100
BATCH_SIZE = 64

dis = Discriminator()
# print(dis)
model = Generator()
# print(model)
model.load_state_dict(torch.load("./ACG2_model92.pkt"))

fake = model(fixed_input).detach().cpu()
image = torchvision.utils.make_grid(fake, padding=2, normalize=True,nrow=10)
plt.imshow(np.transpose(image,(1,2,0)))
plt.savefig(sys.argv[1]+"fig2_2"+".jpg")
# plt.show()

# for i in range(200):
# 	random.seed(i)
# 	torch.manual_seed(i)
# 	# use for random generation

# 	up = np.ones(10)
# 	down = np.zeros(10)
# 	fixed_class = np.hstack((up,down))
# 	fixed_class = torch.from_numpy(fixed_class).view(20,1,1,1).type(torch.FloatTensor)
# 	fixed_noise = torch.randn(10, 100, 1, 1)
# 	fixed_noise = torch.cat((fixed_noise,fixed_noise))
# 	fixed_input = Variable(torch.cat((fixed_noise, fixed_class),1))
# 	latent_size = 100
# 	BATCH_SIZE = 64
# 	fake = model(fixed_input).detach().cpu()
# 	image = torchvision.utils.make_grid(fake, padding=2, normalize=True,nrow=10)
# 	plt.imshow(np.transpose(image,(1,2,0)))
# 	plt.savefig("./acgan_image/"+"image"+str(i)+".png")
