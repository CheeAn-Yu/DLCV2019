import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
# from dataset.data_loader import GetLoader
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from skimage import io, transform
import cv2
import numpy as np
import pandas as pd
import sys
from os import listdir


class MNIST(Dataset):
    def __init__(self, root_dir, transform=None):
        """ Intialize the MNIST dataset """
        self.root_dir = os.path.join(root_dir)
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.file = sorted(listdir(self.root_dir))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img_name = os.path.join(self.root_dir,self.file[index])
        image = io.imread(img_name)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # label = self.landmarks_frame['label'][index]
        # label = torch.FloatTensor([label])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.file)

class USPS(Dataset):
    def __init__(self, root_dir, transform=None):
        """ Intialize the MNIST dataset """
        self.root_dir = os.path.join(root_dir)
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.file = sorted(listdir(self.root_dir))
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img_name = os.path.join(self.root_dir,self.file[index])
        image = io.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # label = self.landmarks_frame['label'][index]
        # label = torch.FloatTensor([label])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.file)


def write_csv(predict):
    save_path = sys.argv[3]
    img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
    csv={'label':predict,'image_name':img_name}
    df = pd.DataFrame(csv,columns=['image_name','label'])
    df.to_csv(os.path.join(save_path),index=0)

def to_csv():
    dataset_name = sys.argv[2]
    print(dataset_name)
    model_root = os.path.join('.','model')
    
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0


    if dataset_name == 'svhn':

        dataset =  MNIST(root_dir=os.path.join(sys.argv[1]),transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        my_net = torch.load(os.path.join(model_root, 'mnist_svhn_model6.pth'))

    elif dataset_name == 'mnistm':

        dataset =  MNIST(root_dir=os.path.join(sys.argv[1]),transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        my_net = torch.load(os.path.join(model_root, 'usps_mnist_model4.pth'))


    elif dataset_name == "usps":

        dataset = USPS(root_dir=os.path.join(sys.argv[1]),transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])) 
        my_net = torch.load(os.path.join(model_root, 'svhn_usps_model7.pth'))


    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )


    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    predict = []
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        # class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            # t_label = t_label.cuda()
            input_img = input_img.cuda()
            # class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        # class_label.resize_as_(t_label.long()).copy_(t_label.long())

        class_output, _ ,_= my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        # n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        
        predict += pred.squeeze().cpu().tolist()
        # n_total += batch_size

        i += 1

    # accu = n_correct.data.numpy() * 1.0 / n_total

    # print (' accuracy of the %s dataset: %f' % (dataset_name, accu))
    write_csv(predict)

if __name__ == "__main__":
    to_csv()