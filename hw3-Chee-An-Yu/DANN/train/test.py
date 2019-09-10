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

def test(dataset_name, epoch):
    assert dataset_name in ['MNIST', 'SVHN']

    model_root = os.path.join('.','model')
    image_root = os.path.join('..', 'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    # img_transform_source = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])

    # img_transform_target = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    if dataset_name == 'SVHN':
        # test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        # dataset = GetLoader(
        #     data_root=os.path.join(image_root, 'mnist_m_test'),
        #     data_list=test_list,
        #     transform=img_transform_target
        # )
        dataset =  MNIST(csv_file="../../hw3_data/digits/svhn/test.csv", root_dir="../../hw3_data/digits/svhn/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    else:
        dataset = MNIST(csv_file="../../hw3_data/digits/mnistm/test.csv", root_dir="../../hw3_data/digits/mnistm/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_svhn_model' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label.long()).copy_(t_label.long())

        class_output, _ ,_= my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))

def write_csv(predict):
    img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
    csv={'label':predict,'image_name':img_name}
    df = pd.DataFrame(csv,columns=['image_name','label'])
    df.to_csv("./M2S_predict.csv",index=0)

def to_csv(dataset_name, epoch):
    assert dataset_name in ['MNIST', 'SVHN']

    model_root = os.path.join('.','model')
    image_root = os.path.join('..', 'dataset', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    # img_transform_source = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])

    # img_transform_target = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    if dataset_name == 'SVHN':
        # test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')

        # dataset = GetLoader(
        #     data_root=os.path.join(image_root, 'mnist_m_test'),
        #     data_list=test_list,
        #     transform=img_transform_target
        # )
        dataset =  MNIST(csv_file="../../hw3_data/digits/svhn/test.csv", root_dir="../../hw3_data/digits/svhn/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    else:
        dataset = MNIST(csv_file="../../hw3_data/digits/mnistm/test.csv", root_dir="../../hw3_data/digits/mnistm/test",transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_svhn_model' + str(epoch) + '.pth'
    ))
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
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label.long()).copy_(t_label.long())

        class_output, _ ,_= my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        
        predict += pred.squeeze().cpu().tolist()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    write_csv(predict)

if __name__ == "__main__":
    to_csv("SVHN",6)