
import torch
from torch.autograd import Variable
import torch.nn as nn

# from net import vgg16, vgg16_bn
# from resnet_yolo import resnet50
# import torchvision.transforms as transforms
import cv2
import numpy as np


idx_to_class={0:'plane', 1:'ship', 2:'storage-tank', 3:'baseball-diamond', 4:'tennis-court', 5:'basketball-court', 
6:'ground-track-field', 7:'harbor', 8:'bridge', 9:'small-vehicle', 10:'large-vehicle', 11:'helicopter', 
12:'roundabout', 13:'soccer-ball-field', 14:'swimming-pool', 15:'container-crane'}

def decoder(pred):
    '''
    pred (tensor) 1x7x7x26
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 7
    boxes=[]
    class_idx_all=[]
    probs = []
    pred = pred.data
    pred = pred.view(7,7,26) #7x7x26
    conf1 = pred[:,:,4].unsqueeze(2)
    conf2 = pred[:,:,9].unsqueeze(2)
    conf= torch.cat((conf1,conf2),2)
    mask1 = conf > 0.1 #大于阈值
    mask2 = (conf==conf.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    for i in range(grid_num):
        for j in range(grid_num):
            for k in range(2):
                if mask[i,j,k] == 1:
                    #print(i,j,b)
                    x_cen = j+pred[i,j,5*k+0]
                    y_cen = i+pred[i,j,5*k+1]
                    xmin = x_cen/7-pred[i,j,5*k+2]/2
                    ymin = y_cen/7-pred[i,j,5*k+3]/2
                    xmax = x_cen/7+pred[i,j,5*k+2]/2
                    ymax = y_cen/7+pred[i,j,5*k+3]/2
                    prob= pred[i,j,5*k+4]
                    box = [xmin,ymin,xmax,ymax]
                    max_prob,class_idx = torch.max(pred[i,j,10:],0)
                    #print(pred[i,j,10:])
                    #print(prob.item(),max_prob.item())
                    #print((prob*max_prob).item())
                    if float((prob*max_prob).item()) > 0.08: # 0.08
                        boxes.append(box)
                        probs.append((prob*max_prob).item()) 
                        class_idx_all.append(class_idx.item())

    if len(boxes) == 0:
        #print('nothing')
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        class_idx_all = torch.zeros(1)
        return 'nothing'
    else:
        #print(boxes,probs,class_idx_all)
        boxes = torch.tensor(np.array(boxes,dtype=np.float).reshape(-1,4))#.view(-1,4) #(n,4)
        probs = torch.tensor(probs)#.view(-1) #(n,)
        class_idx_all = torch.tensor(class_idx_all)#.view(-1) #(n,)
        #print(boxes,probs,class_idx_all)
        result = nms(boxes,probs)
    return boxes[result],class_idx_all[result],probs[result]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    xmin = bboxes[:,0]
    ymin = bboxes[:,1]
    xmax = bboxes[:,2]
    ymax = bboxes[:,3]
    areas = (xmax-xmin) * (ymax-ymin)

    _,order = scores.sort(0,descending=True)
    result = []
    while len(order) > 0:
        i = order[0]
        result.append(i)
        if len(order) == 1:
            break

        xmin_inter = xmin[order[1:]].clamp(min=xmin[i])
        ymin_inter = ymin[order[1:]].clamp(min=ymin[i])
        xmax_inter = xmax[order[1:]].clamp(max=xmax[i])
        ymax_inter = ymax[order[1:]].clamp(max=ymax[i])

        w = (xmax_inter-xmin_inter).clamp(min=0)
        h = (ymax_inter-ymin_inter).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (iou <= threshold).nonzero().view(-1)
        if len(ids) == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(result)
#
#start predict one image
#
def predict(pred):
    if pred=='nothing':
        return []

    result = []
    h,w = 512,512
    boxes,class_idx_all,probs = pred

    for i,box in enumerate(boxes):
        xmin = int(box[0]*w)
        ymin = int(box[1]*h)
        xmax = int(box[2]*w)
        ymax = int(box[3]*h)
        class_idx = int(class_idx_all[i])
        prob = float(probs[i])
        result.append([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,idx_to_class[class_idx],prob])
    return result

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)



# if __name__ == '__main__':
    '''
    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()
    image_name = 'dog.jpg'
    image = cv2.imread(image_name)
    print('predicting...')
    result = predict_gpu(model,image_name)
    for left_up,right_bottom,class_name,_,prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)
    '''
import csv
import models
from dataloader_v4 import Load_Data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# train_fname = [str(i).zfill(5)+".txt" for i in range(15000)]
# train_iname = [str(i).zfill(5)+".jpg" for i in range(15000)]
test_fname = [str(i).zfill(4)+".txt" for i in range(1500)]
test_iname = [str(i).zfill(4)+".jpg" for i in range(1500)]

# train_label = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/train15000/labelTxt_hbb/"
# train_img = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/train15000/images/"
test_img = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/val1500/images/"
test_label = "/home/robot/hw2-Chee-An-Yu/hw2_train_val/val1500/labelTxt_hbb/"

# train_data = Load_Data(train_label, train_fname, train_img, train_iname)
test_data = Load_Data(test_label, test_fname, test_img, test_iname, train=False)


# In[16]:


    # trainset_loader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=1)
testset_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

checkpoint_path="/home/robot/hw2-Chee-An-Yu/pass0.092/yolo-2500.pth"
model=models.Yolov1_vgg16bn(pretrained=True)
model.eval()
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
device=torch.device('cuda')
model.to(device)
print('load model...')
load_checkpoint(checkpoint_path, model, optimizer)
# model.load_state_dict(torch.load('yolo-5500.pth'))
#trainset = Yolotrainset('hw2_train_val/train15000/')
#trainloader = DataLoader(trainset, batch_size=1, shuffle=False)
# valset = Yolovalset('hw2_train_val/val1500/')
# valloader = DataLoader(valset, batch_size=1, shuffle=False)
print('load valset end')

# model.eval()

print('Test')

with torch.no_grad():
    for step, (batch_x,batch_y) in enumerate(testset_loader):

        #if step>2:
        #    break
        #print(step)
        y_pred = model(batch_x.to(device))
        result = predict(decoder(y_pred))
        #result = predict_gpu(model,image_name)
        #print(step,len(result))
        '''
        for i in range (len(result)):
            print('result:',result[i])
        '''
        #gt = predict(decoder(batch_y))

        #print(len(gt))
        '''
        for i in range (len(gt)):
            print('gt:',gt[i])
        '''
        f = open('predict/%04d.txt'%(step),"w")
        csv.register_dialect('outputfile',delimiter = ' ')
        w = csv.writer(f,dialect='outputfile')
        for i in range(len(result)):
            content = result[i]
            #content = [str(content[:]).replace(" ",",")]
            w.writerow(content) 



