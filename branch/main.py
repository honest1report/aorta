import os
import sys
import time
import torch
import random
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
from skimage import transform
import torch.utils.data as Data
from torch.autograd import Variable
import PUNET

LR = 0.001
WEIGHT = 10
BATCH_SIZE = 12
EPOCH  = 1000
SIZE  = 32
Z_SIZE = 48
IN_CHANNEL = 1
OUT_CHANNEL = 1

TRAIN_NUM = 36
TEST_NUM = 9
train_data_dir  = ""
train_label_dir = ""
test_data_dir  = ""
test_label_dir = ""
mark_path = ""   # (0 for healthy, 1 for dissection)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dataset(Data.Dataset):
    def __init__(self, image, label, weight,augmentation=False):
        self.image = image
        self.label = label
        self.weight = weight
        self.indexes = np.arange(len(self.image))

    def __getitem__(self, index):

        idx = self.indexes[index]
        image = self.image[idx]
        label = self.label[idx]
        weight = self.weight[idx]
        np.random.shuffle(self.indexes)
        return image, label, weight

    def __len__(self):
        return len(self.image)

def get_augmen_data(batch_size):
    mark = np.load(mark_path)  # (0 for healthy, 1 for dissection)
    TRAIN_NUM = len(train_image_list)
    images = []
    labels = []
    weights = []
    
    for i in range(TRAIN_NUM):
        for k in range(9):
            #(z,y,x)
            Image0 = np.load(train_data_dir + str(i+1)+"_"+str(k+1)+".npy")
            Label0 = np.load(train_label_dir + str(i+1)+"_"+str(k+1)+"_label.npy")
            SHAPE  = Image0.shape
            Image0 = Image0.transpose((2, 1, 0))   # (x,y,z)
            Label0 = Label0.transpose((2, 1, 0))

            for r in range(4):
                Image = transform.rotate(Image0, 90*r, preserve_range = True)
                Label = transform.rotate(Label0, 90*r, preserve_range = True)
                Image = Image.transpose((2, 1, 0))   #(z,y,x)
                Label = Label.transpose((2, 1, 0))

                start_p = (Z_SIZE-SHAPE[0])//2
                image_patch = np.zeros([Z_SIZE,SIZE,SIZE])
                image_patch[start_p:start_p+SHAPE[0]] = Image
                label_patch = np.zeros([Z_SIZE,SIZE,SIZE])
                label_patch[start_p:start_p+SHAPE[0]] = Label
                image_patch = image_patch[np.newaxis,:,:,:]
                label_patch = label_patch[np.newaxis,:,:,:]
                if mark[i,k] == 1:
                    for m in range(10):
                        images.append(image_patch)
                        labels.append(label_patch)
                        weights.append(mark[i,k])
                images.append(image_patch)
                labels.append(label_patch)
                weights.append(mark[i,k])
    
    dataset = Dataset(images, labels, weights, augmentation=False)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    print(len(images))
    return dataloader

def dice_coff(pred, label):
    inse = np.sum(np.multiply(pred, label))
    l = np.sum(np.multiply(pred, pred))
    r = np.sum(np.multiply(label, label))
    dice = 2.0*inse/(l+r)
    return dice

class DiceLoss(nn.Module):
    """Dice coeff"""
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, label, weight):
        for dim in range(4):
            weight = weight.unsqueeze(-1)
        label1 = weight*label
        pred1  = weight*pred
        l = torch.sum(torch.mul(pred1, pred1))
        r = torch.sum(torch.mul(label1, label1))
        inse = torch.sum(torch.mul(pred1, label1))
        dice1 = 2.0*inse/(l+r+1e-15)

        label0 = (1-weight)*label
        pred0  = (1-weight)*pred
        l = torch.sum(torch.mul(pred0, pred0))
        r = torch.sum(torch.mul(label0, label0))
        inse = torch.sum(torch.mul(pred0, label0))
        dice0 = 2.0*inse/(l+r+1e-15)

        dice = dice0 + 10*dice1
        return  -dice

def train(model):
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    diceloss = DiceLoss().cuda()
    dataloader = get_augmen_data(BATCH_SIZE)

    save_dir = "./PUnet_model/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(EPOCH):
        if epoch == 500:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        #trainning
        model.train()
        for step, (b_image,b_label,b_weight) in enumerate(dataloader):
            optimizer.zero_grad()
            with torch.no_grad():
                image = Variable(b_image.float().cuda())
                label = Variable(b_label.float().cuda())
                weight = Variable(b_weight.float().cuda())

            pred, aux2, aux1 = model(image)

            loss0 = diceloss(pred, label, weight)
            loss1 = diceloss(aux1, label, weight)
            loss2 = diceloss(aux2, label, weight)
            loss  = loss0 + 0.8*loss1 + 0.4*loss2

            print("epoch:{},step:{},loss:{}".format(epoch,step,loss0))
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        if epoch % 20 ==0:
            wholetime = int(time.time() - start_time)
            print("the whole time is {}h".format(wholetime/3600))
            torch.save(model.state_dict(),save_dir+str(epoch) +'_PUnet.pth')

    wholetime = int(time.time() - start_time)
    print("the trainning finished!, the whole time is {}h".format(wholetime/3600))
    torch.save(model.state_dict(),save_dir+str(epoch) +'_PUnet.pth')

def test(model):

    para = 999
    print("para is", para)
    para_path = "./PUnet_model/" + str(para) + '_PUnet.pth'
    model.load_state_dict(torch.load(para_path))
    model.eval()

    start_time = time.time()
    all_dice = 0
    count = 0

    for i in range(TEST_NUM):
        print(i+1)

        for j in range(9):

            IMAGE = np.load(test_data_dir+str(i+1)+"_"+str(j+1)+".npy")
            LABEL = np.load(test_label_dir+str(i+1)+"_"+str(j+1)+"_label.npy")
            predict  = np.zeros(IMAGE.shape)
            SHAPE = IMAGE.shape

            start_p = (Z_SIZE-IMAGE.shape[0])//2
            image_patch = np.zeros([Z_SIZE,SIZE,SIZE])
            image_patch[start_p:start_p+SHAPE[0]] = IMAGE
            image_patch = image_patch[np.newaxis,np.newaxis,:,:,:]

            with torch.no_grad():
                input_image = Variable(torch.from_numpy(image_patch)).cuda().float()
                pred, aux2, aux1 = model(input_image)
                pred = pred.data.cpu().numpy()
                predict = pred[0,0]
                predict = predict[start_p:start_p+SHAPE[0]]
            
            dice = dice_coff(predict, LABEL)
            all_dice += dice
            count += 1
            print("the {}th image's {}th branch's dice cofficient is {}".format(i+1, j+1, dice))

    wholetime = int(time.time() - start_time)
    print("the mean dice cofficient is {}".format(all_dice/count))
    print("the whole time is {}h".format(wholetime/3600))

if __name__ == '__main__':
    model = PUNET.Unet_3d(in_channel = 1).cuda()
    phase = sys.argv[1]

    if phase == "train":
        print("model training!")
        train(model)

    elif phase == "test":
        print("model testing!")
        test(model)

    else:
        print("wrong input!")