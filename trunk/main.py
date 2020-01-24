import os
import glob
import sys
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import MODEL

LR = 0.001
BATCH_SIZE = 10
EPOCH  = 1500
STEP   = 50
START_P = 5
SIZE  = 64
SIZE_Z = 16
Z_GAP = 8
MEMORY_SIZE = 1500
IN_CHANNEL = 1
OUT_CHANNEL = 1
HIDEN_CHANNEL = 512

train_data_dir  = ""
train_label_dir = ""
test_data_dir  = ""
test_label_dir = ""
train_image_list = glob.glob(train_image_dir+"*.npy")
train_label_list = glob.glob(train_label_dir+"*.npy") 
test_image_list = glob.glob(test_image_dir+"*.npy")
test_label_list = glob.glob(test_label_dir+"*.npy") 

class data_set(object):
    def __init__(self, batch_size):
        self.memory_counter = 0
        self.batch_size = batch_size
        self.image = np.zeros([MEMORY_SIZE,IN_CHANNEL,SIZE_Z,SIZE,SIZE], dtype = np.float32)
        self.label = np.zeros([MEMORY_SIZE,OUT_CHANNEL,SIZE_Z,SIZE,SIZE], dtype = np.uint8)
        self.next_image = np.zeros([MEMORY_SIZE,IN_CHANNEL,SIZE_Z,SIZE,SIZE], dtype = np.float32)
        self.next_label = np.zeros([MEMORY_SIZE,OUT_CHANNEL,SIZE_Z,SIZE,SIZE], dtype = np.uint8)
        self.hidden = np.zeros([MEMORY_SIZE,HIDEN_CHANNEL,SIZE_Z//8,SIZE//8,SIZE//8], dtype = np.float32)

    def store_data(self, image, label, next_image, next_label, hidden):
        index = self.memory_counter % MEMORY_SIZE
        self.image[index] = image
        self.label[index] = label
        self.next_image[index] = next_image
        self.next_label[index] = next_label
        self.hidden[index] = hidden
        self.memory_counter += 1

    def pop_data(self):
        if self.memory_counter > MEMORY_SIZE:
            avaiable_num = MEMORY_SIZE
        else:
            avaiable_num = self.memory_counter

        sample_index = np.random.choice(avaiable_num, self.batch_size)

        b_image = self.image[sample_index]
        b_label = self.label[sample_index]
        next_image = self.next_image[sample_index]
        next_label = self.next_label[sample_index]
        b_hidden = self.hidden[sample_index]

        return b_image, b_label, next_image, next_label, b_hidden

class Dataset(Data.Dataset):
    def __init__(self, image, label, next_image, next_label,augmentation=False):
        self.image = image
        self.label = label
        self.next_img= next_image
        self.next_l  = next_label
        self.indexes = np.arange(len(self.image))

    def __getitem__(self, index):

        idx = self.indexes[index]
        image = self.image[idx]
        label = self.label[idx]
        next_image = self.next_img[idx]
        next_label = self.next_l[idx]
        np.random.shuffle(self.indexes)
        return image, label, next_image, next_label

    def __len__(self):
        return len(self.image)

def get_head_data(batch_size):
    images = []
    labels = []
    next_labels = []
    next_images = []
    
    for i in range(len(train_image_list)):
        #(z,y,x)
        Image = np.load(train_image_list[i])
        Label = np.load(train_label_list[i])

        for j in range(START_P):
            image_patch = Image[j:j+SIZE_Z]
            label_patch = Label[j:j+SIZE_Z]
            next_image  = Image[j+Z_GAP:j+Z_GAP+SIZE_Z]
            next_label  = Label[j+Z_GAP:j+Z_GAP+SIZE_Z]
            image_patch = image_patch[np.newaxis,:,:,:]
            label_patch = label_patch[np.newaxis,:,:,:]
            next_image  = next_image[np.newaxis,:,:,:]
            next_label  = next_label[np.newaxis,:,:,:]
            images.append(image_patch)
            labels.append(label_patch)
            next_images.append(next_image)
            next_labels.append(next_label)

    dataset = Dataset(images, labels, next_images, next_labels, augmentation=False)
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

    def forward(self, pred, label):
        inse = torch.sum(torch.mul(pred, label))
        l = torch.sum(torch.mul(pred, pred))
        r = torch.sum(torch.mul(label, label))
        dice = 2.0*inse/(l+r)
        return  -dice

def train(model):
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    diceloss = DiceLoss().cuda()
    dataloader = get_head_data(BATCH_SIZE)
    train_data = data_set(BATCH_SIZE)
    next_layer = MODEL.RUnet_3d(in_channel = 1).cuda()

    save_dir = "./RUnet_model/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    TRAIN_NUM = len(train_image_list)
    for epoch in range(EPOCH):
        if epoch == 600:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if epoch == 1000:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        img_num = random.randint(0, TRAIN_NUM-1)
        IMAGE = np.load(train_image_list[i])
        LABEL = np.load(train_label_list[i])
        start_p = random.randint(0, START_P)
        end_p = IMAGE.shape[0]

        patch_n = (end_p-start_p)//Z_GAP - 2
        left_gap = (end_p-start_p)%Z_GAP

        # collection data
        model.eval()
        with torch.no_grad():
            for i in range(patch_n):
                if i == 0:
                    image_patch = IMAGE[start_p:start_p+SIZE_Z]
                    image_patch = image_patch[np.newaxis,np.newaxis,:,:,:]
                    input_image = Variable(torch.from_numpy(image_patch)).cuda().float()
                    pred, aux2, aux1, ht = model(input_image)

                else:
                    start_p += Z_GAP
                    image_patch = IMAGE[start_p:start_p+SIZE_Z]
                    label_patch = LABEL[start_p:start_p+SIZE_Z]
                    image_next = IMAGE[start_p+Z_GAP:start_p+Z_GAP+SIZE_Z]
                    label_next = LABEL[start_p+Z_GAP:start_p+Z_GAP+SIZE_Z]
                    image_patch = image_patch[np.newaxis,:,:,:]
                    label_patch = label_patch[np.newaxis,:,:,:]
                    image_next  = image_next[np.newaxis,:,:,:]
                    label_next  = label_next[np.newaxis,:,:,:]

                    ht0 = ht.data.cpu().numpy()
                    train_data.store_data(image_patch,label_patch,image_next,label_next,ht0[0])

                    image_patch = image_patch[np.newaxis,:,:,:,:]
                    input_image = Variable(torch.from_numpy(image_patch)).cuda().float()
                    pred, aux2, aux1, ht = model(input_image,ht)

            if left_gap!= 0:
                start_p += Z_GAP
                image_patch = IMAGE[start_p:start_p+SIZE_Z]
                label_patch = LABEL[start_p:start_p+SIZE_Z]
                image_next = np.zeros([SIZE_Z,SIZE,SIZE])
                label_next = np.zeros([SIZE_Z,SIZE,SIZE])
                img = IMAGE[start_p+Z_GAP:]
                lab = LABEL[start_p+Z_GAP:]
                image_next[0:left_gap+Z_GAP] = img
                label_next[0:left_gap+Z_GAP] = lab
                image_patch = image_patch[np.newaxis,:,:,:]
                label_patch = label_patch[np.newaxis,:,:,:]
                image_next  = image_next[np.newaxis,:,:,:]
                label_next  = label_next[np.newaxis,:,:,:]

                ht0 = ht.data.cpu().numpy()
                train_data.store_data(image_patch,label_patch,image_next,label_next,ht0[0])

        # trainning
        model.train()
        next_layer.load_state_dict(model.state_dict())
        next_layer.eval()
        for step in range(STEP):
            optimizer.zero_grad()
            b_image,b_label,image_next,label_next,ht = train_data.pop_data()
            with torch.no_grad():
                image = Variable(torch.from_numpy(b_image)).cuda().float()
                label = Variable(torch.from_numpy(b_label)).cuda().float()
                image_next = Variable(torch.from_numpy(image_next)).cuda().float()
                label_next = Variable(torch.from_numpy(label_next)).cuda().float()
                ht = Variable(torch.from_numpy(ht)).cuda().float()

            pred, aux2, aux1, ht = model(image,ht)

            loss0 = diceloss(pred, label)
            loss1 = diceloss(aux1, label)
            loss2 = diceloss(aux2, label)

            pred_next, aux2, aux1, ht = next_layer(image_next,ht)
            loss_next0 = diceloss(pred_next, label_next)
            loss_next1 = diceloss(aux1, label_next)
            loss_next2 = diceloss(aux2, label_next)
            loss_next  = loss_next0 + 0.8*loss_next1 + 0.4*loss_next2
            loss  = loss0 + 0.8*loss1 + 0.4*loss2 + loss_next

            print("epoch:{},image_n:{},step:{},loss:{},loss_next:{}".format(epoch,img_num,step,loss0,loss_next0))

            
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        
        if epoch % 10 ==0:
            for step, (b_image,b_label,image_next,label_next) in enumerate(dataloader):
                optimizer.zero_grad()
                with torch.no_grad():
                    image = Variable(b_image.float().cuda())
                    label = Variable(b_label.float().cuda())
                    image_next = Variable(image_next.float().cuda())
                    label_next = Variable(label_next.float().cuda())

                pred, aux2, aux1, ht = model(image)

                loss0 = diceloss(pred, label)
                loss1 = diceloss(aux1, label)
                loss2 = diceloss(aux2, label)

                pred_next, aux2, aux1, ht = next_layer(image_next,ht)
                loss_next0 = diceloss(pred_next, label_next)
                loss_next1 = diceloss(aux1, label_next)
                loss_next2 = diceloss(aux2, label_next)
                loss_next  = loss_next0 + 0.8*loss_next1 + 0.4*loss_next2
                loss  = loss0 + 0.8*loss1 + 0.4*loss2 + loss_next

                print("epoch:{},step:{},loss:{},loss_next:{}".format(epoch,step,loss0,loss_next0))
                if optimizer is not None:
                    loss.backward()
                    optimizer.step()

        if epoch % 25 ==0:
            wholetime = int(time.time() - start_time)
            print("the whole time is {}h".format(wholetime/3600))
            torch.save(model.state_dict(),save_dir+str(epoch) +'_RUnet.pth')

    wholetime = int(time.time() - start_time)
    print("the trainning finished!, the whole time is {}h".format(wholetime/3600))
    torch.save(model.state_dict(),save_dir+str(epoch) +'_RUnet.pth')

def test(model):
    para = 1499 
    print("para is", para)
    para_path = './RUnet_model/' + str(para) + '_RUnet.pth'
    model.load_state_dict(torch.load(para_path))
    model.eval()

    test_num = len(test_image_list)
    start_time = time.time()
    all_dice = 0
    count = 0

    for i in range(test_num):
        print(i+1)
        count += 1
        IMAGE = np.load(test_image_list[i])
        LABEL = np.load(test_label_list[i])
        predict  = np.zeros(IMAGE.shape)
        start_p = 0
        end_p = IMAGE.shape[0]
        patch_n = (end_p-start_p)//Z_GAP - 1
        left_gap = (end_p-start_p)%Z_GAP

        with torch.no_grad():
            for step in range(patch_n):
                if step == 0:
                    image_patch = IMAGE[start_p:start_p+SIZE_Z]
                    image_patch = image_patch[np.newaxis,np.newaxis,:,:,:]
                    input_image = Variable(torch.from_numpy(image_patch)).cuda().float()
                    pred, aux2, aux1, ht = model(input_image)
                    pred = pred.data.cpu().numpy()
                    predict[start_p:start_p+SIZE_Z] = pred[0,0]

                else:
                    start_p += Z_GAP
                    image_patch = IMAGE[start_p:start_p+SIZE_Z]
                    image_patch = image_patch[np.newaxis,np.newaxis,:,:,:]
                    input_image = Variable(torch.from_numpy(image_patch)).cuda().float()
                    pred, aux2, aux1, ht = model(input_image,ht)
                    pred = pred.data.cpu().numpy()
                    predict[start_p:start_p+SIZE_Z] = pred[0,0]
                    
            if left_gap!= 0:
                left_gap += Z_GAP
                start_p += Z_GAP
                image_patch = np.zeros([SIZE_Z,SIZE,SIZE])
                image_patch[0:left_gap] = IMAGE[start_p:]
                image_patch = image_patch[np.newaxis,np.newaxis,:,:,:]

                input_image = Variable(torch.from_numpy(image_patch)).cuda().float()
                pred, aux2, aux1, ht = model(input_image,ht)
                pred = pred.data.cpu().numpy()
                predict[start_p:] = pred[0,0,0:left_gap]

        dice = dice_coff(predict, LABEL)
        all_dice += dice
        print("the {}th image's dice cofficient is {}".format(i+1, dice))

    wholetime = int(time.time() - start_time)
    print("the mean dice cofficient is {}".format(all_dice/count))
    print("the whole time is {}h".format(wholetime/3600))

if __name__ == '__main__':
    model = MODEL.RUnet_3d(in_channel = 1).cuda()
    phase = sys.argv[1]

    if phase == "train":
        print("model training!")
        train(model)

    elif phase == "test":
        print("model testing!")
        test(model)

    else:
        print("wrong input!")