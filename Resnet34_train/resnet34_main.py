#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:12:42 2018

@author: junyang
"""

from __future__ import print_function
import argparse
import os
import os.path as osp
import shutil
import random
#from data_utils import get_train_test_data
import numpy as np
import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
import pdb
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from model import resnet34
from net_ArcFace import ArcFace
from dataset import ImageList
from torchvision.datasets import ImageFolder
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils
#import pdb
from torch.autograd import Variable
def print_network(model,name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))
   
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--out_class', type=int, default=62338, help='number of classes')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.1')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD. default=0.9')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay parameter. default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--Resnet34', default='', help="path to Resnet34 (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--train_list', default='/data/dataset/ms-celeb-1m/processed/ms-celeb-1m/training_list_without_deduplication.txt', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')
gpu_id = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
print('GPU: ',gpu_id)
opt = parser.parse_args()
print(opt)

out_class = opt.out_class
#root_path = opt.train_dataroot

try:
    os.makedirs(opt.outf)
except OSError:
    pass
 
shutil.copyfile('./resnet34_main.py',osp.join(opt.outf,'resnet34_main.py'))
shutil.copyfile('./model.py',osp.join(opt.outf,'model.py'))
shutil.copyfile('./draw_loss_curve.py',osp.join(opt.outf,'draw_loss_curve.py'))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

box = (16, 17, 214, 215)
transform=transforms.Compose([transforms.Lambda(lambda x: x.crop(box)),
                             transforms.Resize((230,230)),
                             #transforms.Resize(opt.imageSize),                            
                             transforms.RandomGrayscale(p=0.1),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(),
                             transforms.RandomCrop((opt.imageSize,opt.imageSize)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
tensor_dataset = ImageList(opt.train_list,transform)
                          
dataloader = DataLoader(tensor_dataset,                          
                        batch_size=opt.batchSize,    
                        shuffle=True,    
                        num_workers=opt.workers)  


ngpu = int(opt.ngpu)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 :
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
        #m.weight.data.normal_(1.0, 0.02)
        #m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
        #m.weight.data.normal_(0.0, 0.02)

def compute_accuracy(x, y):
     _, predicted = torch.max(x, dim=1)
     correct = (predicted == y).float()
     accuracy = torch.mean(correct) * 100.0
     return accuracy

Resnet34 = resnet34(num_classes=opt.out_class)

Resnet34.apply(weights_init)

criterion = nn.CrossEntropyLoss()


if opt.cuda:
    Resnet34.cuda()
    criterion.cuda()

if ngpu>1:
    Resnet34 = nn.DataParallel(Resnet34)

if opt.Resnet34 != '':
    Resnet34.load_state_dict(torch.load(opt.Resnet34))
print_network(Resnet34, 'Resnet34')

optimizer = optim.SGD(Resnet34.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay = opt.weight_decay)
Resnet34.train()
cnt = 0
loss_log = []
print('initial learning rate is: {}'.format(opt.lr))
for epoch in range(opt.niter):
    if epoch == 5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2.0
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/5.0
            print('lower learning rate to {}'.format(param_group['lr']))
    elif epoch == 15 or epoch == 20:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.0
            print('lower learning rate to {}'.format(param_group['lr']))
    for i, (real_cpu,label) in enumerate(dataloader,0):
        cnt += 1
        Resnet34.zero_grad()
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
            label = label.cuda()                    
        inputv = Variable(real_cpu)
        labelv = Variable(label)
        out = Resnet34(inputv,labelv)
        loss = criterion(out,labelv)
        loss.backward()
        optimizer.step()
        if (i+1)%opt.log_step == 0:
            accuracy = compute_accuracy(out,labelv).data[0]           
            print ('Epoch[{}/{}], Iter [{}/{}], training loss: {} , accuracy: {} %'.format(epoch+1,opt.niter,i+1,len(dataloader),loss.data[0],accuracy))
    torch.save(Resnet34.state_dict(), '%s/Resnet34_epoch_%d.pth' % (opt.outf, epoch))        
    loss_log.append([loss.data[0]])
        
loss_log = np.array(loss_log)
plt.plot(loss_log[:,0], label="Training Loss")
plt.legend(loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
filename = os.path.join('./', ('Loss_log_'+time1_str+'.png'))
plt.savefig(filename, bbox_inches='tight')
