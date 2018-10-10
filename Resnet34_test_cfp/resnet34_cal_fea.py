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
import pickle
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
from dataset_identification import ImageList,CaffeCrop
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import pairwise_distances
import math
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
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--Resnet34', default='', help="path to resnet34 (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--dim_features', type=int, default=512, help='dim of features to use')
parser.add_argument('--gallery_list', default='/data/dataset/CFP/aligned_Pair_list_P.txt', type=str, metavar='PATH',
                    help='path to gallery list (default: none)')
parser.add_argument('--probe_list', default='', type=str, metavar='PATH',
                    help='path to probe list (default: none)')

parser.add_argument('--manualSeed', type=int, help='manual seed')
time1 = datetime.datetime.now()
time1_str = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
time1_str.replace(' ','_')
gpu_id = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
print('GPU: ',gpu_id)
opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass 

f = open(opt.gallery_list,'r')
savedir_set = []
pklpath_set = []
for line in f.readlines():
    index,imgPath = line.strip().rstrip('\n').split(' ')
    imgpath_split = imgPath.split('/')
    dir1 = imgpath_split[-3]
    dir2 = imgpath_split[-2]
    save_dir = osp.join(opt.outf,dir1,dir2)
    savedir_set.append(save_dir)
    pkl_name = imgpath_split[-1].split('.')[0] + '.pkl'
    pkl_path = osp.join(save_dir,pkl_name)
    pklpath_set.append(pkl_path)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
f.close()

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
                             transforms.CenterCrop((opt.imageSize,opt.imageSize)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
tensor_dataset_gallery = ImageList(opt.gallery_list,transform)

                    
dataloader_gallery = DataLoader(tensor_dataset_gallery,                    
                                batch_size=opt.batchSize,     
                                shuffle=False,     
                                num_workers=opt.workers)    


ngpu = int(opt.ngpu)

def cal_cosine_distance(a,b):
    dot_product = np.sum(a*b)
    norm_a = math.sqrt(np.sum(np.square(a)))
    norm_b = math.sqrt(np.sum(np.square(b)))
    cosine_similarity = float(dot_product) / (norm_a * norm_b)
    return cosine_similarity

def compute_accuracy(x, y):
     _, predicted = torch.max(x, dim=1)
     correct = (predicted == y).float()
     accuracy = torch.mean(correct) * 100.0
     return accuracy

Resnet34 = resnet34(num_classes=62338)

if opt.cuda:
    Resnet34.cuda()
Resnet34 = nn.DataParallel(Resnet34)
if opt.Resnet34 != '':
    Resnet34.load_state_dict(torch.load(opt.Resnet34))
Resnet34 = Resnet34.module

if ngpu>1:
    Resnet34 = nn.DataParallel(Resnet34)

Resnet34.eval()
cnt = 0

for i, (data) in enumerate(dataloader_gallery):
    print('processing %d th batch' %(i+1))
    batch_size = data.size(0)
    real_cpu = data
    if opt.cuda:
        real_cpu = real_cpu.cuda()
    inputv = Variable(real_cpu)
    feature = Resnet34(inputv)
    feature = feature.cpu().data.numpy()
    for j in range(batch_size):
        f_out = open(pklpath_set[cnt],'w')
        cnt += 1
        pickle.dump({'feature':feature[j,:]},f_out)
        f_out.close()
print('finished')
