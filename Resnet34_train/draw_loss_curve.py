#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:19:25 2018

@author: junyang
"""

import re
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os

def save_img(save_path,loss_set,loss_set_name):
    loss_set = np.array(loss_set)
    label_name = loss_set_name
    plt.figure()
    plt.plot(loss_set, label=label_name)
    plt.legend(loc='upper right')
    plt.xlabel("iter/10")
    plt.ylabel("Loss")
    filename = os.path.join(save_path, (label_name+'.png'))
    plt.savefig(filename, bbox_inches='tight')
s = os.getcwd()    
log_dir = osp.join(osp.split(osp.split(s)[0])[0],'log')
s = osp.split(s)[1]
time = s[-19:]
log_path = log_dir + '/log-' + time + '.log'
log_name = osp.split(log_path)[1].split('.')[0]
save_path = './loss_curve_' + log_name
if not osp.exists(save_path):
    os.makedirs(save_path)

loss_set = []
acc1_set = []
f_log = open(log_path,'r')
for line in f_log.readlines():
    s = line.rstrip().strip('\n')
    loss = re.findall(r"training loss: (.*) , accuracy",s)
    if loss == []:
        continue
    acc1 = re.findall(r"accuracy: (.*) %",s)
    loss = float(loss[0])
    acc1 = float(acc1[0])
    loss_set.append(loss)
    acc1_set.append(acc1)
save_img(save_path,loss_set,'training_loss')
save_img(save_path,acc1_set,'accuracy')
    
