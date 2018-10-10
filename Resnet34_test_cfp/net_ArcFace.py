import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30, m=0.5):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        nn.init.xavier_normal(self.weight)
        #self.weight.data = F.normalize(self.weight.data, p=2, dim=0)
        print('num_classes: ',out_features)
        print('init sucessfully')
        #self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m		
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm0 = self.sin_m * self.m
        #print(self.mm0)        

    def forward(self, input, target):
        #x = input   # size=(B,F)    F is feature len
        #print ('x',x)
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features
        x = F.normalize(input, p=2, dim=1)
        ww = F.normalize(w, p=2, dim=0)
        #print('ww',ww)
        
        cos_theta = x.mm(ww) # size=(B,Classnum)
        #cos_theta = cos_theta.clamp(-1,1)
        
        target = target.view(-1,1) #size=(B,1)		
        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        #index = index.byte()
        index = Variable(index)
        index_reverse = 1.0 - index
        
        sin_theta = 1.0 - cos_theta.pow(2)
        sin_theta = sin_theta.sqrt()
        cos_m_theta = cos_theta * self.cos_m - sin_theta * self.sin_m
        cond_v = cos_theta - self.threshold
        replace_index = (F.relu(cond_v)==0).float()
        keep_val = cos_theta - self.mm0
        #cos_m_theta[replace_index] = keep_val[replace_index] #bug at the pytorch 0.3.0 version
        #cos_m_theta[replace_index.detach()] = keep_val[replace_index.detach()]
        replace_index_reverse = 1.0 - replace_index
        cos_m_theta = keep_val * replace_index + cos_m_theta * replace_index_reverse

        output = self.s *( cos_theta * index_reverse + cos_m_theta * index)
        return output # size=(B,Classnum)


