from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #set default group to 2
    def __init__(self,norm_type="BN",grp_size=2):
        super(Net, self).__init__()
        self.grp_size = grp_size
        self.norm = norm_type
        self.convblk1 = nn.Sequential(nn.Conv2d(1, 10, 3, padding=1),
                                      self.NormFn(10),
                                      nn.ReLU())
        self.convblk2 = nn.Sequential(nn.Conv2d(10, 20, 3, padding=1),
                                      self.NormFn(20),
                                      nn.ReLU())
        self.transitionBlock1 = nn.MaxPool2d(2,2)
        self.convblk3 = nn.Sequential(nn.Conv2d(20,20,3,padding=1),
                                      self.NormFn(20),
                                      nn.ReLU())
        self.convblk4 = nn.Sequential(nn.Conv2d(20,20,3,padding=1),
                                      self.NormFn(20),
                                      nn.ReLU())
        self.transitionBlock2 = nn.MaxPool2d(2,2)
        self.convblk5 = nn.Sequential(nn.Conv2d(20,10,1),
                                      self.NormFn(10),
                                      nn.ReLU())
        #input 20x7x7 -? OUtput 10x1x1? RF 33
        self.avgPoolblk = nn.AvgPool2d(7,7)
    def forward(self, x):
      #maxpool must be used at least after 2 convolution and sud be as far as possible from last layer
        x = self.convblk1(x)         #input 1x28x28 -? OUtput 10x28x28? RF 3
        x = self.convblk2(x)         #input 10x28x28 -? OUtput 20x28x28? RF 5 
        x = self.transitionBlock1(x) #input 20x28x28 -? OUtput 20x14x14? RF 6
        x = self.convblk3(x)         #input 20x14x14 -? OUtput 20x14x14? RF 10 
        x = self.convblk4(x)         #input 20x14x14 -? OUtput 20x14x14? RF 14  
        x = self.transitionBlock2(x) #input 20x14x14 -? OUtput 20x7x7? RF 16 
        x = self.convblk5(x)         #input 20x7x7 -? OUtput 10x7x7? RF 16 
        x = self.avgPoolblk(x)       #input 20x7x7 -? OUtput 10x7x7? RF 40 
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
    def NormFn(self,channels):
      #custom normalization function based on the input
      if self.norm == 'GN':
        print(f'(Group Normalization)')
        return nn.GroupNorm(int(channels/self.grp_size),channels)
      elif self.norm == 'LN':
        print(f'(Layer Normalization)')
        return nn.LayerNorm(channels)
      else : #by default apply batch normalization
        print(f'(Batch Normalization)') 
        return nn.BatchNorm2d(channels) 
