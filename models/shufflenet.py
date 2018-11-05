# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:31:41 2018

@author: tshzzz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    
    def __init__(self,groups):
        super(ShuffleBlock,self).__init__()
        self.groups = groups
        
        
    
    def forward(self,x):
        '''
        [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]] -> [[1,4,2],[5,3,6]]        
        '''

        g = self.groups
        
        b,c,h,w = x.size()

        
        return x.view(b,g,c//g,h,w).permute(0,2,1,3,4).contiguous().view(b,c,h,w)






class BottlenBlock(nn.Module):

    
    def __init__(self,in_planes,out_planes,group,stride=1):
        super(BottlenBlock,self).__init__()
        
        self.stride = stride
        
        planes = out_planes//4
        
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False,groups=group)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.shuffle = ShuffleBlock(group)
        
        
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,groups=planes,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,out_planes,kernel_size=1,bias=False,groups=group)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        
        self.pass_by = nn.Sequential()
        
        if stride == 1 or in_planes != out_planes:
            self.pass_by = nn.Sequential(
                        nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=1,bias=False),
                        nn.BatchNorm2d(out_planes)
                    )

    def forward(self,x):

        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.shuffle(out)
        
        
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))

        if self.stride == 1:

            out += self.pass_by(x)

       
        #print(x.shape,out.shape)
        return out


class ShuffleNet(nn.Module):
    
    
    def __init__(self,cfg,group=3,num_class=10):
        super(ShuffleNet,self).__init__()
        self.cfg = cfg
        
        self.group = group
        self.conv1 = nn.Conv2d(3,24,kernel_size=3,stride=2,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.body = self.make_layers(24)
        
        self.pool2 = nn.AvgPool2d(7,stride=1)
        
        #self.fc = nn.Linear(cfg[-1],num_class)
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(cfg[-1],num_class)
                )
        
    def make_layers(self,inplane,num_layer=2):
        layers = []
        for plane in self.cfg:
            stride = 2
            for i in range(num_layer):
                layers.append(BottlenBlock(inplane,plane,self.group,stride=stride))
                inplane = plane
                stride = 1
        
        return nn.Sequential(*layers)


    def forward(self,x):
        
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.body(out)
        out = self.pool2(out)
        out = out.view(-1,self.cfg[-1])
        out = self.fc(out)
        
        return out
        
        
def shuffe_g3():

    cfg = [240,480,960]
    
    return ShuffleNet(cfg,3)
    
    
    
        
def test():

    net = shuffe_g3()
    count = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
                
    print(net)
    y = net(torch.randn(1,3,224,224))
    print(y.size()) 
        
#test()     
        
        







