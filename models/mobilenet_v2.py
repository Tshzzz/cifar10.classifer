# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:53 2018

@author: tshzzz
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottlenBlock(nn.Module):

    def __init__(self,in_planes,out_planes,expansion,stride=1):
        super(BottlenBlock,self).__init__()
        
        self.stride = stride
        
        planes = in_planes*expansion
        
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,groups=planes,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,out_planes,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        
        self.pass_by = nn.Sequential()
        
        if stride == 1 or in_planes != out_planes:
            self.pass_by = nn.Sequential(
                        nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=1,bias=False),
                        nn.BatchNorm2d(out_planes)
                    )

    def forward(self,x):

        out = F.relu(self.bn1(self.conv1(x)))
        
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))

        if self.stride == 1:

            out += self.pass_by(x)

        return out



class MobileNet_v2(nn.Module):
    
    net_cfg = [ [1,16,1,1],
                [6,24,2,2],
                [6,32,3,2],
                [6,64,4,2],
                [6,96,3,1],
                [6,160,3,2],
                [6,320,1,1]
            ]
    
    
    def __init__(self,num_class=10):
        super(MobileNet_v2,self).__init__()

        self.conv1 = nn.Conv2d(3,32,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bottlen = self.make_layers(32)
        self.conv3 = nn.Conv2d(320,1280,kernel_size=1,stride=1,padding=0,bias=False)

        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1280,num_class)
                )

    def make_layers(self,in_planes):
        layers = []
        
        for expansion,out_plane,num_block,stride in self.net_cfg:
            
            for i in range(num_block):
                
                layers.append(BottlenBlock(in_planes,out_plane,expansion,stride))
                in_planes = out_plane
                stride = 1
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.bottlen(out)
        out = self.conv3(out)
        out = out.mean(3).mean(2)
        out = self.fc(out)    
        
        return out
        

    
def test():

    net = MobileNet_v2(10)
    count = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
                
    print(net)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

if __name__ == "__main__":
    test()


