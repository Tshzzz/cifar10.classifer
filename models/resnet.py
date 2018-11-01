# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:28:39 2018

@author: tshzzz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.pass_by = nn.Sequential()
        if stride != 1 or self.expansion*in_planes != planes:
            self.pass_by = nn.Sequential(
                        nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride,bias=False),
                        nn.BatchNorm2d(planes)
                    )

    def forward(self,x):
        #print(self.conv1)
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))
        
        out += self.pass_by(x)
        
        out = F.relu(out)
        
        return out


class BottlenBlock(nn.Module):
    expansion = 4
    
    
    def __init__(self,in_planes,planes,stride=1):
        super(BottlenBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        
        self.pass_by = nn.Sequential()
        
        if stride != 1 or in_planes*self.expansion != planes:
            self.pass_by = nn.Sequential(
                        nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=stride,bias=False),
                        nn.BatchNorm2d(planes*self.expansion)
                    )

    def forward(self,x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))

        out += self.pass_by(x)

        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    
    def __init__(self,block,layers,num_class=10):
        super(ResNet,self).__init__()
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_class)        
            
            
    def _make_layer(self,block,plane,layer,stride=1):

        
        layers = []
        layers.append(block(self.inplanes,plane,stride))
        
        self.inplanes = plane*block.expansion

        for i in range(1,layer):
            
            layers.append(block(self.inplanes,plane))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)

        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet50():
    model = ResNet(BottlenBlock, [3, 4, 6, 3])
    return model

    
def test():
    net = ResNet50()
    
    print(net)
    y = net(torch.randn(1,3,224,224))
    print(y.size())

#test()





