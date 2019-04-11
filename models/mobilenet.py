# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:23:31 2018

@author: tshzzz
"""

import torch
import torch.nn as nn


def conv_dw(inplane,outplane,stride=1):
    return nn.Sequential(
        nn.Conv2d(inplane,inplane,kernel_size = 3,groups = inplane,stride=stride,padding=1),
        nn.BatchNorm2d(inplane),
        nn.ReLU(),
        nn.Conv2d(inplane,outplane,kernel_size = 1,groups = 1,stride=1),
        nn.BatchNorm2d(outplane),
        nn.ReLU()    
        )

def conv_bw(inplane,outplane,kernel_size = 3,stride=1):
    return nn.Sequential(
        nn.Conv2d(inplane,outplane,kernel_size = kernel_size,groups = 1,stride=stride,padding=1),
        nn.BatchNorm2d(outplane),
        nn.ReLU() 
        )


class MobileNet(nn.Module):
    
    def __init__(self,num_class):
        super(MobileNet,self).__init__()
        
        layers = []
        layers.append(conv_bw(3,32,3,1))
        layers.append(conv_dw(32,64,1))
        layers.append(conv_dw(64,128,2))
        layers.append(conv_dw(128,128,1))
        layers.append(conv_dw(128,256,2))
        layers.append(conv_dw(256,256,1))
        layers.append(conv_dw(256,512,2))

        for i in range(5):
            layers.append(conv_dw(512,512,1))
        layers.append(conv_dw(512,1024,2))
        layers.append(conv_dw(1024,1024,1))

        self.classifer = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1024,num_class)
                )
        self.feature = nn.Sequential(*layers)
        
        

    def forward(self,x):
        out = self.feature(x)
        out = out.view(-1,1024)
        out = self.classifer(out)
        return out


def test():
    a = MobileNet(10)
    y = a(torch.randn(1,3,32,32))
    print(y.size())
    
    
if __name__ == "__main__":

    test()
