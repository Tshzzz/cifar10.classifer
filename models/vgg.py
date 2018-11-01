# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:39:30 2018

@author: tshzzz
"""

import torch
import torch.nn as nn


cfg = {
       'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
       'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
       
       }


class VGG(nn.Module):
    
    def __init__(self,layers):
        super(VGG,self).__init__()
        
        self.img_channel = 3
        self.layers = layers
        
        self.conv = self.make_layers()
        
        
    def make_layers(self):
        
        layers = []
        
        in_channel = self.img_channel
        
        
        for idx in range(len(self.layers)):
            if self.layers[idx] == 'M':          
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channel,self.layers[idx],kernel_size=3,padding=1))
                layers.append(nn.ReLU())
                
                in_channel = self.layers[idx]
                    
        return nn.Sequential(*layers)
            
    
    def forward(self,x):
        
        out = self.conv(x)
        
        return out
        

def vgg11():
    return VGG(cfg['vgg11'])

def vgg13():
    return VGG(cfg['vgg13']) 
       
def vgg16():
    return VGG(cfg['vgg16'])     
    
def test():


    a = vgg16()        
    
    y = a(torch.randn(1,3,32,32))
    print(y.size())
    
    
#test()


