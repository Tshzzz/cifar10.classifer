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
    
    def __init__(self,layers,num_class=10):
        super(VGG,self).__init__()
        
        self.img_channel = 3
        self.layers = layers
        
        self.conv = self.make_layers()
        self.fc = nn.Linear(512*7*7, num_class)  
        self.avgpool = nn.AvgPool2d(1, stride=1)
        
    def make_layers(self):
        
        layers = []
        
        in_channel = self.img_channel
        
        
        for idx in range(len(self.layers)):
            if self.layers[idx] == 'M':          
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                #print(in_channel,self.layers[idx])
                layers.append(nn.Conv2d(in_channel,self.layers[idx],kernel_size=3,padding=1))
                layers.append(nn.BatchNorm2d(self.layers[idx]))
                layers.append(nn.ReLU())
                
                in_channel = self.layers[idx]
                    
        return nn.Sequential(*layers)
            
    
    def forward(self,x):
        
        out = self.conv(x)
        out = self.avgpool(out)
        out = out.view(-1,512*7*7)
        out = self.fc(out)
        
        return out
        

def vgg11():
    return VGG(cfg['vgg11'])

def vgg13():
    return VGG(cfg['vgg13']) 
       
def vgg16():
    return VGG(cfg['vgg16'])     
    
def test():
    import torchvision.models as models
    model = models.vgg16()
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)    
    #print(model)

    a = vgg16()        
    count = 0
    for m in a.modules():
        if isinstance(m, nn.Conv2d):
            count += 1
    print(count)
    
    y = a(torch.randn(3,3,224,224))
    print(y.size())
    
    
#test()


