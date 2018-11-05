#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:22:53 2018

@author: tshzzz
"""
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
import tqdm

import argparse

#import torchvision.models as models


from models.mobilenet import MobileNet
from models.resnet import ResNet50
from models.vgg import vgg16

from models.mobilenet_v2 import MobileNet_v2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--model', type=str, default='MobileNet_v2')
    parser.add_argument('--output', type=str, default='saved_/')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

args = parse_args()

def load_pretrain(pred_dict,model):
    
    #pred_dict = {k: v for k , v in pred_dict.items() if k in model_dict}
    #pred_dict.pop('fc.weight')
    #pred_dict.pop('fc.bias')
    #model_dict.update(pred_dict)
    #pred_dict = torch.load('/home/tshzzz/.torch/models/vgg16-397923af.pth')
    
    model_dict = model.state_dict()

    for (k,v),(k1,v1) in zip(pred_dict.items(),model_dict.items()):
        if v.shape == v1.shape:
            model_dict[k1] = v
            #print(k,k1)
    
    #dd
    return model_dict



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CHANNEL_MEAN = [0.4914, 0.4822, 0.4465]
CHANNEL_STD = [0.2023, 0.1994, 0.2010]
train_transforms = transforms.Compose([transforms.Resize([256,256]),
                                      transforms.RandomCrop([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(CHANNEL_MEAN,
                                                           CHANNEL_STD)])


eval_transforms = transforms.Compose([transforms.Resize([224,224]),
                                      transforms.CenterCrop([224,224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(CHANNEL_MEAN,
                                                           CHANNEL_STD)])

trainsets = torchvision.datasets.CIFAR10('./datasets/',train=True,transform=train_transforms)

trainloader = torch.utils.data.DataLoader(trainsets,batch_size = args.batch_size,shuffle = True)

evalsets = torchvision.datasets.CIFAR10('./datasets/',train=False,transform=eval_transforms)

evalloader = torch.utils.data.DataLoader(evalsets,batch_size = 5)


'''
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
'''

logger = SummaryWriter()

def mkdir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    
def eval_train(model,loader):
    model.eval()    
    correct = 0.0
    for img,label in loader:
            
        img = img.to(device)
        pred = model(img)
        _,pred_idx = torch.max(pred,1)
        correct += torch.sum(pred_idx.cpu() == label)
            
    eval_acc = float(correct) / len(loader.dataset)

    model.train()
    
    return eval_acc


def train(model,optimizer,criterion,epochs,out_dir,trainloader,evalloader):

    step = 0
    best_acc = 0

    for epoch in range(epochs):
    
        correct = 0.0
        total_loss = 0
        
        for img,label in tqdm.tqdm(trainloader):
            optimizer.zero_grad()
            label = label.to(device)
            img = img.to(device)
            pred = model(img)
            
            _,pred_idx = torch.max(pred,1)
            correct += torch.sum(pred_idx == label)
            
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
            step += 1
            
        total_loss /= len(trainloader.dataset)
        logger.add_scalar('train loss', total_loss, step)
        train_acc = float(correct) / len(trainloader.dataset)        
        logger.add_scalar('train score', train_acc, step)
    
        print('train_loss: {}  train_score: {} '.format(total_loss,train_acc))
        
        eval_acc = eval_train(model,evalloader)

        logger.add_scalar('eval score', eval_acc, step)
        
    
        if eval_acc > best_acc:
            best_acc = eval_acc
            model_path = os.path.join(out_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
                
        print('eval_score: {} best_score:{}'.format(eval_acc,best_acc))      


if __name__ == '__main__':

    model = locals()[args.model]()# vgg16()
    model = model.to(device)
    

    #pred_dict = torch.load('/home/tshzzz/.torch/models/vgg16_bn-6c64b313.pth')
    #model_dict = load_pretrain(pred_dict,model)
    #model.load_state_dict(model_dict)
    
    #logger(comment=args.model)
    #input_ = torch.randn(3,3,224,224)
    #logger.add_graph(model,(input_,))    
        
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = 60  
    
    out_dir = args.output+args.model
    mkdir(out_dir)
    
    train(model,optimizer,criterion,epochs,out_dir,trainloader,evalloader)
    

    
    
    
    