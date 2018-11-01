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


#import torchvision.models as models


from models.mobilenet import MobileNet

logger = SummaryWriter()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

trainloader = torch.utils.data.DataLoader(trainsets,batch_size = 64,shuffle = True)

evalsets = torchvision.datasets.CIFAR10('./datasets/',train=False,transform=eval_transforms)

evalloader = torch.utils.data.DataLoader(evalsets,batch_size = 16)


'''
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
'''

model = MobileNet(10)
model = model.to(device)

out_dir = 'saved_'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

epochs = 30

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


    model.eval()
    correct = 0.0
    for img,label in evalloader:
        
        img = img.to(device)
        pred = model(img)
        _,pred_idx = torch.max(pred,1)
        correct += torch.sum(pred_idx.cpu() == label)
        
    eval_acc = float(correct) / len(evalloader.dataset)
    model.train()
    logger.add_scalar('eval score', eval_acc, step)
    

    if eval_acc > best_acc:
        best_acc = eval_acc
        model_path = os.path.join(out_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
            
    print('eval_score: {} best_score:{}'.format(eval_acc,best_acc))      



    
    
    