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
import os
import argparse
from utils import txt_logger
import torch.optim.lr_scheduler as lr_scheduler

from models import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='MobileNet_v2')
    parser.add_argument('--output', type=str, default='./saved_/')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

args = parse_args()

def load_pretrain(pred_dict,model):
    model_dict = model.state_dict()
    for (k,v),(k1,v1) in zip(pred_dict.items(),model_dict.items()):
        if v.shape == v1.shape:
            model_dict[k1] = v
    return model_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([transforms.Resize([36,36]),
                                      transforms.RandomCrop([32,32]),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])

eval_transforms = transforms.Compose([transforms.Resize([36,36]),
                                      transforms.CenterCrop([32,32]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

trainsets = torchvision.datasets.CIFAR10('./datasets/',train=True,transform=train_transforms,download=True)
trainloader = torch.utils.data.DataLoader(trainsets,batch_size = args.batch_size,shuffle = True)
evalsets = torchvision.datasets.CIFAR10('./datasets/',train=False,transform=eval_transforms)
evalloader = torch.utils.data.DataLoader(evalsets,batch_size = 5)

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


def train(model,
          optimizer,
          criterion,
          epochs,
          out_dir,
          logger,
          trainloader,
          evalloader):

    step = 0
    best_acc = 0
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs // 9) + 1)

    for epoch in range(epochs):
    
        correct = 0.0
        total_loss = 0
        
        for img,label in trainloader:
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
        eval_acc = eval_train(model,evalloader)
        logger.add_scalar('eval score', eval_acc, step)
        if eval_acc > best_acc:
            best_acc = eval_acc
            model_path = os.path.join(out_dir, 'best.pth')
            torch.save(model.state_dict(), model_path)
        model_path = os.path.join(out_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        logger.print_info(epoch)
        scheduler(epoch)


if __name__ == '__main__':
    print(device)
    model = locals()[args.model]()
    model = model.to(device)

    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs
    mkdir(args.output)
    out_dir = args.output+args.model
    mkdir(out_dir)
    logger = txt_logger(out_dir, 'training', 'log.txt')
    train(model,optimizer,criterion,epochs,out_dir,logger,trainloader,evalloader)
    

    
    
    
    