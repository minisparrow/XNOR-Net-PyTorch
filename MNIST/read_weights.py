from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import models
import util
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import util

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        #DATA, TARGET = DATA.CUDA(), TARGET.CUDA()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]

    return


if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    pretrained_model = torch.load('models/LeNet_5.best.pth.tar')
    model.load_state_dict(pretrained_model['state_dict'])

    print(model)
    params = []
    
    param_dict = dict(model.named_parameters())
    for key,value in param_dict.items():
        print(key)
        print(value.shape)
        np.savetxt("weights/"+key,value.detach().numpy().reshape(-1,1),fmt="%10f")
