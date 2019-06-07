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

import util
import numpy as np

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/'+args.arch+'.best.pth.tar')


input_feature = {}
output_feature = {}
inter_gradient = {}

def make_hook(name, flag):
    if flag == 'forward':
       def hook(m, input, output):
          input_feature[name] = input
          output_feature[name] = output  
       return hook
    elif flag == 'backward':
       def hook(m, input, output):
          inter_gradient[name] = output
       return hook
    else:
       assert False

#    m.register_forward_hook(make_hook(name, 'forward'))
#    m.register_backward_hook(make_hook(name, 'backward'))

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0

    for data, target in test_loader:
        cnt += 1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
  
        model.conv_1.register_forward_hook(make_hook('conv_1', 'forward'))
        model.conv_2.register_forward_hook(make_hook('conv_2', 'forward'))
        model.fc_1.register_forward_hook(make_hook('fc_1', 'forward'))
        model.fc_2.register_forward_hook(make_hook('fc_2', 'forward'))
        model.fc_3.register_forward_hook(make_hook('fc_3', 'forward'))

        for key,value in input_feature.items():
            print(key)
            np.savetxt('inner_features/'+key+'_input', value[0].cpu().detach().numpy().reshape(-1,1), fmt='%10f')

        for key,value in output_feature.items():
            print(key)
            np.savetxt('inner_features/'+key+'_output', value[0].cpu().detach().numpy().reshape(-1,1), fmt='%10f')

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if cnt == 2:
            break

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))

    return

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
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
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    pretrained_model = torch.load('./models/LeNet_5.best.pth.tar')
    model.load_state_dict(pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()
    
    print(model)
    param_dict = dict(model.named_parameters())
    params = []
    
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.lr,
            'weight_decay': args.weight_decay,
            'key':key}]
    
    test()

    param_dict = dict(model.named_parameters())
    for key,value in param_dict.items():
        print(key)
        print(value.shape)
        np.savetxt("weights/"+key,value.detach().cpu().numpy().reshape(-1,1),fmt="%10f")

