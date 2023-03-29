import sys
import os
import os.path as pth

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
import argparse
import pickle
import random
import math

import nets
import datasets
import tools
import layers as L
import train

from io import BytesIO
from datetime import datetime
from pytz import timezone
from slacker import Slacker

from quantization import *
from slack import *

import warnings
warnings.simplefilter("ignore")


'''
# def dump_act(module, input, output):
#     if len(output) > 0:
#         input_act_list.append(input[0].detach().cpu().numpy())
#         output_act_list.append(output[0].detach().cpu().numpy())

# # Calculate weight density
# def cal_density(model):
#     num_pruned, num_weights = 0, 0
#     for m in model.modules():
#         if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
#             num = torch.numel(m.weight.data)
#             weight_mask = (abs(m.weight.data) > 0).float()

#             num_pruned += num - torch.sum(weight_mask)
#             num_weights += num

#     return 1 - num_pruned / num_weights



# def dump_batch(model, testloader, batch_size, arch):
#     print("Dumping batch for simulation")
#     for n, m in model.named_modules():
#         # if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
#         if isinstance(m, nn.Conv2d):
#             if m.kernel_size == (3, 3):
#                 m.register_forward_hook(dump_act)
#                 weight_list.append(m.weight.data.detach().cpu().numpy())
#     model.eval()
#     with torch.no_grad():
#         for data in testloader:
#             inputs = data[0][0:batch_size, :, :, :]
#             inputs = inputs.cuda()
#             model(inputs)
#             break

#     for i in range(0, len(weight_list)):
#         np.save("./py_sim_dump/{}/wgt-layer_{}.npy".format(arch, i), weight_list[i])
#         np.save("./py_sim_dump/{}/act-layer_{}-{}.npy".format(arch, i, batch_size), input_act_list[i])
'''

def parse():

    file = open('latest.txt' , 'r' )
    line = file.readline()
    pretrained_checkpoint = line
    file.close()

    # Default settings for arch, dataset, and checkpoint
    arch = "mnist"
    dataset = "mnist"
    pretrained_checkpoint = pretrained_checkpoint

    # Choices
    #model_names = ['resnet20', 'vggnagamult', 'cnnc', 'vggnagacnn', 'resnet20cnn', 'cnnc-conv', 'resnet18cnn']
    dataset_names = ['cifar10', 'cifar100', 'imagenet', 'mnist']

    # Start parsing
    parser = argparse.ArgumentParser(description='PyTorch HTNN Evaluation')
    parser.add_argument('--arch', '-a', metavar='ARCH', default=arch,
                        help='model architecture (default: resnet20)')
    parser.add_argument('--dataset', metavar='DATA', default=dataset,
                        choices=dataset_names,
                        help='dataset (default: cifar10')
    parser.add_argument('--pretrained_checkpoint', metavar='PRETRAINED', default=pretrained_checkpoint,
                        choices=dataset_names,
                        help='pretrained_checkpoint')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size (default: 256)')
    
    parser.add_argument('--dataset_dir', metavar='path', default='./data_quantized', help='dataset path')
    parser.add_argument('--quant', metavar='quant', default='1', help='Do quant or not')
    parser.add_argument('--slack', metavar='slack', default='0', help='send msg or not')

    # Include above arguments
    args = parser.parse_args()

    return args





# Evaluation
def eval(model, args, testloader):

    print("########## Running evaluation on validation split")
    model.cuda()
    model.eval()

    lossfunc = nn.CrossEntropyLoss().cuda()
    error_top1 = []
    error_top5 = []
    vld_loss = []

    data_len = 10000
    if args.dataset == 'cifar10':
        with open("./data_quantized/quant_test_data.pkl","rb") as k:
            data_test_list = pickle.load(k)
        with open("./data_quantized/quant_test_label.pkl","rb") as y:
            label_test_list = pickle.load(y)
    elif args.dataset =='mnist':
        print('MNIST dataset')
        with open("./data_quantized/quant_test_data_mnist.pkl","rb") as k:
            data_test_list = pickle.load(k)
        with open("./data_quantized/quant_test_label_mnist.pkl","rb") as y:
            label_test_list = pickle.load(y)

    if args.quant == '1':
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = quant_signed_05(m.weight.data)
                m.bias.data = quant_signed_05(m.bias.data)
            else:
                pass
        print("########## Done quantization")
    else:
        print("########## No quantization")

    #print(model.conv1.weight*8)
    
    model.cuda()
    with torch.no_grad():
        for idxx, datax in enumerate(testloader, 0):
            num = random.sample(range(data_len),args.batch_size)
            data_test_num = []
            label_test_num = []
            for i in num:
                data_test_num.append(data_test_list[i])
                label_test_num.append(label_test_list[i])
            if args.arch == 'mnist' or args.arch =='mnist_quant':
                inputs, labels = torch.stack(data_test_num).view([args.batch_size,1,32,32]), torch.tensor(label_test_num)
            else:
                inputs, labels = torch.stack(data_test_num), torch.tensor(label_test_num)
            
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            vld_loss.append(lossfunc(outputs, labels).item())

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        vld_loss = np.average(vld_loss)
        print("########## Validation result -- acc_top1: %.4f acc_top5: %.4f loss:%.4f" % (1-error_top1, 1-error_top5, vld_loss))
        store = args.pretrained_checkpoint.split("/")
        store1 = store[3]
        store2 = store[4]
        if args.slack == '1':
            slack('''
    -------------------------------------------------
    -- Model saved in : %s
    -- Model name  is : %s
    -- Quantization : %s
    -- Validation result -- acc_top1: %.2f%%
    -- Validation result -- acc_top5: %.2f%%
    -------------------------------------------------
                ''' % (store1, store2, args.quant, (1-error_top1)*100, (1-error_top5)*100))
        else:
            pass
    

# Main function include parse, quant, eval
def main():
    
    
    print("############################################# Start #############################################")

    #dataset = 'cifar10'
    #arch = 'CNN_627'
    #pretrained_checkpoint = pretrained_checkpoint
    args = parse()

    if args.dataset == 'cifar10':
        trainloader, _, testloader = datasets.get_cifar10(args.batch_size)
    elif args.dataset == 'mnist':
        trainloader, _, testloader = datasets.get_mnist(args.batch_size)
    else:
        trainloader, _, testloader = datasets.get_cifar10(args.batch_size)

    if args.arch == 'CNN_627_small':
        model = nets.CNN_627_small()
    elif args.arch == 'CNN_627_large':
        model = nets.CNN_627_large()
    elif args.arch == 'mnist_quant':
        model = nets.mnist_quant()
    elif args.arch == 'mnist':
        model = nets.mnist()
    elif args.arch == 'VGGnagaCNN':
        model = nets.VGGnagaCNN()
    elif args.arch == 'VGGnagaCNN_quant':
        model = nets.VGGnagaCNN_quant()
    else:
        model = nets.CNN_627_large()

    # args.pretrained_checkpoint = "./checkpoints_train/VGGnagaCNN_cifar10/2_15_Time_17_50/checkpoint_56_99.9.tar"
    args.pretrained_checkpoint = "./checkpoints_train/mnist_mnist/2_18_Time_16_22/checkpoint_101_99.7.tar"

    # load pretrained checkpoint
    pretrained_ckpt = torch.load(args.pretrained_checkpoint)
    model.load_state_dict(pretrained_ckpt['state_dict'])
    print("########## Loaded checkpoint '{}'".format(args.pretrained_checkpoint))

    
    eval(model,args, testloader)

    print("############################################# Finish #############################################")

if __name__ == '__main__':#####
    main()
