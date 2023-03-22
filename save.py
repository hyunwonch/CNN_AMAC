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

network = "CNN_627"
dataset = "cifar10"

model_names = ['resnet20', 'vggnagamult', 'cnnc', 'vggnagacnn', 'resnet20cnn', 'cnnc-conv', 'resnet18cnn']
dataset_names = ['cifar10', 'cifar100', 'imagenet']
parser = argparse.ArgumentParser(description='PyTorch HTNN Evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default=network,
                    choices=model_names,
                    help='model architecture (default: resnet20)')
parser.add_argument('--dataset', metavar='DATA', default=dataset,
                    choices=dataset_names,
                    help='dataset (default: cifar10')
parser.add_argument('--dataset_dir', metavar='path', default='/tmpssd/pabillam/ILSVRC/Data/CLS-LOC', help='dataset path')
parser.add_argument('--quant', metavar='quant', default='1', help='Do quant or not')



def quant_signed_1(original, bit=6):
    
    bit = bit-2
    a = torch.flatten(original)
    length = len(a)
    output = []

    for i in range(len(a)):
        if a[i].item() >= 1:
            val = 1
        else:
            val = math.trunc(a[i].item()*(2**bit))/(2**bit)
        output.append(val)

    output = torch.tensor(output, dtype=torch.float64)
    output = torch.reshape(output, original.shape)
    torch.set_printoptions(precision=bit)
    output = output.float()
    output = output.cuda()
    return output


def main():
    args = parser.parse_args()



    args.dataset = 'cifar10'
    args.arch = 'CNN_627'

    pretrained_checkpoint = './checkpoints_quant/CNN_627_cifar10/Sun_Feb_12_14_09_22_2023_train_checkpoints/checkpoint_200_0.00.tar'
    pretrained_checkpoint = './checkpoints_train/CNN_627_cifar10/2_12_Time_22_30/checkpoint_170.tar'
    model = nets.CNN_627()
    #model.cuda()

    output_act_list = []
    input_act_list = []
    weight_list = []

    #_, _, testloader = datasets.get_cifar10()

    # file = open('hello.txt', 'r')    # hello.txt 파일을 읽기 모드(r)로 열기. 파일 객체 반환
    # s = file.read()                  # 파일에서 문자열 읽기
    # print(s)                         # Hello, world!
    # file.close()    

    # load pretrained checkpoint
    pretrained_ckpt = torch.load(pretrained_checkpoint)
    model.load_state_dict(pretrained_ckpt['state_dict'])
    print("Loaded checkpoint '{}'".format(pretrained_checkpoint))


    # if args.quant == '1':
    #     for n, m in model.named_modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.kernel_size == (3, 3):
    #                 m.weight.data = quant_signed_1(m.weight.data)
    #         elif isinstance(m, nn.Linear):
    #             m.bias.data = quant_signed_1(m.bias.data)
    #     print("Done quantization")
    # else:
    #     print("No quantization")

    #print(model.conv1_1.weight.data)

    print(model.conv1_1.weight.shape)

    (row, col, dep, filter) = model.conv1_1.weight.shape
    print(row)
    file  = open('./weight.txt' , 'w' )
    print(str(model.children))

    #file.write(tf.tensor.toString(model.conv1_1.weight[0])) 
    file.close()    
    #np.savetxt('weight1.txt', model.conv1_1.weight.data[0][0].cpu().numpy())

    #density = cal_density(model)
    #print("-- Weight density after learning sparsity and CSD quantization: %.4f" % density)

    


if __name__ == '__main__':
    main()