import sys
import os
import os.path as pth
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import itertools
import math
import pickle
import random
import time

import nets
import datasets
import tools
import layers as L

# =====================================
# Training configuration default params
# =====================================


network = "CNN_627"
dataset = "cifar10"

mon = str(time.localtime().tm_mon)
day = str(time.localtime().tm_mday)
hour = str(time.localtime().tm_hour)
minu = str(time.localtime().tm_min)
now = mon + "_" + day + "_Time_" + hour + "_" + minu

parser = argparse.ArgumentParser(description='PyTorch CNN Quantization')

parser.add_argument('--pretrained_weights', type=str, default='./{}_{}/Sun_Feb_12_13_00_35_2023_train_checkpoints/checkpoint_200.tar'.format(network, dataset),
                    help='Path to pretrained model')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_quant/{}_{}/{}/'.format(network, dataset, now),
                    help='Folder to save checkpoints')


parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size (default: 512)')
# parser.add_argument('--admm_epochs', type=int, default=30,
#                     help='Number of ADMM pruning epochs (default: 30')
parser.add_argument('--retraining_epochs', type=int, default=200,
                    help='Number of retraining epochs (default: 30')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='Base learning rate (default: 0.001')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay rate (default: 0.0001')

# parser.add_argument('--checkpoint_dir', type=str, default='./quant_cnn_checkpoints/{}_{}/'.format(network, dataset),
#                     help='Folder to save checkpoints')

parser.add_argument('--log_path', type=str, default='./logs/{}_{}/quant_cnn.log'.format(network, dataset),
                    help='Path to write logs into')
model_names = ['resnet20', 'vggnagamult', 'cnnc', 'vggnagacnn', 'resnet20cnn', 'cnnc-conv', 'resnet18cnn']
dataset_names = ['cifar10', 'cifar100', 'imagenet']
parser.add_argument('--arch', '-a', metavar='ARCH', default=network,
                    choices=model_names,
                    help='model architecture (default: vggnagacnn)')
parser.add_argument('--dataset', metavar='DATA', default=dataset,
                    choices=dataset_names,
                    help='dataset (default: cifar10')
parser.add_argument('--dataset_dir', metavar='path', default='/z/pabillam/imagenet', help='dataset path')


# Retrieve pruning masks
def retrieve_masks(model):
    num_pruned, num_weights = 0, 0
    weight_masks = []
    for m in model.modules():

        if isinstance(m, nn.Conv2d):
            num = torch.numel(m.weight.data)
            weight_mask = (abs(m.weight.data) > 0).float()
            weight_masks.append(weight_mask)

            num_pruned += num - torch.sum(weight_mask)
            num_weights += num

    # print('-- compress rate: %.4f' % (num_pruned / num_weights))
    return weight_masks


def apply_mask(model, weight_masks):
    idx = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, L.MultLayer) or isinstance(m, L.MultLayer3):
            m.weight.data *= weight_masks[idx]
            idx += 1



# Quant with 1 integer bit
def quant_signed_1(original, bit=6):
    bit = bit - 2
    original = original.clamp(max=1.9375,min=-1.9375)
    torch.set_printoptions(precision=bit)
    return ((original * (2**bit)).int()) / (2**bit)


# Quant with no integer bit
def quant_signed_0(original, bit=6):
    bit = bit - 1
    original = original.clamp(max=0.96875,min=-0.96875)
    torch.set_printoptions(precision=bit)
    return ((original * (2**bit)).int()) / (2**bit)



def train(model, trainloader, testloader, args):
    log = tools.StatLogger(args.log_path)

    # ===================================
    # initialize and run training session
    # ===================================
    model.cuda()
    lossfunc = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)

    # retrive pruning masks
    weight_masks = retrieve_masks(model)


    with open("./data_quantized/quant_data.pkl","rb") as f:
        data_list = pickle.load(f)


    with open("./data_quantized/quant_label.pkl","rb") as g:
        label_list = pickle.load(g)

    # with open("./quant_test_data.pkl","rb") as k:
    #         data_test_list = pickle.load(k)


    # with open("./quant_test_label.pkl","rb") as y:
    #     label_test_list = pickle.load(y)


    # # Apply quantization
    index = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (3, 3):
                m.weight.data = quant_signed_1(m.weight.data,8)
                #m.weight.data.cuda()
                index += 1

    # Retraining steps
    for epoch in range(args.retraining_epochs):
        epoch += 1
        model.train()
        error_top1 = []
        error_top5 = []
        running_loss = []

        

        for idx, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            num = random.sample(range(50000),256)
            data_num = []
            label_num = []
            for i in num:
                data_num.append(data_list[i])
                label_num.append(label_list[i])
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = torch.stack(data_num), torch.tensor(label_num)
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model.cuda()
            outputs = model(inputs)

            loss = lossfunc(outputs, labels)

            loss.backward()

            optimizer.step()

            # get masked weights
            apply_mask(model, weight_masks)

            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            running_loss.append(loss.item())

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        running_loss = np.average(running_loss)
        # print statistics
        print("Retraining : epoch:%-4d error_top1: %.2f error_top5: %.2f loss:%.4f" % (
            epoch, error_top1*100, error_top5*100, running_loss))
        log.report(epoch=epoch,
                   split='RETRAIN',
                   error_top5=float(error_top5),
                   error_top1=float(error_top1),
                   loss=float(running_loss))

        


        # Quantize again
        index = 0
        if args.quant == '1':
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.weight.data = quant_signed_0(m.weight.data)
                    m.bias.data = quant_signed_0(m.bias.data)
                else:
                    pass
            print("########## Done quantization")
            index += 1
        else:
            print("########## No quantization")
            


        di, div = divmod(epoch,10)
        if div == 0:
            print('Retraining : saving model check point')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, os.path.join(args.checkpoint_dir, 'checkpoint_{}_{}.tar'.format(epoch,str(error_top1)[:4])))

    print('Finished Retraining')



def main():
    args = parser.parse_args()


    if args.arch == 'CNN_627_small':
        model = nets.CNN_627_small()
    elif args.arch == 'CNN_627_large':
        model = nets.CNN_627_large()
    elif args.arch == 'mnist_quant':
        model = nets.mnist_quant()
    elif args.arch == 'mnist':
        model = nets.mnist()
    else:
        model = nets.CNN_627_large()

    args.dataset = 'cifar10'
    args.arch = 'CNN_627'
    pretrained_checkpoint = './checkpoints_quant/CNN_627_cifar10/Sun_Feb_12_14_09_22_2023_train_checkpoints/checkpoint_200_0.00.tar'
    # pretrained_checkpoint = './checkpoints/CNN_627_cifar10/Sun_Feb_12_14_09_22_2023_train_checkpoints/checkpoint_200_0.00.tar'
    model = nets.CNN_627()
    model.cuda()


    if args.dataset == 'cifar10':
        trainloader, _, testloader = datasets.get_cifar10(args.batch_size)
    elif args.dataset == 'mnist':
        trainloader, _, testloader = datasets.get_mnist(args.batch_size)
    else:
        trainloader, _, testloader = datasets.get_cifar10(args.batch_size)
    
    # load pretrained checkpoint
    if args.pretrained_weights is not None:
        pretrained_ckpt = torch.load(args.pretrained_weights)
        model.load_state_dict(pretrained_ckpt['state_dict'])
        print("Loaded checkpoint '{}'".format(args.pretrained_weights))
    else:
        print("No Loaded checkpoint")

    # setup checkpoint directory
    if not pth.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train(model, trainloader, testloader, args)


if __name__ == '__main__':
    main()
