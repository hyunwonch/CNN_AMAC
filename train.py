import sys
import os
import os.path as pth
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import math
import pickle
import random
import tqdm

from io import BytesIO
from datetime import datetime
from pytz import timezone
from slacker import Slacker

import nets
import datasets
import tools
import shutil



from quantization import *
from slack import *

import warnings
warnings.simplefilter("ignore")

# =====================================
# Training configuration params
# =====================================




###############################################
################# Arguments ###################
###############################################
def parse():

    # Default Setting
    mon = str(time.localtime().tm_mon)
    day = str(time.localtime().tm_mday)
    hour = str(time.localtime().tm_hour)
    minu = str(time.localtime().tm_min)
    now = mon + "_" + day + "_Time_" + hour + "_" + minu

    file = open('latest.txt' , 'r' )
    line = file.readline()
    pretrained_checkpoint = line
    file.close()

    arch = "CNN_627_large"
    dataset = "cifar10"
    #batch_size = 256

    #model_names = ['resnet20', 'vggnagamult', 'cnnc', 'vggnagacnn', 'resnet20cnn', 'cnnc-conv', 'resnet18cnn', 'resnet18']
    dataset_names = ['cifar10', 'cifar100', 'imagenet', 'mnist']

    # Start Parsing
    parser = argparse.ArgumentParser(description='PyTorch HTNN Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default=arch,
                        help='model architecture (default: CNN_627)')
    parser.add_argument('--dataset', metavar='DATA', default=dataset,
                        choices=dataset_names,
                        help='dataset (default: cifar10')
    parser.add_argument('--quant', metavar='quant', default='0', help='Do quant or not')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs (default: 200')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='Base learning rate (default: 0.001')
    args = parser.parse_args()


    parser.add_argument('--dataset_dir', metavar='path', default='./data_{}'.format(args.dataset), help='dataset path')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_train/{}_{}/{}/'.format(args.arch, args.dataset, now),
                            help='Folder to save checkpoints')
    parser.add_argument('--log_path', type=str, default='./checkpoints_train/{}_{}/{}/train.log'.format(args.arch, args.dataset, now),
                        help='Path to write logs into')
    args = parser.parse_args()

    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay rate (default: 1e-4')
    parser.add_argument('--hmt_weight_decay', type=float, default=5e-5,
                        help='HMT Weight decay rate (default: 5e-5')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pretrained model')
    args = parser.parse_args()

    return args



###############################################
############## Several Scheduler ##############
###############################################
def lr_schedule_vgg(optimizer, epoch, base_lr=0.01):
    lr = base_lr
    if epoch >= 50:
        lr = base_lr * (0.5 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'''
# def lr_schedule_resnet(optimizer, epoch, base_lr=0.1):
#     lr = base_lr
#     if epoch > 200:
#         lr *= 1e-3
#     elif epoch > 150:
#         lr *= 1e-2
#     elif epoch > 100:
#         lr *= 1e-1
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# def lr_schedule_resnet18(optimizer, epoch, base_lr=0.1):
#     lr = base_lr * (0.5 ** (epoch // 8))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


'''




###############################################
################## Training ###################
###############################################
def train(model, trainloader, testloader, args):
    log = tools.StatLogger(args.log_path)

    # ===================================
    # initialize and run training session
    # ===================================
    model.cuda()

    """
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, L.IWHT2Layer) or isinstance(m, L.IWHT3Layer) \
        or isinstance(m, L.WHT2Layer) or isinstance(m, L.WHT3Layer) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.MaxPool2d):
            m.register_forward_hook(get_dump_act(n))
    """

    lossfunc = nn.CrossEntropyLoss().cuda()

    # HMT_w = dict()
    # CNN_w = dict()
    # for name, param in model.named_parameters():
    #     if 'layer1.0' in name:
    #         HMT_w[name] = param
    #     else:
    #         CNN_w[name] = param

    # optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = optim.SGD([{'params': HMT_w.values(), 'weight_decay': args.hmt_weight_decay},
    #                        {'params': CNN_w.values()}],
    #                        lr=args.base_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)#, weight_decay=args.weight_decay)
    print("----------     Start epoch    ----------")


    # Open quantized dataset
    if args.dataset == 'cifar10':
        with open("./data_quantized/quant_data.pkl","rb") as f:
            data_list = pickle.load(f)
        with open("./data_quantized/quant_label.pkl","rb") as g:
            label_list = pickle.load(g)
    elif args.dataset == 'mnist':
        print("mnist dataset")
        with open("./data_quantized/quant_data_mnist.pkl","rb") as f:
            data_list = pickle.load(f)
        with open("./data_quantized/quant_label_mnist.pkl","rb") as g:
            label_list = pickle.load(g)

    # with open("./data_quantized/quant_test_data.pkl","rb") as k:
    #     data_test_list = pickle.load(k)
    # with open("./data_quantized/quant_test_label.pkl","rb") as y:
    #     label_test_list = pickle.load(y)


    
    # Start epoch
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        lr_schedule_vgg(optimizer, epoch, args.base_lr)


        # print("----------     Scheduler    ----------")
        print('Training : current lr {}'.format(optimizer.param_groups[0]['lr']))

        epoch += 1

        forward = 0
        backward = 0
        model.train()
        error_top1 = []
        error_top5 = []
        running_loss = []

        # Quantize
        if args.quant == '1':
            for n, m in model.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.weight.data = quant_signed_0(m.weight.data)
                    m.bias.data = quant_signed_0(m.bias.data)
                else:
                    pass
            print("########## Done quantization")
        else:
            print("########## No quantization")
        model.cuda()
        

        for idx, data in enumerate(trainloader,0):
            num = random.sample(range(50000),args.batch_size)
            data_num = []
            label_num = []
            for i in num:
                data_num.append(data_list[i])
                label_num.append(label_list[i])
            # get the inputs; data is a list of [inputs, labels]
            if args.arch == 'mnist' or args.arch =='mnist_quant':
                inputs, labels = torch.stack(data_num).view([args.batch_size,1,32,32]), torch.tensor(label_num)
            else:
                inputs, labels = torch.stack(data_num), torch.tensor(label_num)

            #inputs, labels = torch.stack(data_num).view([256,1,28,28]), torch.tensor(label_num)
            #inputs, labels = torch.stack(data_num).view([args.batch_size,3,32,32]), torch.tensor(label_num)
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # print("Batch [{}/{}]: Forward = {} us".format(idx, len(trainloader), forward))

            loss = lossfunc(outputs, labels)
            
            loss.backward()

            # print("Batch [{}/{}]: Backward = {} us".format(idx, len(trainloader), backward))
            
            optimizer.step()

            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            running_loss.append(loss.item())

        # print("----------     Finish inner for loop    ----------")

        # Quantize
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
        model.cuda()

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        running_loss = np.average(running_loss)
        # print statistics
        print("Training : epoch:%-4d acc_top1: %.2f %% acc_top5: %.2f %% loss:%.2f" % (epoch, (1-error_top1)*100, (1-error_top5)*100, running_loss))
        log.report(epoch=epoch,
                   split='TRAIN',
                   error_top5=float(error_top5),
                   error_top1=float(error_top1),
                   loss=float(running_loss))
        



        # di, div = divmod(epoch,10)
        # if div == 0:
        # print('Training : current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, os.path.join(args.checkpoint_dir, 'checkpoint_{}_{}.tar'.format(epoch,str((1-error_top1)*100)[:4])))
        print('Training : saved model check point')

        file  = open('./latest.txt' , 'w' )
        file.write(os.path.join(args.checkpoint_dir, 'checkpoint_{}_{}.tar'.format(epoch,str((1-error_top1)*100)[:4]))) 
        file.close()    
        #slack('epoch : {}'.format(epoch))

    print('Finished Training')


###############################################
################# Evaluation ##################
###############################################
def eval(model, args, testloader):

    print("Running evaluation on validation split")
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

    model.cuda()
    model.eval()

    lossfunc = nn.CrossEntropyLoss().cuda()
    error_top1 = []
    error_top5 = []
    vld_loss = []


    if args.dataset == 'cifar10':
        with open("./data_quantized/quant_test_data.pkl","rb") as k:
            data_test_list = pickle.load(k)
        with open("./data_quantized/quant_test_label.pkl","rb") as y:
            label_test_list = pickle.load(y)
    elif args.dataset == 'mnist':
        with open("./data_quantized/quant_test_data_mnist.pkl","rb") as k:
            data_test_list = pickle.load(k)
        with open("./data_quantized/quant_test_label_mnist.pkl","rb") as y:
            label_test_list = pickle.load(y)


    with torch.no_grad():
        for idxx, datax in enumerate(testloader, 0):
            num = random.sample(range(10000),50)
            data_test_num = []
            label_test_num = []
            for i in num:
                data_test_num.append(data_test_list[i])
                label_test_num.append(label_test_list[i])
            #inputs, labels = torch.stack(data_test_num), torch.tensor(label_test_num)
            inputs, labels = torch.stack(data_test_num).view([50,1,28,28]), torch.tensor(label_test_num)
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            error_top1.append(tools.topK_error(outputs, labels, K=1).item())
            error_top5.append(tools.topK_error(outputs, labels, K=5).item())
            vld_loss.append(lossfunc(outputs, labels).item())

        error_top1 = np.average(error_top1)
        error_top5 = np.average(error_top5)
        vld_loss = np.average(vld_loss)
        print("-- Validation result -- acc_top1: %.2f %% acc_top5: %.2f %% loss:%.4f" % ((1-error_top1)*100, (1-error_top5)*100, vld_loss))
        slack('''
-------------------------------------------------
                Training Finished
-------------------------------------------------
-- Quantization : %s
-- Validation result -- acc_top1: %.2f%%
-- Validation result -- acc_top5: %.2f%%
-------------------------------------------------
              ''' % (args.quant, (1-error_top1)*100, (1-error_top5)*100))


###############################################
#################### Main #####################
###############################################
def main():

    print(time.ctime())

    args = parse()

    # If want to change settings
    #args.dataset = 'cifar10'
    #args.arch = 'CNN_627_large'
    #args.quant = '1'
    #args.batch_size = 256
    #args.pretrained_weights = True
    args.base_lr = 0.001

    #args.pretrained_weights = './checkpoints_train/{}_{}'.format(args.arch, args.dataset) + "/2_14_Time_21_30/checkpoint_1_80.0.tar"
    # print(args.pretrained_weights)

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

    if args.dataset == 'cifar10':
        trainloader, _, testloader = datasets.get_cifar10(args.batch_size)
    elif args.dataset == 'mnist':
        trainloader, _, testloader = datasets.get_mnist(args.batch_size)
    else:
        print("Wrong dataset, choose between Cifar10 or MNIST")

    print(args.pretrained_weights)
    if args.pretrained_weights is not None:
        pretrained_ckpt = torch.load(args.pretrained_weights)
        model.load_state_dict(pretrained_ckpt['state_dict'])
        print("Loaded checkpoint '{}'".format(args.pretrained_weights))
    else:
        print("No Loaded checkpoint")



    try:
        #setup checkpoint directory
        if not pth.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
            print("Directory created")
            file = open(os.path.join(args.checkpoint_dir, 'info.txt') , 'w' )
            file.write("arch : {} \n".format(args.arch)) 
            file.write("Dataset : {} \n".format(args.dataset)) 
            file.write("Batchsize : {} \n".format(args.batch_size)) 
            file.write("Quantization : {} \n".format(args.quant)) 
            file.write(str(model.parameters))
            #file.write("Summary : {} \n".format() 
            file.close()
        else:
            print("Directory already exist")

        train(model, trainloader, testloader, args)
        try:    
            eval(model, args, testloader) 
        except Exception as e: 
            print("-------------------- Evaluation Error -------------------- ")
            print(e)
            print("Evaulation Error, but training complete")
    except Exception as e: 
        print("-------------------- Training Error -------------------- ")
        print(e)
        shutil.rmtree(args.checkpoint_dir)
        print('Directory deleted')
        slack("Error Happened")


# Main function
if __name__ == '__main__':
    main()
