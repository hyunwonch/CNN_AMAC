import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from quantization import *




class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RESNET20CNN(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[3,3,3], num_classes=10):
        super(RESNET20CNN, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


'''RESNET18CNN structure with 3 transforms for ImageNet
'''
class RESNET18CNN(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=1000):
        super(RESNET18CNN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
	
	# Zero-initialize the last BN in each residual branch
        for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, option='B'))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x



class mnist(nn.Module):
    # assuming 32x32x1 input_tensor

    def __init__(self, num_classes=10):
        super(mnist, self).__init__()
        #self.in_channel = 1
        
        # block 0 -- outputs 16x16x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 -- outputs 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 2x2x256
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout(p=0.3, inplace=False)
        self.dropout4 = nn.Dropout(p=0.3, inplace=False)
        self.dropout5 = nn.Dropout(p=0.3, inplace=False)
        
        # fully connected
        self.fc5 = nn.Linear(256*2*2, 32)
        self.fc6 = nn.Linear(32, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)       
        x = self.fc6(x)
        return x


class mnist_quant(nn.Module):
    # assuming 32x32x1 input_tensor

    def __init__(self, num_classes=10):
        super(mnist_quant, self).__init__()
        #self.in_channel = 1

        print("Quant MNIST")
        # block 0 -- outputs 16x16x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 -- outputs 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 2x2x256
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout(p=0.3, inplace=False)
        self.dropout4 = nn.Dropout(p=0.3, inplace=False)
        self.dropout5 = nn.Dropout(p=0.3, inplace=False)
        
        # fully connected
        self.fc5 = nn.Linear(256*2*2, 32)
        self.fc6 = nn.Linear(32, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(quant_signed_05(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = F.relu(quant_signed_05(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = F.relu(quant_signed_05(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool3(x)
        x = F.relu(quant_signed_05(self.conv4(x)))
        x = self.dropout4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = F.relu(quant_signed_05(self.fc5(x)))
        x = self.dropout5(x)       
        x = self.fc6(x)
        return x


class VGGnagaCNN_quant(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(VGGnagaCNN_quant, self).__init__()
        # block 1 -- outputs 16x16x64
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 8x8x128
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 4x4x256
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected
        self.fc6 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout()
        self.fc7 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout()
        self.fc8 = nn.Linear(1024, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = quant_signed_1(x)
        x = F.relu(self.conv1_2(x))
        x = quant_signed_1(x)
        x = self.pool1(x)
        x = quant_signed_1(x)
        x = F.relu(self.conv2_1(x))
        x = quant_signed_1(x)
        x = F.relu(self.conv2_2(x))
        x = quant_signed_1(x)
        x = self.pool2(x)
        x = quant_signed_1(x)
        x = F.relu(self.conv3_1(x))
        x = quant_signed_1(x)
        x = F.relu(self.conv3_2(x))
        x = quant_signed_1(x)
        x = F.relu(self.conv3_3(x))
        x = quant_signed_1(x)
        x = F.relu(self.conv3_4(x))
        x = quant_signed_1(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = quant_signed_1(x)
        x = self.dropout1(x)       
        x = F.relu(self.fc7(x))
        x = quant_signed_1(x)
        x = self.dropout2(x)
        x = self.fc8(x)
        return x


class VGGnagaCNN(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(VGGnagaCNN, self).__init__()
        # block 1 -- outputs 16x16x64
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 8x8x128
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 4x4x256
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected
        self.fc6 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout()
        self.fc7 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout()
        self.fc8 = nn.Linear(1024, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)       
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        x = self.fc8(x)
        return x
    

class CNN_627_large_quant(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(CNN_627_large, self).__init__()
        
        # block 0 -- outputs 16x16x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 -- outputs 8x8x128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 4x4x256
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 2x2x512
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout(p=0.3, inplace=False)
        self.dropout4 = nn.Dropout(p=0.3, inplace=False)
        
        # fully connected
        self.fc5 = nn.Linear(2048, 256)
        self.fc6 = nn.Linear(256, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(quant_signed_1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = F.relu(quant_signed_1(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = F.relu(quant_signed_1(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool3(x)
        x = F.relu(quant_signed_1(self.conv4(x)))
        x = self.dropout4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = F.relu(quant_signed_1(self.fc5(x)))
        x = self.dropout4(x)       
        x = self.fc6(x)
        return x


class CNN_627_large(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(CNN_627_large, self).__init__()
        
        # block 0 -- outputs 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 -- outputs 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 2x2x256
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout(p=0.3, inplace=False)
        self.dropout4 = nn.Dropout(p=0.3, inplace=False)
        
        # fully connected
        self.fc5 = nn.Linear(1024, 32)
        self.fc6 = nn.Linear(32, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc5(x))
        x = self.dropout4(x)       
        x = self.fc6(x)
        return x


class CNN_627_small(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(CNN_627_small, self).__init__()
        
        # block 0 -- outputs 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 -- outputs 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 2x2x128
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4 -- outputs 1x1x128
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout(p=0.3, inplace=False)
        self.dropout4 = nn.Dropout(p=0.3, inplace=False)
        self.dropout5 = nn.Dropout(p=0.3, inplace=False)
        self.dropout6 = nn.Dropout(p=0.3, inplace=False)
        
        # fully connected
        self.fc6 = nn.Linear(128, 32)
        self.fc7 = nn.Linear(32, num_classes)
        

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.dropout5(x)
        x = self.pool5(x)


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)       
        x = self.fc7(x)
        return x


class CNN_627(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=10):
        super(CNN_627, self).__init__()
        
        # block 0 -- outputs 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 1 -- outputs 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block 2 -- outputs 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 2x2x128
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)
        self.dropout3 = nn.Dropout(p=0.3, inplace=False)
        self.dropout4 = nn.Dropout(p=0.3, inplace=False)
        
        # fully connected
        self.fc5 = nn.Linear(512, 32)
        self.fc6 = nn.Linear(32, num_classes)
        self.fc7 = nn.Linear(128, 32)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.pool4(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.pool4(x)


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc7(x))
        x = self.dropout4(x)       
        x = self.fc6(x)
        return x


'''
ConvPool-CNN-C CNN structure
'''
class CNNC(nn.Module):
    # assuming 32x32x3 input_tensor

    def __init__(self, num_classes=100):
        super(CNNC, self).__init__()
        # block 1 -- outputs 16x16x64
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(96)
        self.bn1_2 = nn.BatchNorm2d(96)
        self.bn1_3 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2 -- outputs 8x8x128
        self.conv2_1 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(192)
        self.bn2_2 = nn.BatchNorm2d(192)
        self.bn2_3 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3 -- outputs 8x8x256
        self.conv3_1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
        self.conv3_3 = nn.Conv2d(in_channels=192, out_channels=num_classes, kernel_size=1)
        self.bn3_1 = nn.BatchNorm2d(192)
        self.bn3_2 = nn.BatchNorm2d(192)
        self.bn3_3 = nn.BatchNorm2d(num_classes)

        # average_pooling
        self.avgpool = nn.AvgPool2d((8, 8))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

