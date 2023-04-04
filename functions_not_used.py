#Library
import sys
import os
import os.path as pth

#!pip install torchplot
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.backends.cudnn as cudnn
import torchplot as plt



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


import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import random
import itertools
import math

from tqdm import tqdm
from numba import jit

import warnings
warnings.simplefilter("ignore")



# Function for integer operation -> Not completed
@jit(nopython=True)
def partial_sum_int_fa(original, bit=5):
    a = original
    bit = bit - 2
    result = np.zeros(a.shape[0])
    for d in range(a.shape[0]):
        partial = a[d].sum()
        if partial > 15:
            partial = 15
        elif partial < -15:
            partial = -15
        else:
            partial = partial
        result[d] = math.trunc(partial)
    return result.sum()


@jit(cache=True)
def conv_custom_fa_int_plot(x_original, w, b):
   
    filt = w.shape[0]
    depth = x_original.shape[0] 
    row = x_original.shape[1] - 2
    col = x_original.shape[2] - 2

    c = np.zeros((filt,row,col))
    one_layer = np.zeros((row,col))
    for f in range(filt):
        for i in range(row):
            for j in range(col):
                r = x_original[:,i:i+3,j:j+3] * w[f]
                one_layer[i,j] = partial_sum_int_fa(r)
        c[f,:,:] = one_layer + b[f]
    return c, one_layer


# Partial_sum_fa_version2 -> Completed & Not used
@jit(nopython=True)
def partial_sum_fa2(original, bit=5):
    bit = bit - 2
    result = np.zeros(original.shape[0])
    for d in range(original.shape[0]):
        result[d] = original[d].sum()
    result = np.clip(result, -1.875, 1.875)
    for d in range(original.shape[0]):
        result[d] = math.trunc(result[d]* (2**bit)) / (2**bit)
    return result.sum()

# Extra Convolution function - Completed
@jit(cache=True)
def conv_custom_fa_plot(x_original, w, b):
   
    filt = w.shape[0]
    depth = x_original.shape[0] 
    row = x_original.shape[1] - 2
    col = x_original.shape[2] - 2

    c = np.zeros((filt,row,col))
    one_layer = np.zeros((row,col))
    before_sum = np.zeros((row,col,filt,3,3))
    for f in range(filt):
        for i in range(row):
            for j in range(col):
                r = x_original[:,i:i+3,j:j+3] * w[f]
                #print(r.shape)
                before_sum[i][j][f] = r
                one_layer[i,j] = partial_sum_fa_conv(r)
        c[f,:,:] = one_layer + b[f]
    return c, one_layer, before_sum

@jit(cache=True)
def conv_custom_fa_test(x_original, w, b):
   
    filt = w.shape[0]
    depth = x_original.shape[0] 
    row = x_original.shape[1] - 2
    col = x_original.shape[2] - 2

    c = np.zeros((filt,row,col))
    one_layer = np.zeros((row,col))
    
    for f in range(filt):
        for i in range(row):
            for j in range(col):
                r = x_original[:,i:i+3,j:j+3] * w[f]
                re = 0
                re1 = partial_sum_fa_conv(r)
                for d1 in range(depth):
                    partial_sum = r[d1].sum()
                    re = re + quant_signed_15_1(partial_sum)
                if (re != re1):
                    print("wrong")
                    print(re, re1)
                    
                one_layer[i,j] = re
        c[f,:,:] = one_layer + b[f]
    return c

@jit
def quant_signed_15_1(original, bit=5):
    bit = bit -2
    original = np.clip(original, -1.875, 1.875)
    original = original * (2**bit)

    result = math.trunc(original)/ (2**bit)
    return result


# Partial Sum Fast - Completed & Not used
# Fixed point quantization , Not dynamically quantized
@jit(cache=True)
def partial_sum_fa_fc(original, bit=5):
    a = original
    bit = bit - 2
    result = np.zeros(a.shape[0])
    for d in range(a.shape[0]):
        partial = a[d].sum()
        if partial > 1.875:
            partial = 1.875
        elif partial < -1.875:
            partial = -1.875
        else:
            partial = partial
        result[d] = math.trunc(partial* (2**bit)) / (2**bit)
    return result.sum()

@jit(nopython=True)
def partial_sum_fa_conv(original, bit=5):
    a = original
    bit = bit - 2
    result = np.zeros(a.shape[0])
    for d in range(a.shape[0]):
        partial = a[d].sum()
        if partial > 1.875:
            partial = 1.875
        elif partial < -1.875:
            partial = -1.875
        else:
            partial = partial
        result[d] = math.trunc(partial* (2**bit)) / (2**bit)
    return result.sum()

# FC & Conv function without point parameters - completed
# Not dynamically quantized
@jit(cache=True)
def fc_fa_non(x_original, w, b):
   
    filt = w.shape[0]
    stage = int(x_original.shape[1]/8)
    c = np.zeros((1,filt))
    for f in range(filt):
        re = 0
        for i in range(stage):
            r = x_original[0,i*8:i*8+8] * w[f,i*8:i*8+8]
            re = re + r.sum()
        c[0,f] = quant_signed_15_np_fc(re + b[f])
    return c

@jit(cache=True)
def conv_custom_fa_non(x_original, w, b):
   
    filt = w.shape[0]
    depth = x_original.shape[0] 
    row = x_original.shape[1] - 2
    col = x_original.shape[2] - 2

    c = np.zeros((filt,row,col))
    one_layer = np.zeros((row,col))
    
    for f in range(filt):
        for i in range(row):
            for j in range(col):
                r = x_original[:,i:i+3,j:j+3] * w[f]
                one_layer[i,j] = r.sum()
        c[f,:,:] = quant_signed_15_np(one_layer + b[f])
    return c


# This is original function that I wrote, but does not match with our verilog model
# It saturates every single value of the input -> I don't remember why I wrote code this way
@jit(cache=True)
def partial_sum_fa_fc_point_wrong(original, bit=5, point=1):
    a = original
    bit = bit - 2  + (point - 1)
    value = 1.875/(2**(point-1))
    
    result = np.zeros(a.shape[0])
    for d in range(a.shape[0]):
        print(d, a.shape[0], a[d])
        partial = a[d].sum()
        if partial > value:
            partial = value
        elif partial < -value:
            partial = -value
        else:
            partial = partial
        result[d] = math.trunc(partial* (2**bit)) / (2**bit)
    return result.sum()


# FC & Conv function with point parameters - completed
@jit(cache=True)
def fc_fa_wo_quant(x_original, w, b, p=1):
   
    filt = w.shape[0]
    stage = int(x_original.shape[1]/8)
    c = np.zeros((1,filt))
    for f in range(filt):
        re = 0
        for i in range(stage):
            r = x_original[0,i*8:i*8+8] * w[f,i*8:i*8+8]
            re = re + partial_sum_fa_fc_point(r, point=p)
        c[0,f] = re + b[f]
    return c

@jit(cache=True)
def conv_custom_fa_wo_quant(x_original, w, b, p=1):
   
    filt = w.shape[0]
    depth = x_original.shape[0] 
    row = x_original.shape[1] - 2
    col = x_original.shape[2] - 2

    c = np.zeros((filt,row,col))
    one_layer = np.zeros((row,col))
    
    for f in range(filt):
        for i in range(row):
            for j in range(col):
                r = x_original[:,i:i+3,j:j+3] * w[f]
                one_layer[i,j] = partial_sum_fa_conv_point(r, point=p)
        c[f,:,:] = one_layer + b[f]
    return c