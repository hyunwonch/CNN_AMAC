"""
Define heterogeneous transform domain layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import wht2_cuda
# import wht3_cuda
# import iwht_cuda

_HMT = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float32)
_IHMT = 4 * np.linalg.inv(_HMT)
_IHMT = _IHMT[1:3, :]
_H = torch.from_numpy(_HMT)
_IH = torch.from_numpy(_IHMT)

_HMT2 = _HMT[[1, 0, 2, 3], :]
_H2 = torch.from_numpy(_HMT2)
_IHMT2 = 4 * np.linalg.inv(_HMT2.transpose())
_IHMT2 = _IHMT2[1:3, :]
_IH2 = torch.from_numpy(_IHMT2)

_HMT3 = _HMT[[3, 1, 2, 0], :]
_H3 = torch.from_numpy(_HMT3)
_IHMT3 = 4 * np.linalg.inv(_HMT3.transpose())
_IHMT3 = _IHMT3[1:3, :]
_IH3 = torch.from_numpy(_IHMT3)


'''
NEW HTNN layers with 2 transformswht2_cuda
'''

class WHT2CUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        B = input.size()[0]
        H = input.size()[1]
        W = input.size()[2]
        C = input.size()[3]
        transIm1, transIm2 = wht2_cuda.forward(input, C, B, H, W, 1, 1)
        variables = torch.tensor([B, C, H, W], dtype=torch.int)
        ctx.save_for_backward(*variables)
        return transIm1, transIm2

    @staticmethod
    def backward(ctx, grad_transIm1, grad_transIm2):
        variables = ctx.saved_tensors
        B = variables[0].item()
        C = variables[1].item()
        H = variables[2].item()
        W = variables[3].item()
        d_input = wht2_cuda.backward(grad_transIm1.contiguous(), grad_transIm2.contiguous(), C, B, H, W, 1, 1)
        return d_input

class WHT2Layer(torch.nn.Module):
    def __init__(self):
        super(WHT2Layer, self).__init__()

    def forward(self, input):
        return WHT2CUDAFunction.apply(input)

class IWHT2CUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transIm1, transIm2, weight1, weight2):
        B = transIm1.size()[1]
        nH = transIm1.size()[2]
        nW = transIm1.size()[3]
        C = transIm1.size()[4]
        K1 = weight1.size()[2]
        K2 = weight2.size()[2]
        output1 = iwht_cuda.forward(transIm1, weight1, C, B, nH, nW, K1, 1, 1, 1)
        output2 = iwht_cuda.forward(transIm2, weight2, C, B, nH, nW, K2, 2, 1, 1)
        variables = torch.tensor([B, C, K1, K2], dtype=torch.int)
        ctx.save_for_backward(variables, transIm1, transIm2, weight1, weight2)
        return output1, output2

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        variables, transIm1, transIm2, weight1, weight2 = ctx.saved_tensors
        B = variables[0].item()
        C = variables[1].item()
        K1 = variables[2].item()
        K2 = variables[3].item()
        H = grad_output1.size()[1]
        W = grad_output1.size()[2]
        d_transIm1, d_weight1 = iwht_cuda.backward(grad_output1.contiguous(), transIm1.contiguous(),
                                                   weight1.contiguous(), C, B, H, W, K1, 1, 1, 1)
        d_transIm2, d_weight2 = iwht_cuda.backward(grad_output2.contiguous(), transIm2.contiguous(),
                                                   weight2.contiguous(), C, B, H, W, K2, 2, 1, 1)
        return d_transIm1, d_transIm2, d_weight1, d_weight2

class IWHT2Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(IWHT2Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight1 = torch.nn.Parameter(data=torch.Tensor(16, self.in_channels, self.out_channels // 2),
                                          requires_grad=True)
        self.weight2 = torch.nn.Parameter(
            data=torch.Tensor(16, self.in_channels, self.out_channels // 2 + self.out_channels % 2), requires_grad=True)

        self.bias_flag = bias
        if self.bias_flag:
            self.bias1 = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, self.out_channels // 2), requires_grad=True)
            self.bias2 = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, self.out_channels // 2 + self.out_channels % 2),
                                            requires_grad=True)

    def forward(self, transIm1, transIm2):
        output1, output2 = IWHT2CUDAFunction.apply(transIm1, transIm2, self.weight1, self.weight2)
        if self.bias_flag:
            output1 = output1 + self.bias1
            output2 = output2 + self.bias2
        return output1, output2


'''
NEW HTNN layers with 3 transforms
'''

class WHT3CUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        B = input.size()[0]
        H = input.size()[1]
        W = input.size()[2]
        C = input.size()[3]
        transIm1, transIm2, transIm3 = wht3_cuda.forward(input, C, B, H, W, 1, 1)
        variables = torch.tensor([B, C, H, W], dtype=torch.int)
        ctx.save_for_backward(*variables)
        return transIm1, transIm2, transIm3

    @staticmethod
    def backward(ctx, grad_transIm1, grad_transIm2, grad_transIm3):
        variables = ctx.saved_tensors
        B = variables[0].item()
        C = variables[1].item()
        H = variables[2].item()
        W = variables[3].item()
        d_input = wht3_cuda.backward(grad_transIm1.contiguous(), grad_transIm2.contiguous(), grad_transIm3.contiguous(), C, B, H, W, 1, 1)
        return d_input

class WHT3Layer(torch.nn.Module):
    def __init__(self):
        super(WHT3Layer, self).__init__()

    def forward(self, input):
        return WHT3CUDAFunction.apply(input)

class IWHT3CUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transIm1, transIm2, transIm3, weight1, weight2, weight3):
        B = transIm1.size()[1]
        nH = transIm1.size()[2]
        nW = transIm1.size()[3]
        C = transIm1.size()[4]
        K1 = weight1.size()[2]
        K2 = weight2.size()[2]
        K3 = weight3.size()[2]
        output1 = iwht_cuda.forward(transIm1, weight1, C, B, nH, nW, K1, 1, 1, 1)
        output2 = iwht_cuda.forward(transIm2, weight2, C, B, nH, nW, K2, 2, 1, 1)
        output3 = iwht_cuda.forward(transIm3, weight3, C, B, nH, nW, K3, 3, 1, 1)
        variables = torch.tensor([B, C, K1, K2, K3], dtype=torch.int)
        ctx.save_for_backward(variables, transIm1, transIm2, transIm3, weight1, weight2, weight3)
        return output1, output2, output3

    @staticmethod
    def backward(ctx, grad_output1, grad_output2, grad_output3):
        variables, transIm1, transIm2, transIm3, weight1, weight2, weight3 = ctx.saved_tensors
        B = variables[0].item()
        C = variables[1].item()
        K1 = variables[2].item()
        K2 = variables[3].item()
        K3 = variables[4].item()
        H = grad_output1.size()[1]
        W = grad_output1.size()[2]
        d_transIm1, d_weight1 = iwht_cuda.backward(grad_output1.contiguous(), transIm1.contiguous(),
                                                   weight1.contiguous(), C, B, H, W, K1, 1, 1, 1)
        d_transIm2, d_weight2 = iwht_cuda.backward(grad_output2.contiguous(), transIm2.contiguous(),
                                                   weight2.contiguous(), C, B, H, W, K2, 2, 1, 1)
        d_transIm3, d_weight3 = iwht_cuda.backward(grad_output3.contiguous(), transIm3.contiguous(),
                                                   weight3.contiguous(), C, B, H, W, K3, 3, 1, 1)
        return d_transIm1, d_transIm2, d_transIm3, d_weight1, d_weight2, d_weight3

class IWHT3Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(IWHT3Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight1 = torch.nn.Parameter(data=torch.Tensor(16, self.in_channels, self.out_channels // 3), requires_grad=True)
        self.weight2 = torch.nn.Parameter(data=torch.Tensor(16, self.in_channels, self.out_channels // 3), requires_grad=True)
        self.weight3 = torch.nn.Parameter(data=torch.Tensor(16, self.in_channels, self.out_channels // 3 + self.out_channels % 3), requires_grad=True)

        self.bias_flag = bias
        if self.bias_flag:
            self.bias1 = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, self.out_channels // 3), requires_grad=True)
            self.bias2 = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, self.out_channels // 3), requires_grad=True)
            self.bias3 = torch.nn.Parameter(data=torch.Tensor(1, 1, 1, self.out_channels // 3 + self.out_channels % 3), requires_grad=True)

    def forward(self, transIm1, transIm2, transIm3):
        output1, output2, output3 = IWHT3CUDAFunction.apply(transIm1, transIm2, transIm3, self.weight1, self.weight2, self.weight3)
        if self.bias_flag:
            output1 = output1 + self.bias1
            output2 = output2 + self.bias2
            output3 = output3 + self.bias3
        return output1, output2, output3


'''
OLD HTNN layer with 2 transforms
'''

class MultLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, bias=True):
        super(MultLayer, self).__init__()
        self.stride = stride
        self.kernel = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.padding = torch.nn.ZeroPad2d(padding=1)
        self.weight = torch.nn.Parameter(
            data=torch.Tensor(self.out_channels, self.in_channels, self.kernel, self.kernel), requires_grad=True)

        self.bias_flag = bias
        if self.bias_flag:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, out_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        batch, _, ih, iw = list(x.size())
        bsize = self.kernel - self.stride
        x = self.padding(x)

        patches = x.unfold(3, self.kernel, self.stride).unfold(2, self.kernel, self.stride)
        input_hmt1 = _H.cuda().matmul(patches).matmul(_H.cuda())
        input_hmt2 = _H2.t().cuda().matmul(patches).matmul(_H2.cuda())

        hmult2d = []
        # Transform 1
        for idx in range(0, self.out_channels // 2):
            hmult2d_tmp = input_hmt1 * self.weight[idx, :, :, :].view(1, self.in_channels, 1, 1, self.kernel,
                                                                      self.kernel)  # batch by n_in by n_block by n_block by kernel by kernel tensor
            hmult2d_tmp = hmult2d_tmp.sum(1)  # batch by n_block by n_block by kernel by kernel tensor
            hmult2d_tmp = _IH.cuda().matmul(hmult2d_tmp).matmul(_IH.t().cuda())
            hmult2d.append(hmult2d_tmp)
        # Transform 2
        for idx in range(self.out_channels // 2, self.out_channels):
            hmult2d_tmp = input_hmt2 * self.weight[idx, :, :, :].view(1, self.in_channels, 1, 1, self.kernel,
                                                                      self.kernel)  # batch by n_in by n_block by n_block by kernel by kernel tensor
            hmult2d_tmp = hmult2d_tmp.sum(1)  # batch by n_block by n_block by kernel by kernel tensor
            hmult2d_tmp = _IH2.cuda().matmul(hmult2d_tmp).matmul(_IH2.t().cuda())
            hmult2d.append(hmult2d_tmp)
        hmult = torch.stack(hmult2d, dim=1)  # batch by n_out by n_block by n_block by kernel by kernel tensor

        """ Slower """
        # hmult1 = (input_hmt1.unsqueeze(1)*self.weight[:self.out_channels//2].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        # hmult1 = _IH.cuda().matmul(hmult1).matmul(_IH.t().cuda()) # batch by n_out//2 by n_block by n_block by bsize by bsize tensor
        # hmult2 = (input_hmt2.unsqueeze(1)*self.weight[self.out_channels//2:].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        # hmult2 = _IH2.cuda().matmul(hmult2).matmul(_IH2.t().cuda()) # batch by n_out//2 by n_block by n_block by bsize by bsize tensor
        # hmult = torch.cat([hmult1, hmult2], dim=1) # batch by n_out by n_block by n_block by kernel by kernel tensor

        hmult = hmult.permute(0, 1, 2, 4, 3, 5).reshape(batch, self.out_channels, -1, bsize,
                                                        iw)  # batch by n_out by n_block by bsize by iw tensor
        out = hmult.reshape(batch, self.out_channels, ih, iw)  # batch by n_out by ih by iw tensor

        if self.bias_flag:
            out = out + self.bias

        return out


'''
HTNN layer with 3 transforms
'''

class MultLayer3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, bias=True):
        super(MultLayer3, self).__init__()
        self.stride = stride
        self.kernel = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.padding = torch.nn.ZeroPad2d(padding=1)
        self.weight = torch.nn.Parameter(
            data=torch.Tensor(self.out_channels, self.in_channels, self.kernel, self.kernel), requires_grad=True)

        self.bias_flag = bias
        if self.bias_flag:
            self.bias = torch.nn.Parameter(data=torch.Tensor(1, out_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        batch, _, ih, iw = list(x.size())
        bsize = self.kernel - self.stride
        x = self.padding(x)

        patches = x.unfold(3, self.kernel, self.stride).unfold(2, self.kernel, self.stride)
        input_hmt1 = _H.cuda().matmul(patches).matmul(_H.cuda())
        input_hmt2 = _H2.t().cuda().matmul(patches).matmul(_H2.cuda())
        input_hmt3 = _H3.t().cuda().matmul(patches).matmul(_H3.cuda())

        hmult3d = []
        # Transform 1
        for idx in range(0, self.out_channels // 3):
            hmult3d_tmp = input_hmt1 * self.weight[idx, :, :, :].view(1, self.in_channels, 1, 1, self.kernel,
                                                                      self.kernel)  # batch by n_in by n_block by n_block by kernel by kernel tensor
            hmult3d_tmp = hmult3d_tmp.sum(1)  # batch by n_block by n_block by kernel by kernel tensor
            hmult3d_tmp = _IH.cuda().matmul(hmult3d_tmp).matmul(_IH.t().cuda())
            hmult3d.append(hmult3d_tmp)
        # Transform 2
        for idx in range(self.out_channels // 3, 2 * (self.out_channels // 3)):
            hmult3d_tmp = input_hmt2 * self.weight[idx, :, :, :].view(1, self.in_channels, 1, 1, self.kernel,
                                                                      self.kernel)  # batch by n_in by n_block by n_block by kernel by kernel tensor
            hmult3d_tmp = hmult3d_tmp.sum(1)  # batch by n_block by n_block by kernel by kernel tensor
            hmult3d_tmp = _IH2.cuda().matmul(hmult3d_tmp).matmul(_IH2.t().cuda())
            hmult3d.append(hmult3d_tmp)
        # Transform 3
        for idx in range(2 * (self.out_channels // 3), self.out_channels):
            hmult3d_tmp = input_hmt3 * self.weight[idx, :, :, :].view(1, self.in_channels, 1, 1, self.kernel,
                                                                      self.kernel)  # batch by n_in by n_block by n_block by kernel by kernel tensor
            hmult3d_tmp = hmult3d_tmp.sum(1)  # batch by n_block by n_block by kernel by kernel tensor
            hmult3d_tmp = _IH3.cuda().matmul(hmult3d_tmp).matmul(_IH3.t().cuda())
            hmult3d.append(hmult3d_tmp)

        hmult = torch.stack(hmult3d, dim=1)  # batch by n_out by n_block by n_block by kernel by kernel tensor

        """ Slower """
        # hmult1 = (input_hmt1.unsqueeze(1)*self.weight[:self.out_channels//3].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        # hmult1 = _IH.cuda().matmul(hmult1).matmul(_IH.t().cuda()) # batch by n_out//3 by n_block by n_block by bsize by bsize tensor
        # hmult2 = (input_hmt2.unsqueeze(1)*self.weight[self.out_channels//3:self.out_channels//3*2].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        # hmult2 = _IH2.cuda().matmul(hmult2).matmul(_IH2.t().cuda()) # batch by n_out//3 by n_block by n_block by bsize by bsize tensor
        # hmult3 = (input_hmt3.unsqueeze(1)*self.weight[self.out_channels//3*2:].unsqueeze(0).unsqueeze(3).unsqueeze(4)).sum([2])
        # hmult3 = _IH3.cuda().matmul(hmult3).matmul(_IH3.t().cuda()) # batch by n_out//3 by n_block by n_block by bsize by bsize tensor
        # hmult = torch.cat([hmult1, hmult2, hmult3], dim=1) # batch by n_out by n_block by n_block by kernel by kernel tensor

        hmult = hmult.permute(0, 1, 2, 4, 3, 5).reshape(batch, self.out_channels, -1, bsize,
                                                        iw)  # batch by n_out by n_block by bsize by iw tensor
        out = hmult.reshape(batch, self.out_channels, ih, iw)  # batch by n_out by ih by iw tensor

        if self.bias_flag:
            out = out + self.bias

        return out
