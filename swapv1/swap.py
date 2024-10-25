from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.nn import Module
import torch
import swap_kernel
import numpy as np 
import torch.nn as nn

class Swap_Function(Function):
    
    @staticmethod
    def forward(ctx, input,kernel_size, dilate, stride = 1):
        ctx.input = input
        ctx.stride = stride
        # update 
        output = swap_kernel.forward(input, stride)
        # print("-"*20)
        # print(output)
        # print("-"*20)
        return output
        
    @staticmethod
    def backward(ctx, output_grad):
        output_grad = output_grad.contiguous()
        swap_kernel.backward(output_grad, ctx.stride)
        # input = ctx.input
        return output_grad, None, None, None
        

class Swap(nn.Module):
    # 这个类的作用是可以添加一下可训练的参数
    # 这里可以考虑为卷积核
    def __init__(self, kernel_size, stride, dilate):
        super(Swap, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._dilate = dilate
       # self._kernel = nn.Parameter(torch.zeros((kernel_size, kernel_size),requires_grad = True,device = "cuda"))
    # @staticmethod
    # def init_weight():
    #     pass
    def forward(self, x):
       return Swap_Function.apply(x,self._kernel_size, self._dilate, self._stride)
