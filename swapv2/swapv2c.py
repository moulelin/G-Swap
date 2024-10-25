from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.nn import Module
import torch
import gswapv2
import numpy as np 
import torch.nn as nn



class Swap_Function(Function):
    
    @staticmethod
    def forward(ctx, input,exPx,exPy,sigmax,sigmay):
        ctx.input = input
        ctx.exPx = exPx
        ctx.exPy = exPy
        ctx.sigmax = sigmax
        ctx.sigmay = sigmay


        # update 
        output = gswapv2.swap_fw(input,exPx,exPy,sigmax,sigmay)
        # print("-"*20)
        # print(output.shape)
        # print("-"*20)
        return output
        
    @staticmethod
    def backward(ctx, output_grad):
        input = ctx.input
        exPx = ctx.exPx
        exPy = ctx.exPy
        sigmax = ctx.sigmax
        sigmay = ctx.sigmay
        output_grad = output_grad.contiguous()
        
        d_output_input, d_output_expX, d_output_expY, d_output_sigmaX, d_output_sigmaY = gswapv2.swap_bw(output_grad, input,exPx,exPy,sigmax,sigmay)
        # print(d_output_input)
        # print("*"*30)
        return d_output_input, None, None, None, None
        


        

class SwapV2(nn.Module):
    # 这个类的作用是可以添加一下可训练的参数
    # 这里可以考虑为卷积核
    def __init__(self, p):
        super(SwapV2, self).__init__()
        self.p = p
       # self._kernel = nn.Parameter(torch.zeros((kernel_size, kernel_size),requires_grad = True,device = "cuda"))
    # @staticmethod
    # def init_weight():
    #     pass
    def forward(self, input,exPx,exPy,sigmax,sigmay):
       return Swap_Function.apply(input,exPx,exPy,sigmax,sigmay)


