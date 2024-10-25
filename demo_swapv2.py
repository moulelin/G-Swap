from __future__ import division
from __future__ import print_function

import argparse
import torch

import torch.nn as nn
from torch.autograd import Variable, gradcheck

from swapv2.swapv2c import SwapV2
import datetime


class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),  # 使用LeakyReLU激活函数
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)

        return y


class SwapModule(nn.Module):
    def __init__(self, input_channels, p, original_c):
        super(SwapModule, self).__init__()

        # 通过SELayer获取每个通道的权重

        self.se_layer = SELayer(original_c)

        # 为每个特征点生成四个可更新的参数
        self.conv_offset_x = nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels)
        self.conv_offset_y = nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels)
        self.conv_sigma_x = nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels)
        self.conv_sigma_y = nn.Conv2d(input_channels, input_channels, kernel_size=1, groups=input_channels)

        # Register hooks for the tensors of interest
        # self.hook_exPx = self.conv_offset_x.register_backward_hook(self.hook_fn)
        # self.hook_exPy = self.conv_offset_y.register_backward_hook(self.hook_fn)
        # self.hook_sigmax = self.conv_sigma_x.register_backward_hook(self.hook_fn)
        # self.hook_sigmay = self.conv_sigma_y.register_backward_hook(self.hook_fn)

        self.p = p
        self.swapV2C = SwapV2(p=p)

    # def hook_fn(self, module, input, output):
    #         print(f"Output of {module}:")
    #         print(output)

    def forward(self, x):
        # 通过SELayer获取每个通道的权重
        B, C, W, H = x.shape
        weights = self.se_layer(x)

        # 获取前50%的通道
        _, selected_channels = torch.topk(weights, k=int(C * self.p) if C != 1 else 1, dim=1)
        selected_channels = selected_channels.view(B, -1)  # 将形状更改为 (B, C/2)

        # print(selected_channels)
        # print("*"*20)
        selected_x = torch.stack([x[i, channels, :, :] for i, channels in enumerate(selected_channels)], dim=0)
        # selected_x = x

        exPx = torch.sigmoid(self.conv_offset_x(selected_x)) * (W - 1)  # 缩放到 [0, W]
        exPy = torch.sigmoid(self.conv_offset_y(selected_x)) * (H - 1)  # 缩放到 [0, H]

        sigmax = torch.abs(self.conv_sigma_x(selected_x))
        sigmay = torch.abs(self.conv_sigma_y(selected_x))

        swap_output = self.swapV2C(selected_x, exPx, exPy, sigmax, sigmay)
        # x = x.to(swap_output.device)
        x = torch.cat((x, swap_output), dim=1)

        return x, exPx, exPy, sigmax, sigmay

# 注册钩子函数到中间值（例如，y 的梯度）

starttime = datetime.datetime.now()

B = 5
C = 2
H = 512
W = 512
p = 0.5
input = torch.rand((B, C, H, W))

input = Variable(input.cuda().float(), requires_grad=True)

conv = nn.Conv2d(C, C, kernel_size=1).cuda()
swapOp = SwapModule(int(C * p) if C != 1 else 1, p=p, original_c=C).cuda()

input2 = conv(input)
out, exPx, exPy, sigmax, sigmay = swapOp(input2)

endtime = datetime.datetime.now()
print("fw time: ", endtime - starttime)

sum_out = torch.sum(out)
sum_out.backward(retain_graph=True)  # 添加 retain_graph=True

endtime2 = datetime.datetime.now()
print("bw time: ", endtime2 - endtime)

print("+"*50)
print('*'*20 + '    input')
print(input)
print('*'*20 + '    out')
print(out)
print('*'*20 + '    input.grad')
print(input.grad)

print('*'*20 + '    shift_layer.temporal_position')
