from __future__ import division
from __future__ import print_function

import argparse
import torch

import torch.nn as nn
from torch.autograd import Variable, gradcheck

from swap import Swap_Function,SwapNN


swap_layer = SwapNN(kernel_size = 3,stride=1,dilate = 1)
data = torch.arange(0,4*4*4,1).reshape(1,4,4,4)
input = Variable(data.cuda().float(), requires_grad=True) 
out = swap_layer(input)
# print(out)
sum_out = torch.sum(out)
sum_out.backward()
print("+"*50)
print('*'*20 + '    input')
print(input)
print('*'*20 + '    out')
print(out)
print('*'*20 + '    input.grad')
# print(input.grad)
# print('*'*20 + '    shift_layer.temporal_position')
