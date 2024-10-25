//
// Created by Administrator on 2023/11/14.
//

#ifndef TORCH_CUDA_CPP_UTILS_H
#define TORCH_CUDA_CPP_UTILS_H

#endif //TORCH_CUDA_CPP_UTILS_H

#include <torch/extension.h>
// #include <torch/torch.h>

// 下面的define是必须要添加的，作用类似于python的assert
// CHECK_CUDA 检查变量是否为一个gpu的tensor
// CHECK_CONTIGUOUS 检测每一个tensor在内存上是否是连续的
// CHECK_INPUT 表示检测上面两个函数
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void swap_fw_cu(torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay,const int C,const int H,const int W);
void swap_bw_cu(torch::Tensor dl_ouput, torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay,torch::Tensor d_output_input, torch::Tensor d_output_expX,torch::Tensor d_output_expY,torch::Tensor d_output_sigmaX,torch::Tensor d_output_sigmaY,const int C,const int H,const int W);



torch::Tensor swap_cu_fw_host(torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay);
std::vector<torch::Tensor> swap_cu_bw_host(torch::Tensor dl_ouput, torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay);


torch::Tensor swap_fw(torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay);
std::vector<torch::Tensor> swap_bw(torch::Tensor dl_ouput, torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay);

