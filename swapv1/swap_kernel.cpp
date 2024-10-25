#include <torch/torch.h>
#include <vector>
#include <iostream>
at::Tensor swap_cuda_forward(at::Tensor input,const int stride);

at::Tensor swap_cuda_backward(at::Tensor grad_output, const int stride);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor swap_forward(
    at::Tensor input,const int stride) {

    //  std::cout << "this is out2"<< input << std::endl;
      //at::Tensor out = swap_cuda_forward(input,stride);   
      // at::Tensor out = swap_cuda_forward(input,stride);
      // std::cout << "this is out"<< out << std::endl;

  return swap_cuda_forward(input,stride);;
}

at::Tensor swap_backward(
    at::Tensor grad,
    const int stride) 
{
  // CHECK_INPUT(grad);

  return swap_cuda_backward(
    grad,
    stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &swap_forward, "forward");
  m.def("backward", &swap_backward, "backward");
}
