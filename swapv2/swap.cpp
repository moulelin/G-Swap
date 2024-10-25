#include <torch/torch.h>
#include "include/utils.h"
torch::Tensor swap_fw(torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay){
    CHECK_INPUT(input);  CHECK_INPUT(exPx); CHECK_INPUT(exPy); CHECK_INPUT(sigmax); CHECK_INPUT(sigmay); 

    return swap_cu_fw_host(input,exPx,exPy,sigmax,sigmay);

}
std::vector<torch::Tensor> swap_bw(torch::Tensor dl_ouput, torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay){
    CHECK_INPUT(input);  CHECK_INPUT(exPx); CHECK_INPUT(exPy); CHECK_INPUT(sigmax); CHECK_INPUT(sigmay); 

    return swap_cu_bw_host(dl_ouput,input,exPx,exPy,sigmax,sigmay);

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("swap_fw", &swap_fw);
    m.def("swap_bw", &swap_bw);
}