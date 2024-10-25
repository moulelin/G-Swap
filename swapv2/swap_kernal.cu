#include <cooperative_groups.h>
#include <torch/torch.h>
#include <iostream>
// #include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <cmath>

namespace cg = cooperative_groups;
using namespace std;
template<typename scalar_t>
__device__ float bilinear_interpolation(scalar_t* point4, float row, float col);
__device__ float linear_calculate(float origin, float exp, float gaussian_probability);
template<typename scalar_t>
__device__ float gaussian(scalar_t *distance_bi, scalar_t *sigma);
template<typename scalar_t>
__global__ void swap_fw_cu(
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> exPx, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> exPy, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> sigmax, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> sigmay, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output,
const int C,
const int H,
const int W) {
    const int batch = blockIdx.z/C;
    const int channel = blockIdx.z%C;
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    const int x = threadIdx.x; // col
    const int y = threadIdx.y; // row
    if (row < H&&col<W){
    __shared__ scalar_t data[16][16];
    __shared__ scalar_t expSx[16][16];
    __shared__ scalar_t expSy[16][16];
    __shared__ scalar_t points[16*16][4];
    __shared__ scalar_t sigmaS[16*16][2];

    __shared__ scalar_t distance[16*16][2];

    sigmaS[y*blockDim.x + x][0] = sigmax[batch][channel][row][col];
    sigmaS[y*blockDim.x + x][1] = sigmay[batch][channel][row][col];

    data[y][x] = input[batch][channel][row][col];

    expSx[y][x] = exPx[batch][channel][row][col]; // row
    expSy[y][x] = exPy[batch][channel][row][col]; // col


    int up_row = int(std::ceil(exPx[batch][channel][row][col])); 
    int up_col = int(std::ceil(exPy[batch][channel][row][col])); 
    if(up_row < H && up_col < W){
    int down_row  = up_row-1;
    int down_col  = up_col-1;

    points[y*blockDim.x + x][0] = input[batch][channel][down_row][down_col]; //左上
    points[y*blockDim.x + x][1] = input[batch][channel][down_row][up_col]; //右上
    points[y*blockDim.x + x][2] = input[batch][channel][up_row][down_col]; //左下
    points[y*blockDim.x + x][3] = input[batch][channel][up_row][up_col]; //右下

    distance[y*blockDim.x + x][0] = abs(row - exPx[batch][channel][row][col]);
    distance[y*blockDim.x + x][1] = abs(col - exPy[batch][channel][row][col]);
    __syncthreads();

    float bilinear_value = bilinear_interpolation(points[y*blockDim.x + x], expSx[y][x], expSy[y][x]);
    float gaussian_probability = gaussian(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x]);
    float new_value = linear_calculate(data[y][x], bilinear_value, gaussian_probability);
    // calcu
    atomicAdd(&output[batch][channel][row][col], new_value);
    }}
}


__device__ float linear_calculate(float origin, float exp, float gaussian_probability){

    float final_result = origin + (gaussian_probability*exp);
    return final_result;
}



template<typename scalar_t>
__device__ float bilinear_interpolation(scalar_t* point4, float row, float col){

     // 获取整数部分和小数部分
    int row_floor = static_cast<int>(std::floor(row));
    int col_floor = static_cast<int>(std::floor(col));
    float row_fractional = row - row_floor;
    float col_fractional = col - col_floor;

    float top_left = point4[0];
    float top_right = point4[1];
    float bottom_left = point4[2];
    float bottom_right = point4[3];

    float row_fractional_d = 1 - row_fractional;
    float col_fractional_d = 1 - col_fractional;


    float result = top_left*row_fractional_d*col_fractional_d + top_right*col_fractional*row_fractional_d + 
    bottom_left*col_fractional_d*row_fractional + bottom_right*col_fractional*row_fractional;
    return result;
}

template<typename scalar_t>
__device__ float gaussian(scalar_t *distance_bi, scalar_t *sigma){

    float gaussianX = expf(-(distance_bi[0]) * (distance_bi[0]) / (2.0f * sigma[0] * sigma[0]))/ (sqrtf(2.0f * M_PI) * sigma[0]);
    float gaussianY = expf(-(distance_bi[1]) * (distance_bi[1]) / (2.0f * sigma[1] * sigma[1]))/ (sqrtf(2.0f * M_PI) * sigma[1]);
    float result = gaussianX*gaussianY;
    return result;
}


template<typename scalar_t>
__device__ float gaussian_singal(scalar_t *distance_bi, scalar_t *sigma, const int index){

    float gaussian = expf(-(distance_bi[index]) * (distance_bi[index]) / (2.0f * sigma[index] * sigma[index]))/ (sqrtf(2.0f * M_PI) * sigma[index]);
    
    return gaussian;
}

template<typename scalar_t>
__device__ float gaussian_gradient(scalar_t *distance_bi, scalar_t *sigma, const int index){

    float x_minus_a = distance_bi[index];
    float sigma_squared = sigma[index] * sigma[index];
    float derivative = -x_minus_a / (sqrtf(2.0f * M_PI) * pow(sigma[index], 3.0f)) * expf(-x_minus_a * x_minus_a / (2.0f * sigma_squared));
    return derivative;
}

template<typename scalar_t>
__device__ float sigma_gradient(scalar_t *distance_bi, scalar_t *sigma, const int index){

    float x_minus_a = distance_bi[index];
    float sigma_squared = sigma[index] * sigma[index];
    float derivative = -x_minus_a*x_minus_a / (sqrtf(2.0f * M_PI) * pow(sigma[index], 5.0f)) * expf(-x_minus_a * x_minus_a / (2.0f * sigma_squared));
    return derivative;
}



template<typename scalar_t>
__global__ void swap_bw_cu(
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> dl_ouput,
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> exPx, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> exPy, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> sigmax, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> sigmay, 
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_input,
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_expX,
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_expY,
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_sigmaX,
 torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output_sigmaY,
const int C,
const int H,
const int W) {
    const int batch = blockIdx.z/C;
    const int channel = blockIdx.z%C;
    const int col = blockIdx.x*blockDim.x + threadIdx.x;
    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    const int x = threadIdx.x; // col
    const int y = threadIdx.y; // row
    if (row < H&&col<W){

    __shared__ scalar_t output_grad_shared[16][16];
    __shared__ scalar_t distance[16*16][2];
    __shared__ scalar_t sigmaS[16*16][2];
    __shared__ scalar_t expSx[16][16];
    __shared__ scalar_t points[16*16][4];
    __shared__ scalar_t expSy[16][16];
    
    output_grad_shared[y][x] = dl_ouput[batch][channel][row][col];
    distance[y*blockDim.x + x][0] = abs(row - exPx[batch][channel][row][col]);
    distance[y*blockDim.x + x][1] = abs(col - exPy[batch][channel][row][col]);
    //
    sigmaS[y*blockDim.x + x][0] = sigmax[batch][channel][row][col];
    sigmaS[y*blockDim.x + x][1] = sigmay[batch][channel][row][col];
    // points
    int up_row = int(std::ceil(exPx[batch][channel][row][col])); 
    int up_col = int(std::ceil(exPy[batch][channel][row][col])); 
   
    if(up_row < H && up_col < W){
    int down_row  = up_row-1;
    int down_col  = up_col-1;
    points[y*blockDim.x + x][0] = input[batch][channel][down_row][down_col]; //左上
    points[y*blockDim.x + x][1] = input[batch][channel][down_row][up_col]; //右上
    points[y*blockDim.x + x][2] = input[batch][channel][up_row][down_col]; //左下
    points[y*blockDim.x + x][3] = input[batch][channel][up_row][up_col]; //右下
    // expX
    expSx[y][x] = exPx[batch][channel][row][col]; // row
    expSy[y][x] = exPy[batch][channel][row][col]; // col

    __syncthreads();
    float gaussian_probability = gaussian(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x]);
    // bilinear_interpolation_gradient for input
    float row_exp = expSx[y][x];
    float col_exp = expSy[y][x];

    

    int row_floor = static_cast<int>(std::floor(row_exp));
    int col_floor = static_cast<int>(std::floor(col_exp));
    float row_fractional = row_exp - row_floor;
    float col_fractional = col_exp - col_floor;


    float row_fractional_d = 1 - row_fractional;
    float col_fractional_d = 1 - col_fractional;

    float top_left_grad = row_fractional_d * col_fractional_d * (gaussian_probability);
    float top_right_grad = col_fractional * row_fractional_d * (gaussian_probability);
    float bottom_left_grad = col_fractional_d * row_fractional * (gaussian_probability);
    float bottom_right_grad = col_fractional * row_fractional * (gaussian_probability);
 
    // 双线性插值四个角点的梯度
    atomicAdd(&d_output_input[batch][channel][down_row][down_col], top_left_grad * output_grad_shared[y][x]); 
    atomicAdd(&d_output_input[batch][channel][down_row][up_col], top_right_grad * output_grad_shared[y][x]);
    atomicAdd(&d_output_input[batch][channel][up_row][down_col], bottom_left_grad * output_grad_shared[y][x]);
    atomicAdd(&d_output_input[batch][channel][up_row][up_col], bottom_right_grad * output_grad_shared[y][x]);
    // input的梯度
    atomicAdd(&d_output_input[batch][channel][row][col], output_grad_shared[y][x]); 
    }
    // expX 

   // expX 梯度
    // float result_bilinear_gradX =
    // (input[batch][channel][down_row][down_col] * (-col_fractional_d) +
    // input[batch][channel][down_row][up_col] * col_fractional  +
    // input[batch][channel][up_row][down_col] * (-col_fractional_d) +
    // input[batch][channel][up_row][up_col] * col_fractional) * gaussian_probability;

    // float gaussian_Y = gaussian_singal(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x],1);

    // float bilinear_value = bilinear_interpolation(points[y*blockDim.x + x], expSx[y][x], expSy[y][x]);

    // float result_gaussian_gradX = gaussian_gradient(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x],0) * bilinear_value * gaussian_Y;
    // float expx_grad = result_bilinear_gradX + result_gaussian_gradX;
    // atomicAdd(&d_output_expX[batch][channel][row][col], expx_grad*output_grad_shared[y][x]); 

    // // expY 梯度
    
    // float result_bilinear_gradY =
    // (input[batch][channel][down_row][down_col] * (-row_fractional_d) +
    // input[batch][channel][down_row][up_col] * row_fractional_d  +
    // input[batch][channel][up_row][down_col] * (-row_fractional) +
    // input[batch][channel][up_row][up_col] * row_fractional) * gaussian_probability;
    // float gaussian_X = gaussian_singal(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x], 0);


    // float result_gaussian_gradY = gaussian_gradient(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x],1) * bilinear_value * gaussian_X;
    // float expy_grad = result_bilinear_gradY + result_gaussian_gradY;
    // atomicAdd(&d_output_expY[batch][channel][row][col], expy_grad*output_grad_shared[y][x]); 


    // // sigmaX 梯度
    // float sigmaxG = sigma_gradient(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x], 0);
    // float sigmax_gradient = gaussian_Y * bilinear_value * sigmaxG;
    // atomicAdd(&d_output_sigmaX[batch][channel][row][col], sigmax_gradient*output_grad_shared[y][x]); 

    // //sigmaY 梯度

    // float sigmayG = sigma_gradient(distance[y*blockDim.x + x], sigmaS[y*blockDim.x + x], 1);
    // float sigmay_gradient = gaussian_X * bilinear_value * sigmayG;
    // atomicAdd(&d_output_sigmaY[batch][channel][row][col], sigmay_gradient*output_grad_shared[y][x]); 
    }
}


template<typename scalar_t>
__device__ void bilinear_interpolation_gradient(float row, float col, float gaussian_prob, float* doutput_input){

     // 获取整数部分和小数部分
    int row_floor = static_cast<int>(std::floor(row));
    int col_floor = static_cast<int>(std::floor(col));
    float row_fractional = row - row_floor;
    float col_fractional = col - col_floor;


    float row_fractional_d = 1 - row_fractional;
    float col_fractional_d = 1 - col_fractional;

    float top_left_grad = row_fractional_d * col_fractional_d * (1.0f - gaussian_prob);
    float top_right_grad = col_fractional * row_fractional_d * (1.0f - gaussian_prob);
    float bottom_left_grad = col_fractional_d * row_fractional * (1.0f - gaussian_prob);
    float bottom_right_grad = col_fractional * row_fractional * (1.0f - gaussian_prob);
    

}







torch::Tensor swap_cu_fw_host(torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay) {
    
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const dim3 block(16,16);
    const dim3 grid((W + 16 - 1)/16, (H + 16 - 1)/16, B*C);
    torch::Tensor output = torch::zeros({B,C,H,W}, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "swap_fw_cu",([&]{
        swap_fw_cu<scalar_t><<<grid, block>>>(
        input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(), 
        exPx.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        exPy.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        sigmax.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        sigmay.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        C,H,W);
    }
    ));
    cudaDeviceSynchronize(); 
    return output;
  
}


std::vector<torch::Tensor> swap_cu_bw_host(torch::Tensor dl_ouput, torch::Tensor input, torch::Tensor exPx, torch::Tensor exPy, torch::Tensor sigmax, torch::Tensor sigmay) {
    
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const dim3 block(16,16);
    const dim3 grid((W + 16 - 1)/16, (H + 16 - 1)/16, B*C);

    torch::Tensor d_output_input = torch::zeros({B,C,H,W}, input.options());

    torch::Tensor d_output_expX = torch::zeros({exPx.sizes()}, exPx.options());
    torch::Tensor d_output_expY = torch::zeros({exPy.sizes()}, exPy.options());

    torch::Tensor d_output_sigmaX = torch::zeros({sigmax.sizes()}, sigmax.options());
    torch::Tensor d_output_sigmaY = torch::zeros({sigmay.sizes()}, sigmax.options());

    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "swap_bw_cu",([&]{
        swap_bw_cu<scalar_t><<<grid, block>>>(
        dl_ouput.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(), 
        input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(), 
        exPx.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        exPy.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        sigmax.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        sigmay.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        d_output_input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        d_output_expX.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        d_output_expY.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        d_output_sigmaX.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        d_output_sigmaY.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
        C,H,W);
    }
    ));
    cudaDeviceSynchronize(); 
    std::vector<torch::Tensor> outputs = {d_output_input, d_output_expX, d_output_expY, d_output_sigmaX, d_output_sigmaY};
    return outputs;
  
}
