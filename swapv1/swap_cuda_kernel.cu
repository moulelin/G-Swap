// #pragma once
#ifndef CUDA_swap
#define CUDA_swap
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <iostream>



//  __global__ void shiftnet_(){
//     int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y*blockDim*x + threadIdx.x;

//     w_tile = blockIdx.x / 3; //确定是第几个分割小方块的w
//     h_tile = blockIdx.x % 3; // 确定是第几个分割小方块的h

//     w_idx = w_tile*3 + threadIdx.y // 一页的w
//     h_idx = h_tile*3 + threadIdx.x // 一页的h

//     idx_position_in_one_frame = w_idx*h + h_idx // 一页一维存储的位置

//     channel_idx = blockIdx.y >> 5 + threadIdx.z // 第几个channel，不确定batch

//     batch_idx = blockIdx.z
//     idx_real_input = (batch * channel + channel_idx)*(w*b) + idx_position_in_one_frame;
// // idx_real_input 是把数据看作一维，找到现在（各种索引）对应的位置。
//     cache[tid] = input[idx_real_input];
//     // 比如，tid = 2,2,10，假设第0个batch，第0个tile_channel,第（1，2）个小方块
//     // 这里存储的数据就是：第10个batch中，第（5，8）中的元素  
//     // 同理，2，1，10 就是 第10个batch中，第（5，7）中的元素（前一个元素）
//      // 同理，0，0，11 就是 第11个batch中，第（3，6）中的元素 （后一个元素）

//     __syncthreads();

// }
// template <typename type_c>
namespace {
 template <typename type_c>
 __global__ void swap_cuda_forward_kernel(
     type_c* __restrict__ input,type_c* out_clone, int stride, int width, int high, int channel, int batch, int offset
    ){
    const unsigned FULL_MASK = 0xfffffffff;
    // const int width = input.size(2);
    // const int high = input.size(3);
    const int channel_half = channel/2;
    // const int batch = input.size(0);
    //  __shared__ int mem;
    //   mem = width*high*channel;
    extern __shared__ float cache[];
    //__shared__ float cache[3][3][32];

    // tile_one_frame = (w*h+9-1)/9 //把一张照片按3*3切成小正方形状
    // tile_channel = (channel+32-1)/32 // 把通道按照每个组28个通道来切
    // batch = batch // 不变
    // const dim3 grim_size(tile_one_frame, tile_channel, batch);
    // const dim3 block_size(3,3,32); //每3*3大小的小块，一共28个通道组，作为一个block大小
    
  //  int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y*blockDim*x + threadIdx.x;
    int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
//  tid 是一个block中的线程0 - 3*3*32

    int w_tile = blockIdx.x >> 1; //确定是第几个分割小方块的w，行
    int h_tile = blockIdx.x & 1; // 确定是第几个分割小方块的h，列


    int w_idx = (w_tile<<2) + threadIdx.y; // 一页的w，行
    int h_idx = (h_tile<<2) + threadIdx.x; // 一页的h
    
    // printf("w_idx h_idx is %d, %d\n", w_idx,h_idx);
    
    int idx_position_in_one_frame = w_idx*high + h_idx; // 一页一维存储的位置，一个batch（一页整中）的第几行第几列
    // printf("idx_position_in_one_frame is %d \n", idx_position_in_one_frame);
    int channel_idx = (blockIdx.y<<5) + threadIdx.z; // 第几个channel，不确定batch,5是32是意思

    int batch_idx = blockIdx.z;
    int idx_real_input = (batch_idx * channel + channel_idx)*(width*high) + idx_position_in_one_frame;
// idx_real_input 是把数据看作一维，找到现在（各种索引）对应的位置。
    // cache[tid] = input[idx_real_input];
    //cache[blockIdx.x][blockIdx.y][blockIdx.z] = input[idx_real_input];
    cache[tid] = input[idx_real_input];
   // printf("idx_real_input is %d\n",idx_real_input);
    //printf("batch_idx : %d,channel : %d,channel_idx : %d,width : %d,high : %d,idx_position_in_one_frame : %d \n",batch_idx,channel,channel_idx,width,high,idx_position_in_one_frame);
    //tid 是一个block中的id, 这里存储的是32个通道，4*4大小的一个小块的数据
    // 比如，tid = 2,2,10，假设第0个batch，第0个tile_channel,第（1，2）个小方块
    // 这里存储的数据就是：第10个batch中，第（5，8）中的元素  
    // 同理，2，1，10 就是 第10个batch中，第（5，7）中的元素（前一个元素）
     // 同理，0，0，11 就是 第11个batch中，第（3，6）中的元素 （后一个元素）
    //   printf("thread %d,%d\n", tid, idx_real_input);
    __syncthreads();

    // 考虑每个线程走到这里
    // 针对每个线程来处理对应操作
   // int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y*blockDim*x + threadIdx.x;
//  tid 是一个block中的线程0 - 4*4*32
// 使用异或操作
   // float real_value = cache[blockIdx.x][blockIdx.y][blockIdx.z];
   float real_value = cache[tid];

//     __shared__ int offset;
//     offset = 5;
   if ((threadIdx.z & 1) == 0 && channel_idx >= channel_half){
        // __shfl_down_sync(FULL_MASK, real_value, tid, 16);
        // printf("swap operation runs well");
    out_clone[idx_real_input] =  __shfl_xor_sync(FULL_MASK, real_value, offset, 16);
    }
   //out_clone[idx_real_input] = cache[tid^5];
   // printf("thread z %d \n", threadIdx.z);

   // out_clone[1] = 9999;
}


template <typename type_c>
__global__ void swap_cuda_backward_kernel(
    const type_c* __restrict__ input,
    type_c* out_clone,
    const int stride,const int width,const int high,const int channel,const int batch, const int offset

    ){
    const unsigned FULL_MASK = 0xfffffffff;
    // const int width = input.size(2);
    // const int high = input.size(3);
    const int channel_half = channel/2;
    // const int batch = input.size(0);
    extern __shared__ float cache[];
    //__shared__ float cache[3][3][32];

    // tile_one_frame = (w*h+9-1)/9 //把一张照片按3*3切成小正方形状
    // tile_channel = (channel+32-1)/32 // 把通道按照每个组28个通道来切
    // batch = batch // 不变
    // const dim3 grim_size(tile_one_frame, tile_channel, batch);
    // const dim3 block_size(3,3,32); //每3*3大小的小块，一共28个通道组，作为一个block大小
    
  //  int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y*blockDim*x + threadIdx.x;
    int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
//  tid 是一个block中的线程0 - 3*3*32
     int w_tile = blockIdx.x >> 1; //确定是第几个分割小方块的w，行
    int h_tile = blockIdx.x & 1; // 确定是第几个分割小方块的h，列


    int w_idx = (w_tile<<2) + threadIdx.y; // 一页的w，行
    int h_idx = (h_tile<<2) + threadIdx.x; // 一页的h

    int idx_position_in_one_frame = w_idx*high + h_idx; // 一页一维存储的位置

    int channel_idx = (blockIdx.y<<5) + threadIdx.z; // 第几个channel，不确定batch

    int batch_idx = blockIdx.z;
    int idx_real_input = (batch_idx * channel + channel_idx)*(width*high) + idx_position_in_one_frame;
// idx_real_input 是把数据看作一维，找到现在（各种索引）对应的位置。
    // cache[tid] = input[idx_real_input];
    //cache[blockIdx.x][blockIdx.y][blockIdx.z] = input[idx_real_input];
    cache[tid] = input[idx_real_input];
    // 比如，tid = 2,2,10，假设第0个batch，第0个tile_channel,第（1，2）个小方块
    // 这里存储的数据就是：第10个batch中，第（5，8）中的元素  
    // 同理，2，1，10 就是 第10个batch中，第（5，7）中的元素（前一个元素）
     // 同理，0，0，11 就是 第11个batch中，第（3，6）中的元素 （后一个元素）
     
    __syncthreads();
    // 考虑每个线程走到这里
    // 针对每个线程来处理对应操作
   // int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y*blockDim*x + threadIdx.x;
//  tid 是一个block中的线程0 - 3*3*32
// 使用异或操作
   // float real_value = cache[blockIdx.x][blockIdx.y][blockIdx.z];
    float real_value = cache[tid];

    // __shared__ int offset; //位于寄存器
    // offset = 5;
    if ((threadIdx.z & 1) == 0 && channel_idx >= channel_half){
        //走到这边的线程，返回 该线程^offset的值
    out_clone[idx_real_input] =  __shfl_xor_sync(FULL_MASK, real_value, offset, 16);
     //   printf("swap backward runs well");
    }
    //printf("thread z %d \n", threadIdx.z);
    
    }

}
// End of namespace




at::Tensor swap_cuda_forward(
at::Tensor input, 
const int stride
){
const int width = input.size(2);
const int high = input.size(3);
const int channel = input.size(1);
const int batch = input.size(0);
const int tile_one_frame = (width*high+16-1)/16; //把一张照片按4*4切成小正方形状
// printf("the tile_one_frame is %d\n", tile_one_frame);
// printf("the width is %d, the high is %d\n", width, high);
const int tile_channel = (channel+32-1)/32; // 把通道按照每个组28个通道来切
// printf("the tile_channel is %d\n", tile_channel);
// const int batch = batch; // 不变
const dim3 grim_size(tile_one_frame, tile_channel, batch);
const dim3 block_size(width,high,channel); //每4*4大小的小块，一共32个通道组，作为一个block大小
auto out_clone = input.clone();
const int offset = 5;
// AT_DISPATCH_FLOATING_TYPES(input.type(), "forward", 
// ([&] {swap_cuda_forward<at::Tensor><<<grim_size, block_size, 288>>>(input.data<at::Tensor>(), stride)}));
// printf(input);
AT_DISPATCH_FLOATING_TYPES(input.type(), "forward", ([&] {
    swap_cuda_forward_kernel<float><<<grim_size, block_size,sizeof(float)*width*high*(channel+1)>>>(
      input.data<float>(),out_clone.data<float>(), stride,width,high,channel,batch,offset);
  }));
   return out_clone;
}


at::Tensor swap_cuda_backward(
    at::Tensor input, const int stride
){
const int width = input.size(2);
const int high = input.size(3);
const int channel = input.size(1);
const int batch = input.size(0);
const int tile_one_frame = (width*high+16-1)/16; //把一张照片按3*3切成小正方形状
const int tile_channel = (channel+32-1)/32; // 把通道按照每个组28个通道来切
// const int batch = batch // 不变
const dim3 grim_size(tile_one_frame, tile_channel, batch);
const dim3 block_size(width,high,channel); //每4*4大小的小块，一共32个通道组，作为一个block大小
auto grad_clone = input.clone();

const int offset = 5;
// AT_DISPATCH_FLOATING_TYPES(
//     input.type(), "forward", ([&] {
//         swap_cuda_backward<at::Tensor><<<grim_size, block_size, 288>>>(
//     input.data<at::Tensor>(), stride)}));
AT_DISPATCH_FLOATING_TYPES(input.type(), "backward", ([&] {
    swap_cuda_backward_kernel<float><<<grim_size, block_size, sizeof(float)*width*high*(channel)>>>(
      input.data<float>(),grad_clone.data<float>(), stride,width,high,channel,batch,offset);
}));

// shiftnet_2<<<grim_size, block_size, 288>>>()
   return grad_clone;
}
#endif
