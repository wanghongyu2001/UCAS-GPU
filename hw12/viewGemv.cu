#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 
using namespace std;
const int inputRowSize = 84;
const int outputRowSize =10;




void init_ij(vector<float>& A, int n, int m)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < m; k++)
            A[j * m + k] = k + j;
    }
}

void init_one(vector<float>& A, int n, int m)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < m; k++)
            A[j * m + k] = 1;
    }
}

void init_zero(vector<float>& A, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            A[i * n + j] = 0;
        }
}

void GEMM(vector<float>& input, int inputRowSize, int inputColSize, \
    vector<float>& kernel, int kernelRowSize, int kernelColSize, \
    vector<float>& output, int outputRowSize, int outputColSize)
{
    for (int i = 0; i < outputRowSize; i++)
    {
        for (int j = 0; j < outputColSize; j++)
        {

            double tmp = 0;
            for (int k = 0; k < inputColSize; k++)
                tmp += input[i * inputColSize + k] * kernel[k * kernelColSize + j];
            output[i * outputColSize + j] = tmp;
        }
    }
}


#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

// if N>= 128
__global__ void reluGemv(
    float* __restrict__ A,
    float* __restrict__ ABias,
    float* __restrict__ x,
    float* __restrict__ y,
    const int M,
    const int N) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if (current_row < M) {
        float res = 0;
        int kIteration = (N / warp_size) / 4;
        if (tx == 0) printf("iter : %d, N %d\n", kIteration, N);
        if (kIteration == 0) kIteration = 1;
        A = &A[current_row * N];
#pragma unroll
        for (int i = 0; i < kIteration; i++) {
            int current_col_vec = (i * warp_size + laneId);
            // printf("current_col_vec = %d\n", current_col_vec);
            if (current_col_vec * 4 >= N) continue;
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0)
            res += ABias[current_row];
        if (res >= 0)
            y[current_row] = res;
        else
            y[current_row] = 0;


    }
}


void reluSPMV(std::vector<float> input, int inputRowSize, \
    std::vector<float> kernel, int kernelRowSize, int kernelColSize, \
    std::vector<float> kernelBias,\
    std::vector<float>& output, int outputRowSize)
{

    float* d_output, *d_kernel, *d_input, *d_kernelBias;
    int inputSize = sizeof(float) * inputRowSize, outputSize = sizeof(float) * outputRowSize;
    int kernelSize = sizeof(float) * kernelRowSize * kernelColSize, kernelBiasSize = kernelRowSize * sizeof(float);
//malloc
    checkCudaErrors(cudaMalloc(&d_output, outputSize));
    checkCudaErrors(cudaMalloc(&d_kernel,kernelSize));
    checkCudaErrors(cudaMalloc(&d_input, inputSize));
    checkCudaErrors(cudaMalloc(&d_kernelBias, kernelBiasSize));

    //memcpy H2D
    checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias, kernelBias.data(), kernelBiasSize, cudaMemcpyHostToDevice));

    //call cudakernel M row, N col
    dim3 dimGrid((kernelRowSize + 3) / 4);
    dim3 dimBlock(32, 4);
    reluGemv<< < dimGrid, dimBlock >> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize);

    //memcpy D2H
    checkCudaErrors(cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    //cudaFree
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_kernelBias);

    
    
#if 0
    for (int i = 0; i < outputRowSize; i++)
    {
            double tmp = 0;
            for (int k = 0; k < inputRowSize; k++)
                tmp += kernel[i * kernelColSize + k] * input[k];
            tmp += kernelBias[i];
            if (tmp >= 0) output[i] = tmp;
            else output[i] = 0;
            
    }
#endif
}
void reluSPMVCheck(std::vector<float> input, int inputRowSize, \
    std::vector<float> kernel, int kernelRowSize, int kernelColSize, \
    std::vector<float> kernelBias,\
    std::vector<float>& output, int outputRowSize)
{

    for (int i = 0; i < outputRowSize; i++)
    {
            double tmp = 0;
            for (int k = 0; k < inputRowSize; k++)
                tmp += kernel[i * kernelColSize + k] * input[k];
            tmp += kernelBias[i];
            if (tmp >= 0) output[i] = tmp;
            else output[i] = 0;
    }
}
void reluGEMM(vector<float>& input, int inputRowSize, int inputColSize, \
    vector<float>& kernel, int kernelRowSize, int kernelColSize, \
    vector<float>& output, int outputRowSize, int outputColSize)
{
    for (int i = 0; i < outputRowSize; i++)
    {
        for (int j = 0; j < outputColSize; j++)
        {

            double tmp = 0;
            for (int k = 0; k < inputColSize; k++)
                tmp += input[i * inputColSize + k] * kernel[k * kernelColSize + j];
            if (tmp >= 0) output[i * outputColSize + j] = tmp;
            else output[i * outputColSize + j] = 0;
        }
    }
}

void print_M(vector<float>& A, int rowS, int colS)
{
    for (int i = 0; i < rowS; i++)
    {
        for (int j = 0; j < colS; j++)
        {
            cout << A[i * colS + j] << " ";
        }
        cout << endl;
    }

}
int main()
{
    
    std::vector<float> input(inputRowSize, 0);
    std::vector<float> kernel(inputRowSize * outputRowSize, 0);
    std::vector<float> output(outputRowSize, 0);
    std::vector<float> kernelBias(outputRowSize, 1);
    init_ij(input, inputRowSize, 1);
    // init_(kernelBias, outputRowSize, 1);
    init_ij(kernel, inputRowSize, outputRowSize);
    // GEMM(input, inputRowSize, inputColSize,kernel, kernelRowSize, kernelColSize, output, outputRowSize, outputColSize);
    // reluSPMV(input, inputRowSize, inputColSize,kernel, kernelRowSize, kernelColSize, output, outputRowSize, outputColSize);
    reluSPMV(input, inputRowSize, \
    kernel, outputRowSize, inputRowSize, \
    kernelBias,\
    output, outputRowSize);
    print_M(output, outputRowSize, 1);
    reluSPMVCheck(input, inputRowSize, \
    kernel, outputRowSize, inputRowSize, \
    kernelBias,\
    output, outputRowSize);
    printf("-----------------------check!-----------------------\n");
    print_M(output, outputRowSize, 1);

}
