#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 
using namespace std;
const int inputRowSize = 256;
const int outputRowSize = 120;
// const int inputRowSize = 120;
// const int outputRowSize = 84;
// const int inputRowSize = 84;
// const int outputRowSize = 10;




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
__device__ void print_d(float* y, int len)
{
    for (int i = 0; i < len; i++)
        printf("%f\n", y[i]);
}

__global__ void relugemv_fusion(
    float* __restrict__ A,
    float* __restrict__ ABias,
    float* __restrict__ x,
    float* __restrict__ A1,
    float* __restrict__ ABias1,
    float* __restrict__ A2,
    float* __restrict__ ABias2,
    float* __restrict__ y2,
    float* predict, int t)
{
    //一个warp算y的一个元素
    int height = 120, width = 256;
    int warp_id = threadIdx.y;
    int warp_num = blockDim.y;
    const int warp_size = 32;

    //warp要取的col的start
    int col_vec_start = threadIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // __shared__ Arow_s[width];
    __shared__ float x_s[256];
    __shared__ float y[120];
    __shared__ float y1[84];
    __shared__ float out[10];
    if (tid < width)
        x_s[tid] = x[tid];
    __syncthreads();
    for (int row = warp_id; row < height; row += warp_num)
    {
        float tmp = 0;
        //取数据到Arow_s
        float4 current_val1 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start * 2];
        float4 current_val2 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start * 2 + 1];
        tmp += current_val1.x * x_s[col_vec_start * 8];
        tmp += current_val1.y * x_s[col_vec_start * 8 + 1];
        tmp += current_val1.z * x_s[col_vec_start * 8 + 2];
        tmp += current_val1.w * x_s[col_vec_start * 8 + 3];
        tmp += current_val2.x * x_s[col_vec_start * 8 + 4];
        tmp += current_val2.y * x_s[col_vec_start * 8 + 5];
        tmp += current_val2.z * x_s[col_vec_start * 8 + 6];
        tmp += current_val2.w * x_s[col_vec_start * 8 + 7];
        tmp = warpReduceSum<warp_size>(tmp);
        // printf("tmp %f, ")
        if (threadIdx.x == 0)
        {
            tmp += ABias[row];
            if (tmp >= 0)
                y[row] = tmp;
            else
                y[row] = 0;
        }

    }

    __syncthreads();
    #ifdef DEBUG
    if (tid == 0)
    {
        printf("------------------------------------y------------------------------------\n");
        print_d(y, 120);
    }
    #endif
    //-----------------------------------------------128 * 84-------------------------------------------------------------
    height = 84, width = 120;
    //一个warp算y的一个元素

    if (tid < width)
        x_s[tid] = y[tid];
    __syncthreads();
    for (int row = warp_id; row < height; row += warp_num)
    {

        float tmp = 0;
        //取数据到Arow_s
        if (col_vec_start * 4 < width)
        {
            float4 current_val1 = reinterpret_cast<float4*>(A1)[row * width / 4 + col_vec_start];
            // printf("current_val1 x %f y %f z %f w %f x[%d] %f %f %f %f\n", current_val1.x, current_val1.y, current_val1.z, current_val1.w, col_vec_start * 8 x_s[col_vec_start * 8]
            //     , x_s[col_vec_start * 8 + 1], x_s[col_vec_start * 8 + 2], x_s[col_vec_start * 8 + 3]);
            tmp += current_val1.x * x_s[col_vec_start * 4];
            tmp += current_val1.y * x_s[col_vec_start * 4 + 1];
            tmp += current_val1.z * x_s[col_vec_start * 4 + 2];
            tmp += current_val1.w * x_s[col_vec_start * 4 + 3];
        }

        tmp = warpReduceSum<warp_size>(tmp);
        if (threadIdx.x == 0)
        {
            tmp += ABias1[row];
            if (tmp >= 0)
                y1[row] = tmp;
            else
                y1[row] = 0;
        }

    }

    __syncthreads();
#ifdef DEBUG
    if (tid == 0)
    {
        printf("------------------------------------y1------------------------------------\n");
            print_d(y1, 84);
    }
    #endif
    //-----------------------------------------------128 * 84-------------------------------------------------------------
    height = 10, width = 84;
    //一个warp算y的一个元素
    //warp要取的col的start
    if (tid < width)
        x_s[tid] = y1[tid];
    __syncthreads();
    for (int row = warp_id; row < height; row += warp_num)
    {

        float tmp = 0;
        //取数据到Arow_s
        if (col_vec_start * 4 < width)
        {
            float4 current_val1 = reinterpret_cast<float4*>(A2)[row * width / 4 + col_vec_start];
            tmp += current_val1.x * x_s[col_vec_start * 4];
            tmp += current_val1.y * x_s[col_vec_start * 4 + 1];
            tmp += current_val1.z * x_s[col_vec_start * 4 + 2];
            tmp += current_val1.w * x_s[col_vec_start * 4 + 3];
        }

        tmp = warpReduceSum<warp_size>(tmp);
        if (threadIdx.x == 0)
        {
            tmp += ABias[row];
            if (tmp >= 0)
            {
                out[row] = tmp;
                y2[row] = tmp;
            }
            else
            {

                out[row] = 0;
                y2[row] = 0;
            }
        }

    }
    __syncthreads();
#ifdef DEBUG
    if (tid == 0)
    {
        printf("------------------------------------y2------------------------------------\n");
        print_d(y2, 10);
    }
#endif

    if (tid == 0)
    {
        float tmp_max = 0, id = 0;
        for (int i = 0; i < 10; i++)
        {
            if (tmp_max < out[i])
            {
                tmp_max = out[i], id = i;
            }
        }
        predict[t] = id;
    }
}
__global__ void relugemv_new(
    float* __restrict__ A,
    float* __restrict__ ABias,
    float* __restrict__ x,
    float* __restrict__ y,
    int height,
    int width)
{
    //一个warp算y的一个元素
    int warp_id = threadIdx.y;
    int warp_num = blockDim.y;
    const int warp_size = 32;

    //warp要取的col的start
    int col_vec_start = threadIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // __shared__ Arow_s[width];
    __shared__ float x_s[256];
    if (tid < width)
        x_s[tid] = x[tid];
    __syncthreads();
    height = 120, width = 256;
    for (int row = warp_id; row < height; row += warp_num)
    {
        float tmp = 0;
        //取数据到Arow_s
        float4 current_val1 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start * 2];
        float4 current_val2 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start * 2 + 1];
        tmp += current_val1.x * x_s[col_vec_start * 8];
        tmp += current_val1.y * x_s[col_vec_start * 8 + 1];
        tmp += current_val1.z * x_s[col_vec_start * 8 + 2];
        tmp += current_val1.w * x_s[col_vec_start * 8 + 3];
        tmp += current_val1.x * x_s[col_vec_start * 8 + 4];
        tmp += current_val2.y * x_s[col_vec_start * 8 + 5];
        tmp += current_val2.z * x_s[col_vec_start * 8 + 6];
        tmp += current_val2.w * x_s[col_vec_start * 8 + 7];
        tmp = warpReduceSum<warp_size>(tmp);
        if (threadIdx.x == 0)
        {
            tmp += ABias[row];
            if (tmp >= 0)
                y[row] = tmp;
            else
                y[row] = 0;
        }

    }


}
__global__ void relugemv_new2(
    float* __restrict__ A,
    float* __restrict__ ABias,
    float* __restrict__ x,
    float* __restrict__ y,
    int height,
    int width)
{
    height = 84, width = 120;
    //一个warp算y的一个元素
    int warp_id = threadIdx.y;
    int warp_num = blockDim.y;
    const int warp_size = 32;

    //warp要取的col的start
    int col_vec_start = threadIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // __shared__ Arow_s[width];
    __shared__ float x_s[256];
    if (tid < width)
        x_s[tid] = x[tid];
    __syncthreads();
    for (int row = warp_id; row < height; row += warp_num)
    {
        
        float tmp = 0;
        //取数据到Arow_s
        if (col_vec_start * 4 < width)
        {
            float4 current_val1 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start];
            tmp += current_val1.x * x_s[col_vec_start * 8];
            tmp += current_val1.y * x_s[col_vec_start * 8 + 1];
            tmp += current_val1.z * x_s[col_vec_start * 8 + 2];
            tmp += current_val1.w * x_s[col_vec_start * 8 + 3];
        }
        
        tmp = warpReduceSum<warp_size>(tmp);
        if (threadIdx.x == 0)
        {
            tmp += ABias[row];
            if (tmp >= 0)
                y[row] = tmp;
            else
                y[row] = 0;
        }

    }


}

__global__ void relugemv_new3(
    float* __restrict__ A,
    float* __restrict__ ABias,
    float* __restrict__ x,
    float* __restrict__ y,
    int height,
    int width)
{
    height = 10, width = 84;
    //一个warp算y的一个元素
    int warp_id = threadIdx.y;
    int warp_num = blockDim.y;
    const int warp_size = 32;

    //warp要取的col的start
    int col_vec_start = threadIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // __shared__ Arow_s[width];
    __shared__ float x_s[256];
    if (tid < width)
        x_s[tid] = x[tid];
    __syncthreads();
    for (int row = warp_id; row < height; row += warp_num)
    {

        float tmp = 0;
        //取数据到Arow_s
        if (col_vec_start * 4 < width)
        {
            float4 current_val1 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start];
            tmp += current_val1.x * x_s[col_vec_start * 8];
            tmp += current_val1.y * x_s[col_vec_start * 8 + 1];
            tmp += current_val1.z * x_s[col_vec_start * 8 + 2];
            tmp += current_val1.w * x_s[col_vec_start * 8 + 3];
        }

        tmp = warpReduceSum<warp_size>(tmp);
        if (threadIdx.x == 0)
        {
            tmp += ABias[row];
            if (tmp >= 0)
                y[row] = tmp;
            else
                y[row] = 0;
        }

    }


}

void reluSPMV(std::vector<float> input, int inputRowSize, \
    std::vector<float> kernel, int kernelRowSize, int kernelColSize, \
    std::vector<float> kernelBias,\
    std::vector<float>& output, int outputRowSize)
{

    float* d_output, * d_kernel, * d_output1, * d_output2, * d_kernel1, * d_kernel2, * d_input, * d_kernelBias, * d_kernelBias1, * d_kernelBias2;
    int inputSize = sizeof(float) * inputRowSize, outputSize = sizeof(float) * outputRowSize;
    int kernelSize = sizeof(float) * kernelRowSize * kernelColSize, kernelBiasSize = kernelRowSize * sizeof(float);

    std::vector<float> kernel1(120 * 84, 1);
    std::vector<float> kernel2(84 * 10, 1), kernelBias1(84, 1), kernelBias2(10, 1);
    //malloc
    checkCudaErrors(cudaMalloc(&d_output, outputSize));
    checkCudaErrors(cudaMalloc(&d_output1, outputSize));
    checkCudaErrors(cudaMalloc(&d_output2, outputSize));
    checkCudaErrors(cudaMalloc(&d_kernel, kernelSize));
    checkCudaErrors(cudaMalloc(&d_input, inputSize));
    checkCudaErrors(cudaMalloc(&d_kernelBias, kernelBiasSize));
    checkCudaErrors(cudaMalloc(&d_kernel1, 120 * 84 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_kernelBias1, 84 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_kernel2, 84 * 10 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_kernelBias2, 10 * sizeof(float)));

    //memcpy H2D
    checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias, kernelBias.data(), kernelBiasSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_kernel1, kernel1.data(), 120 * 84 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias1, kernelBias1.data(), 84 * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_kernel2, kernel2.data(), 84 * 10 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias2, kernelBias2.data(), 10 * sizeof(float), cudaMemcpyHostToDevice));

    //call cudakernel M row, N col
    if (inputRowSize == 256)
    {
        dim3 block(32, 32);
        relugemv_fusion << < 1, block >> > (d_kernel, d_kernelBias, d_input, 
            d_kernel1, d_kernelBias1, 
            d_kernel2, d_kernelBias2, d_output2, d_output, 0);
            // printf("1111\n");
        cudaDeviceSynchronize();
    }
    else if (inputRowSize == 120)
    {
        dim3 block(32, 32);
        relugemv_new2 << < 1, block >> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize);
        cudaDeviceSynchronize();
    }
    else if (inputRowSize == 84)
    {
        dim3 block(32, 32);
        relugemv_new3 << < 1, block >> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize);
        cudaDeviceSynchronize();
    }
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {

        fprintf(stderr, "CUDA error111: %s %d\n", cudaGetErrorString(cudaError), inputRowSize);
        // 处理错误
    }
    //memcpy D2H
    checkCudaErrors(cudaMemcpy(output.data(), d_output2, 10 * sizeof(float), cudaMemcpyDeviceToHost));
    
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
void reluSPMVCheck(std::vector<float> input, int inputRowSize, \
    std::vector<float> kernel, int kernelRowSize, int kernelColSize, \
    std::vector<float> kernelBias,\
    std::vector<float>& output, int outputRowSize)
{
    std::vector<float> outTmp1(120, 0), outTmp2(84, 0);
    outputRowSize = 120, inputRowSize = 256, kernelColSize = inputRowSize;
    for (int i = 0; i < outputRowSize; i++)
    {
        double tmp = 0;
        for (int k = 0; k < inputRowSize; k++)
            tmp += kernel[i * kernelColSize + k] * input[k];
        tmp += kernelBias[i];
        if (tmp >= 0) outTmp1[i] = tmp;
        else outTmp1[i] = 0;
    }

    printf("-----------------------check 1!-----------------------\n");
    print_M(outTmp1, 120, 1);

    outputRowSize = 84, inputRowSize = 120, kernelColSize = inputRowSize;
    for (int i = 0; i < outputRowSize; i++)
    {
        double tmp = 0;
        for (int k = 0; k < inputRowSize; k++)
            tmp += 1 * outTmp1[k];
        tmp += kernelBias[i];
        if (tmp >= 0) outTmp2[i] = tmp;
        else outTmp2[i] = 0;
    }
    printf("-----------------------check 2!-----------------------\n");
    print_M(outTmp2, 84, 1);

    outputRowSize = 10, inputRowSize = 84, kernelColSize = inputRowSize;
    for (int i = 0; i < outputRowSize; i++)
    {
        double tmp = 0;
        for (int k = 0; k < inputRowSize; k++)
            tmp += 1 * outTmp2[k];
        tmp += kernelBias[i];
        if (tmp >= 0) output[i] = tmp;
        else output[i] = 0;
    }
    printf("-----------------------check 3!-----------------------\n");
    print_M(output, 10, 1);
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

    printf("-----------------------input!-----------------------\n");
    print_M(kernel, inputRowSize, outputRowSize);
    reluSPMV(input, inputRowSize, \
        kernel, outputRowSize, inputRowSize, \
    kernelBias,\
    output, outputRowSize);
    print_M(output, 10, 1);
    reluSPMVCheck(input, inputRowSize, \
    kernel, outputRowSize, inputRowSize, \
    kernelBias,\
    output, outputRowSize);

}
