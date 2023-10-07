
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// CUDA runtime
#include <cublas_v2.h>
using namespace std;
#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}
// #define DEBUG
#define element_type float
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])


int inputRowSize = 28, inputColSize = 28, inputChannel = 1;
 int outputRowSize = 24, outputColSize = 24, outputChannel = 6;
const int kernelRowSize = 5, kernelColSize = 5;
const int kernelOCSize = kernelColSize * kernelRowSize * inputChannel;
const int THREAD_HEIGHT = 1, THREAD_WIDTH = 1,                                         // 一个线程负责的元素数
        KERNEL_HEIGHT = kernelRowSize, KERNEL_WIDTH = kernelColSize,                                   // 卷积核大小
        BLOCK_HEIGHT = 8, BLOCK_WIDTH = 4,                                                 // 分块大小
        MALLOC_KERNEL_HEIGHT = KERNEL_HEIGHT % 2 == 0 ? KERNEL_HEIGHT : KERNEL_HEIGHT + 1, // 用于kernel在SMEM的修正尺寸 奇数尺寸无法分配空间
        MALLOC_KERNEL_WIDTH = KERNEL_WIDTH % 2 == 0 ? KERNEL_WIDTH : KERNEL_WIDTH + 1,     // 用于kernel在SMEM的修正尺寸
        MALLOC_BLOCK_HEIGHT = (BLOCK_HEIGHT + KERNEL_HEIGHT) * 2,                          // 用于block在SMEM的修正尺寸
        MALLOC_BLOCK_WIDTH = (BLOCK_WIDTH + KERNEL_WIDTH) * 2,                             // 用于block在SMEM的修正尺寸
        MALLOC_TEMP_SIZE = outputChannel * 4;  
        
template <
    const int BLOCK_HEIGHT,
    const int BLOCK_WIDTH,
    const int KERNEL_HEIGHT,
    const int KERNEL_WIDTH,
    const int MALLOC_TEMP_SIZE,
    const int MALLOC_KERNEL_HEIGHT,
    const int MALLOC_KERNEL_WIDTH,
    const int MALLOC_BLOCK_HEIGHT,
    const int MALLOC_BLOCL_WIDTH>
__global__ void v1_convolution(element_type* in, element_type* out, element_type* kernel, element_type* kernelBias, int batch_size,
    int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelH, int kernelW)
{
    // block id 与 thread id的读取与计算 分块是对target矩阵去分的
    // 目前按一个线程负责一个in的计算
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y, thread_col = threadIdx.x;
    int threadH = BLOCK_HEIGHT, threadW = BLOCK_WIDTH; // 线程网格范围
    int thread_num_per_block = threadH * threadW, tid = thread_row * threadW + thread_col;
    // 分块边界 boundary是限制正常范围 edge是需要特殊处理的范围
    int row_boundary = outH / BLOCK_HEIGHT - 1,
        col_boundary = outW / BLOCK_WIDTH - 1;
    int row_edge = outH % BLOCK_HEIGHT, col_edge = outW % BLOCK_WIDTH;
    // 线程计算值暂存区大小 动态分配不是很方便 需要在外部分配并带进来
    // 一般取单个计算元素和oc之积的2倍即可 因为block比较小
    const int temp_size = MALLOC_TEMP_SIZE;

    // if (tid==0)
    //     printf("(%d %d)\n", block_row, block_col);

    /// 转移存储 GMEM --> SMEM
    // __shared__ float s_in[BLOCK_HEIGHT + KERNEL_HEIGHT - 1][BLOCK_WIDTH + KERNEL_WIDTH - 1];
    __shared__ float s_kernel[MALLOC_KERNEL_HEIGHT][MALLOC_KERNEL_WIDTH]; // 开奇数内存会出错
    __shared__ float s_in[MALLOC_BLOCK_HEIGHT][MALLOC_BLOCL_WIDTH];       // 要满足修正的尺寸
    float load_reg[4];

    // 当前block的起始位置
    // int begin_pos = (block_row + thread_row) * BLOCK_HEIGHT + (block_col) * BLOCK_WIDTH + thread_col;
    //记录in矩阵的起始位置，他是根据in矩阵进行划分的
    int begin_pos = block_row * BLOCK_HEIGHT * inW + block_col * BLOCK_WIDTH;

    int single_trans_ele_num = 4;                               // 线程一次转移的数据数
    int cur_in_block_height = BLOCK_HEIGHT + KERNEL_HEIGHT - 1, // 读入in的block height，读入in的block的row大小
        cur_in_block_width = BLOCK_WIDTH + KERNEL_WIDTH - 1,    // 读入in的block width，读入in的block的col大小
        in_tile_thread_per_row,                                 // 以tile为单位转移数据，一行需要的thread数
        in_tile_row_start,                                      // tile的行起始位置
        in_tile_col,                                            // tile的列
        in_tile_row_stride;                                     // tile行跨度

    // 修正边缘block尺寸，如果是in边缘的block，需要多读几个数据，相当于处理边界情况
    if (block_row == row_boundary)
    {
        cur_in_block_height = BLOCK_HEIGHT + row_edge + kernelH - 1;
    }
    if (block_col == col_boundary)
    {
        cur_in_block_width = BLOCK_WIDTH + col_edge + kernelW - 1;
    }

    in_tile_thread_per_row = cur_in_block_width / single_trans_ele_num; //每个线程读取single_trans_ele_num个数据，则一行需要的线程数
    in_tile_row_start = tid / in_tile_thread_per_row; //就是说这个tid对应的行是多少，我理解的tile是一行
    in_tile_col = tid % in_tile_thread_per_row * single_trans_ele_num; // 获得这个tid对应的这行第几个然后*4就知道他从哪一列开始读取
    in_tile_row_stride = thread_num_per_block / in_tile_thread_per_row; // 每个thread需要跳跃的大小

    // 下方都是读取第一个channel的数据
    // 按行读取 每行令线程以tile为单位读取 tile大小目前为single_trans_ele_num，余量特殊处理
    for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
        i += in_tile_row_stride)
    {
        // if (block_row == 0 && block_col == 0)
        // {
        //     printf("%d (%d %d) %d %d\n", tid, in_tile_row_start + i, in_tile_col, cur_in_block_height, cur_in_block_width);
        // }
        FETCH_FLOAT4(load_reg[0]) =
            FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inW)]);
        s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
        s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
        s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
        s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
        if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
            cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
        {
            for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
            {
                s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, inW)];
            }
        }
    }

    // 读取第一个kernel的数据
    if (thread_row >= 0 && thread_row < KERNEL_HEIGHT && thread_col == 0)
    {
        for (int j = 0; j < KERNEL_WIDTH; j++)
        {
            s_kernel[thread_row][j] = kernel[OFFSET(thread_row, j, KERNEL_WIDTH)];
        }
    }

    __syncthreads();
    // 验证数据转移正确性
    // if (block_row == 0 && block_col == 0 && thread_row == 0 && thread_col == 0) // 16 8
    // {
    //     for (int i = 0; i < cur_in_block_height; i++)
    //     {
    //         for (int j = 0; j < cur_in_block_width; j++)
    //         {
    //             printf("(%d %d) %.2f|%.2f\n", i, j, s_in[i][j], in[begin_pos + OFFSET(i, j, inW)]);
    //         }
    //     }
    // }
    // if (block_row == 2 && block_col == 2 && tid == 0)
    // {
    //     for (int i = 0; i < KERNEL_HEIGHT; i++)
    //     {
    //         for (int j = 0; j < KERNEL_WIDTH; j++)
    //         {
    //             printf("(%d %d) %.2f|%.2f\n", i, j, s_kernel[i][j], kernel[OFFSET(thread_row, j, KERNEL_WIDTH)]);
    //         }
    //     }
    // }

    // 逐个channel计算 一个线程负责block中的一个元素计算
    // 修正out block的大小
    int cur_out_block_height = BLOCK_HEIGHT, // 输出block height
        cur_out_block_width = BLOCK_WIDTH,   // 输出block width
        single_calculate_num = 1,            // 线程一次负责计算的元素数目
        out_tile_thread_per_row,             // block按tile划分需要的线程数目
        out_tile_row_start,                  // tile的行起始位置
        out_tile_col,                        // tile的列起始位置
        out_tile_row_stride;                 // tile行跨度
    if (block_row == row_boundary)
    {
        cur_out_block_height = BLOCK_HEIGHT + row_edge;
    }
    if (block_col == col_boundary)
    {
        cur_out_block_width = BLOCK_WIDTH + col_edge;
    }

    out_tile_thread_per_row = cur_out_block_width / single_calculate_num;
    out_tile_row_start = tid / out_tile_thread_per_row;
    out_tile_col = tid % out_tile_thread_per_row * single_calculate_num;
    out_tile_row_stride = thread_num_per_block / out_tile_thread_per_row;

    float val[temp_size]; // 存储累积和 避免多次读取GMEM
    for (int i = 0; i < temp_size; i++)
        val[i] = 0;

    int out_pos, temp_pos;

    for (int oc = 0; oc < outC; oc++)
    {
        for (int ic = 0; ic < inC; ic++)
        {
            // i,j 是相当于当前block起始位置而言
            // 每个线程负责一个tile，元素数>线程数会进行轮替，会有少部分的重叠区域，代价不大（只要width不大）
            // 用ic的每个block去对oc的kernel进行计算
            for (int i = 0; i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height;
                i += out_tile_row_stride)
            {
                for (int j = 0; j < single_calculate_num; j++)
                {
                    // 计算线程负责的元素 同一个oc的缓存顺序排列
                    // 不同oc偏移一个cur_out_block_height / out_tile_row_stride + 1的位置
                    temp_pos = i / out_tile_row_stride + j +
                        oc * (cur_out_block_height / out_tile_row_stride + 1);
                    for (int ii = 0; ii < KERNEL_HEIGHT; ii++)
                    {
                        for (int jj = 0; jj < KERNEL_WIDTH; jj++) // 更换的是SMEM中的内容，相对位置不变
                        {
                            val[temp_pos] += s_in[out_tile_row_start + i + ii][out_tile_col + j + jj] * s_kernel[ii][jj];
                        }
                    }
                }
            }
            // 读取下一个in channel和对应kernel的数据
            if (ic + 1 < inC)
            {
                for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
                    i += in_tile_row_stride)
                {
                    FETCH_FLOAT4(load_reg[0]) =
                        FETCH_FLOAT4(in[begin_pos + (ic + 1) * inH * inW + OFFSET(in_tile_row_start + i, in_tile_col, inW)]);
                    s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
                    s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
                    s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
                    s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
                    if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
                        cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
                    {
                        for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
                        {
                            s_in[in_tile_row_start + i][j] = in[begin_pos + (ic + 1) * inH * inW + OFFSET(in_tile_row_start + i, j, inW)];
                        }
                    }
                }
                if (thread_row >= 0 && thread_row < KERNEL_HEIGHT && thread_col == 0)
                {
                    for (int j = 0; j < KERNEL_WIDTH; j++)
                    {
                        s_kernel[thread_row][j] = kernel[(oc * inC + ic + 1) * kernelH * kernelW + OFFSET(thread_row, j, KERNEL_WIDTH)];
                    }
                }
            }

            __syncthreads();
            // 验证数据转移
            // if (ic + 1 < inC)
            //     if (block_row == 2 && block_col == 2 && thread_row == 0 && thread_col == 0) // 16 8
            //     {
            //         for (int i = 0; i < cur_in_block_height; i++)
            //         {
            //             for (int j = 0; j < cur_in_block_width; j++)
            //             {
            //                 printf("(%d %d) %.2f|%.2f\n", i, j, s_in[i][j], in[begin_pos + (ic + 1) * inH * inW + OFFSET(i, j, inW)]);
            //             }
            //         }
            //     }
        }
        // 读取下一个kernel channel数据
        if (oc + 1 < outC)
        {
            if (thread_row >= 0 && thread_row < KERNEL_HEIGHT && thread_col == 0)
            {
                for (int j = 0; j < KERNEL_WIDTH; j++)
                {
                    s_kernel[thread_row][j] = kernel[(oc + 1) * inC * kernelH * kernelW + OFFSET(thread_row, j, KERNEL_WIDTH)];
                }
            }
        }
        __syncthreads();
        // 写回 利用线程id计算写回位置
        int i = 0, j = 0;
        while (i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height)
        {
            while (j < single_calculate_num)
            {
                out_pos = oc * outH * outW +
                    block_row * BLOCK_HEIGHT * outW + block_col * BLOCK_WIDTH +
                    OFFSET(out_tile_row_start + i, out_tile_col + j, outW);
                temp_pos = i / out_tile_row_stride + j +
                    oc * (cur_out_block_height / out_tile_row_stride + 1);
                // if (tid == 0 && block_row == 0 && block_col == 0)
                // {
                //     printf("%d %d-(%d %d) %d %d\n", i, j, out_tile_row_start + i, out_tile_col + j,
                //            temp_pos, out_pos);
                // }
                out[out_pos] = val[temp_pos] + kernelBias[oc];
                j++;
            }
            i += out_tile_row_stride;
            j = 0;
        }
        // 读取下一个in channel数据
        for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
            i += in_tile_row_stride)
        {
            // if (block_row == 0 && block_col == 0)
            // {
            //     printf("%d (%d %d) %d %d\n", tid, in_tile_row_start + i, in_tile_col, cur_in_block_height, cur_in_block_width);
            // }
            FETCH_FLOAT4(load_reg[0]) =
                FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inW)]);
            s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
            s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
            s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
            s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
            if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
                cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
            {
                for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
                {
                    s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, inW)];
                }
            }
        }
    }
}


void conv2d(std::vector<float>  input, int inputRowSize, int inputColSize, int inputChannel, \
    std::vector<float>  kernel, std::vector<float> kernelBias, int kernelRowSize, int kernelColSize, \
    std::vector<float>& output, int outputRowSize, int outputColSize, int outputChannel)
{

    int N = 1; //batch size
    float* d_input, * d_output, * d_kernel, *d_kernelBias;
    int inputSize = inputChannel * inputRowSize * inputColSize * sizeof(float);
    int outputSize = outputChannel * outputRowSize * outputRowSize * sizeof(float);
    int kernelSize = kernelRowSize * kernelColSize * inputChannel * outputChannel * sizeof(float);
    int kernelBiasSize = outputChannel * sizeof(float);
    dim3 dimGrid(outputColSize / BLOCK_WIDTH, outputRowSize / BLOCK_HEIGHT);
    dim3 dimBlock(BLOCK_WIDTH / THREAD_WIDTH, BLOCK_HEIGHT / THREAD_HEIGHT);
    checkCudaErrors(cudaMalloc(&d_input, inputSize));
    checkCudaErrors(cudaMalloc(&d_output, outputSize));
    checkCudaErrors(cudaMalloc(&d_kernel, kernelSize));
    checkCudaErrors(cudaMalloc(&d_kernelBias, kernelBiasSize));
    
    checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias, kernelBias.data(), kernelBiasSize, cudaMemcpyHostToDevice));

    // v1_convolution<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT, KERNEL_WIDTH, MALLOC_TEMP_SIZE,
    //     MALLOC_KERNEL_HEIGHT, MALLOC_KERNEL_WIDTH, MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH>
    //     << <dimGrid, dimBlock >> > (d_input, d_output, d_kernel, d_kernelBias,
    //         N, inputChannel, inputRowSize, inputColSize, outputChannel, outputRowSize, outputColSize, kernelRowSize, kernelColSize);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));

#if 0
    for (int c = 0; c < outputChannel; c++)
    {
        for (int i = 0; i < outputRowSize; i++)
        {
            for (int j = 0; j < outputColSize; j++)
            {
                //elementwise + reduce
                double tmp = 0;
                for (int tc = 0; tc < inputChannel; tc++)
                {
                    for (int row = i; row < i + kernelRowSize; row++)
                    {
                        for (int col = j; col < j + kernelColSize; col++)
                        {
                            tmp += kernel[c * kernelRowSize * kernelColSize * inputChannel + \
                                tc * kernelColSize * kernelRowSize + \
                                (row - i) * kernelColSize + (col - j)] * \
                                input[tc * inputRowSize * inputColSize + row * inputColSize + col];
                        }
                    }
                }

                output[c * outputRowSize * outputColSize + i * outputColSize + j] = tmp + kernelBias[c];

            }
        }
    }
#endif
}
void init_ij(vector<float>& A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
                A[i * n * m + j * m + k] = -k + j + i;
        }
}
void init_one(vector<float>& A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
                A[i * n * m + j * m + k] = 1;
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
void conv2dCheck(std::vector<float>  input, int inputRowSize, int inputColSize, int inputChannel, \
    std::vector<float>  kernel, std::vector<float> kernelBias, int kernelRowSize, int kernelColSize, \
    std::vector<float>& output, int outputRowSize, int outputColSize, int outputChannel)
{
#if 0
    int N = 1; //batch size
    float* d_input, * d_output, * d_kernel;
    int inputSize = inputChannel * inputRowSize * inputColSize * sizeof(float);
    int outputSize = outputChannel * outputRowSize * outputRowSize * sizeof(float);
    int kernelSize = kernelRowSize * kernelColSize * inputChannel * outputChannel * sizeof(float);
    dim3 dimGrid(outputColSize / BLOCK_WIDTH, outputRowSize / BLOCK_HEIGHT);
    dim3 dimBlock(BLOCK_WIDTH / THREAD_WIDTH, BLOCK_HEIGHT / THREAD_HEIGHT);
    checkCudaErrors(cudaMalloc(&d_input, inputSize));
    checkCudaErrors(cudaMalloc(&d_output, outputSize));
    checkCudaErrors(cudaMalloc(&d_kernel, kernelSize));
    checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));

    v1_convolution<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT, KERNEL_WIDTH, MALLOC_TEMP_SIZE,
        MALLOC_KERNEL_HEIGHT, MALLOC_KERNEL_WIDTH, MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH>
        << <dimGrid, dimBlock >> > (d_input, d_output, d_kernel,
            N, inputChannel, inputRowSize, inputColSize, outputChannel, outputRowSize, outputColSize, kernelRowSize, kernelColSize);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));

#else
    for (int c = 0; c < outputChannel; c++)
    {
        for (int i = 0; i < outputRowSize; i++)
        {
            for (int j = 0; j < outputColSize; j++)
            {
                //elementwise + reduce
                double tmp = 0;
                for (int tc = 0; tc < inputChannel; tc++)
                {
                    for (int row = i; row < i + kernelRowSize; row++)
                    {
                        for (int col = j; col < j + kernelColSize; col++)
                        {
                            tmp += kernel[c * kernelRowSize * kernelColSize * inputChannel + \
                                tc * kernelColSize * kernelRowSize + \
                                (row - i) * kernelColSize + (col - j)] * \
                                input[tc * inputRowSize * inputColSize + row * inputColSize + col];
                        }
                    }
                }

                output[c * outputRowSize * outputColSize + i * outputColSize + j] = tmp + kernelBias[c];

            }
        }
    }
#endif
}
void print_M(vector<float>& A, int rowS, int colS, int chaS)
{
    for (int c = 0; c < chaS; c++)
    {

        cout <<"channel : " << c <<  endl;
        for (int i = 0; i < rowS; i++)
        {

            for (int j = 0; j < colS; j++)
            {
                cout << A[c * rowS * colS + i * colS + j] << " ";
            }
            cout << endl;
        }
    }
    
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ void conv2d1_old(float* input, float* output, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias,
    int inputChannel, int outputChannel, int inputSize, int kernelSize) //是方形
{
    //放入一个channle的大小，每个block处理一个output channel,大小是inputSize, 还是多个output channel？
    __shared__ float in_s[28][28];
    __shared__ float ker_s[5][5];
    //确定要处理哪个outputchannel, 2d grid
    int oc = blockIdx.x + blockIdx.y * blockDim.x;
    int outputSize = inputSize - kernelSize + 1;
    //循环遍历所有的inputchannel

    for (int ic = 0; ic < inputChannel; ic++)
    {

        //确定处理output的元素位置和input的位置；
        int destY = threadIdx.y, destX = threadIdx.x;
        int srcY = destY , srcX = destX ;
        // printf("intputSize = %d, destY = %d, destX = %d\n", inputSize, destY, destX);
        //先把input kernel的数据放到共享内存
        if (destY < kernelSize && destX < kernelSize)
        {
            int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                ic * kernelSize * kernelSize + destY * kernelSize + destX;
            ker_s[destY][destX] = kernel[ker_pos];
        }
        if (destY < inputSize && destX < inputSize)
        {
            int in_pos = ic * inputSize * inputSize + destY * inputSize + destX;
            in_s[destY][destX] = input[in_pos];
            // printf("intput[%d] = %f assign in_s[%d][%d] =%f\n", in_pos, input[in_pos], destY, destX, in_s[destY][destX]);
        }
        //数据同步
        __syncthreads();
        #if 0
        if (threadIdx.x == 0 && threadIdx.y == 0 && oc == 2)
        {
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < inputSize; j++)
                {
                    printf("in_s[%d][%d] = %f\n", i, j, in_s[i][j]);
                    printf("ker_s[%d][%d] = %f\n", i, j, ker_s[i][j]);
                }
        }
        #endif
        //计算‘
        float accum = 0;
        if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
        {
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    accum += in_s[srcY + i][srcX + j] * ker_s[i][j];
                    // if (oc == 2) printf("acc %f = in_s[%d][%d] = %f, ker_s[%d][%d] = %f \n", accum, srcY + i,
                    //     srcX + j, in_s[srcY + i][srcX + j] ,i, j, ker_s[i][j]);
                }
            }
            int out_pos = oc * outputSize * outputSize + destY * outputSize + destX;
            output[out_pos] = accum + kernel_bias[oc];
            // if (oc == 2)
            // {
            //     // printf("11111\n");
            //     printf("oc = 2, output[%d] = %f\n", out_pos, accum);
            // }
            
        }
    }
}

__global__ void conv2d1(float* input, float* output,float* output_pool, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias,
    int inputChannel, int outputChannel, int inputSize, int kernelSize) //是方形
{
    //放入一个channle的大小，每个block处理一个output channel,大小是inputSize, 还是多个output channel？
    __shared__ float in_s[28][28];
    __shared__ float in_pool_s[28][28];
    __shared__ float ker_s[5][5];
    //确定要处理哪个outputchannel, 2d grid
    int oc = blockIdx.x + blockIdx.y * blockDim.x;
    float tmp_bias = kernel_bias[oc];
    int outputSize = inputSize - kernelSize + 1;

    for (int ic = 0; ic < inputChannel; ic++)
    {
        int destY = threadIdx.y, destX = threadIdx.x;
        int srcY = destY, srcX = destX;
        if (destY < kernelSize && destX < kernelSize)
        {
            int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                ic * kernelSize * kernelSize + destY * kernelSize + destX;
            ker_s[destY][destX] = kernel[ker_pos];
        }
        if (destY < inputSize && destX < inputSize)
        {
            int in_pos = ic * inputSize * inputSize + destY * inputSize + destX;
            in_s[destY][destX] = input[in_pos];
        }

        __syncthreads();


        float accum = 0;
        if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
        {
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    accum += in_s[srcY + i][srcX + j] * ker_s[i][j];
                }
            }
            int out_pos = oc * outputSize * outputSize + destY * outputSize + destX;
            // output[out_pos] = accum + tmp_bias;


            if (destY < outputSize && destX < outputSize)
                in_pool_s[destY][destX] = accum + tmp_bias;
        }
        __syncthreads();
        //maxpool + relu
        int output_pool_size = outputSize / 2;
        int kernel_pool_size = 2;

        if (srcY  < output_pool_size && srcX < output_pool_size)
        {
            accum = 0;
            for (int i = 0; i < kernel_pool_size; i++)
                for (int j = 0; j < kernel_pool_size; j++)
                {
                    accum = max(accum, in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
                }
            if (accum > 0)
            {
                int out_pos = oc * output_pool_size * output_pool_size + srcY * output_pool_size + srcX;
                output_pool[out_pos] = accum;
            }
        }
    }
}

__global__ void conv2d2(float* input, float* output_pool, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias,
    int inputChannel, int outputChannel, int inputSize, int kernelSize) //是方形
{
    //放入一个channle的大小，每个block处理一个output channel,大小是inputSize, 还是多个output channel？
    __shared__ float in_s[28][28];
    __shared__ float in_pool_s[28][28];
    __shared__ float ker_s[5][5];
    //确定要处理哪个outputchannel, 2d grid
    int oc = blockIdx.x + blockIdx.y * blockDim.x;
    float tmp_bias = kernel_bias[oc];
    int outputSize = inputSize - kernelSize + 1;

        int destY = threadIdx.y, destX = threadIdx.x;
        int srcY = destY, srcX = destX;
    if (oc < outputChannel)
    {
        
    
    float accum = 0;
    for (int ic = 0; ic < inputChannel; ic++)
    {
        if (destY < kernelSize && destX < kernelSize)
        {
            int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                ic * kernelSize * kernelSize + destY * kernelSize + destX;
            ker_s[destY][destX] = kernel[ker_pos];
        }
        if (destY < inputSize && destX < inputSize)
        {
            int in_pos = ic * inputSize * inputSize + destY * inputSize + destX;
            in_s[destY][destX] = input[in_pos];
        }

        __syncthreads();


        
        if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
        {
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    accum += in_s[srcY + i][srcX + j] * ker_s[i][j];
                }
            }

        }
    
        
    }
    
    int out_pos = oc * outputSize * outputSize + destY * outputSize + destX;
            // output[out_pos] = accum + tmp_bias;
    if (destY < outputSize && destX < outputSize)
        in_pool_s[destY][destX] = accum + tmp_bias;

    __syncthreads();
        //maxpool + relu
    
    
    int output_pool_size = outputSize / 2;
    int kernel_pool_size = 2;

    if (srcY < output_pool_size && srcX < output_pool_size)
    {
        accum = 0;
        for (int i = 0; i < kernel_pool_size; i++)
            for (int j = 0; j < kernel_pool_size; j++)
            {
                accum = max(accum, in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
            }
        if (accum > 0)
        {
            int out_pos = oc * output_pool_size * output_pool_size + srcY * output_pool_size + srcX;
            output_pool[out_pos] = accum;
        }
        else 
        {
            output_pool[out_pos] = 0;
        }
    }
    
    }
}

template<int type>
__global__ void _conv2d(float* input, float* output_pool, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias) //是方形
{
    int inputChannel, outputChannel, inputSize, kernelSize;
    if (type == 1)
    {
        inputChannel = 1, outputChannel = 6, inputSize = 28, kernelSize = 5;
    }
    else
    {
        inputChannel = 6, outputChannel = 16, inputSize = 12, kernelSize = 5;
    }

    
    __shared__ float in_s[28][28];
    __shared__ float in_pool_s[28][28];
    __shared__ float ker_s[5][5];
    //确定要处理哪个outputchannel, 2d grid
    int oc = blockIdx.x + blockIdx.y * blockDim.x;
    float tmp_bias = kernel_bias[oc];
    int outputSize = inputSize - kernelSize + 1;

    int destY = threadIdx.y, destX = threadIdx.x;
    int srcY = destY, srcX = destX;
    if (oc < outputChannel)
    {


        float accum = 0;
        for (int ic = 0; ic < inputChannel; ic++)
        {
            if (destY < kernelSize && destX < kernelSize)
            {
                int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                    ic * kernelSize * kernelSize + destY * kernelSize + destX;
                ker_s[destY][destX] = kernel[ker_pos];
            }
            __syncthreads(); //奇怪，这个同步不能去

            if (threadIdx.y < inputSize && threadIdx.x < inputSize)
            {
                int in_pos = ic * inputSize * inputSize + threadIdx.y * inputSize + threadIdx.x;
                in_s[destY][destX] = input[in_pos];
                int a = 1;
            }

            __syncthreads();



            if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                            ic * kernelSize * kernelSize + i * kernelSize + j;
                        // accum += input[ic * inputSize * inputSize + (srcY + i) * inputSize + srcX + j] * kernel[ker_pos];
                        accum += in_s[srcY + i][srcX + j] * ker_s[i][j];
                    }
                }

            }


        }



        if (destY < outputSize && destX < outputSize)
            in_pool_s[destY][destX] = accum + tmp_bias;

        __syncthreads();


        int output_pool_size = outputSize / 2;
        int kernel_pool_size = 2;

        if (srcY < output_pool_size && srcX < output_pool_size)
        {
            float tmp_max = 0;
            for (int i = 0; i < kernel_pool_size; i++)
                for (int j = 0; j < kernel_pool_size; j++)
                {

                    tmp_max = max(tmp_max, in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
                }
            int out_pos = oc * output_pool_size * output_pool_size + srcY * output_pool_size + srcX;
            if (tmp_max >= 0)
            {

                output_pool[out_pos] = tmp_max;
            }
            else
            {
                output_pool[out_pos] = 0;
            }
        }



    }
}


__global__ void _conv2d_fusion(float* input, float* output_pool, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias, float* output_pool2, const float* __restrict__ kernel2,
    const float* __restrict__ kernel_bias2) //是方形
{
    // printf("111111111 blockDImx.x%d y %d z %d\n", blockDim.x, blockDim.y, blockDim.z);
    int inputChannel, outputChannel, inputSize, kernelSize;
    inputChannel = 1, outputChannel = 6, inputSize = 28, kernelSize = 5;
    
    
    __shared__ float in_s[28][28];
    __shared__ float in_pool_s[28][28];
    __shared__ float ker_s[5][5];



    //确定要处理哪个outputchannel, 2d grid
    // int oc = threadIdx.z;
    
    int outputSize = inputSize - kernelSize + 1;

    int destY = threadIdx.y, destX = threadIdx.x;
    int srcY = destY, srcX = destX;
    for (int oc = 0; oc < outputChannel; oc ++ )
    {

        float tmp_bias = kernel_bias[oc];
        float accum = 0;
        for (int ic = 0; ic < inputChannel; ic++)
        {
            if (destY < kernelSize && destX < kernelSize)
            {
                int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                    ic * kernelSize * kernelSize + destY * kernelSize + destX;
                ker_s[destY][destX] = kernel[ker_pos];
            }
            __syncthreads(); //奇怪，这个同步不能去

            if (threadIdx.y < inputSize && threadIdx.x < inputSize)
            {
                int in_pos = ic * inputSize * inputSize + threadIdx.y * inputSize + threadIdx.x;
                in_s[destY][destX] = input[in_pos];
                int a = 1;
            }

            __syncthreads();

            if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                            ic * kernelSize * kernelSize + i * kernelSize + j;
                        // accum += input[ic * inputSize * inputSize + (srcY + i) * inputSize + srcX + j] * kernel[ker_pos];
                        accum += in_s[srcY + i][srcX + j] * ker_s[i][j];
                    }
                }

            }


        }



        if (destY < outputSize && destX < outputSize)
            in_pool_s[destY][destX] = accum + tmp_bias;

        __syncthreads();


        int output_pool_size = outputSize / 2;
        int kernel_pool_size = 2;

        if (srcY < output_pool_size && srcX < output_pool_size)
        {
            float tmp_max = 0;
            for (int i = 0; i < kernel_pool_size; i++)
                for (int j = 0; j < kernel_pool_size; j++)
                {

                    tmp_max = max(tmp_max, in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
                }
            int out_pos = oc * output_pool_size * output_pool_size + srcY * output_pool_size + srcX;
            if (tmp_max >= 0)
            {

                output_pool[out_pos] = tmp_max;
            }
            else
            {
                output_pool[out_pos] = 0;
            }
        }
    }
    __syncthreads();
    //-----------_conv2d_1<2> << < 1, block >> > (d_output_pool, d_output_pool2, d_kernel2, d_kernelBias2);-----------
    //------------------------------------------------second--------------------------------------------------------------
    inputChannel = 6, outputChannel = 16, inputSize = 12, kernelSize = 5;
    outputSize = inputSize - kernelSize + 1;


    for (int oc = 0; oc < outputChannel; oc++)
    {

        float tmp_bias = kernel_bias2[oc];
        float accum = 0;
        for (int ic = 0; ic < inputChannel; ic++)
        {
            if (destY < kernelSize && destX < kernelSize)
            {
                int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                    ic * kernelSize * kernelSize + destY * kernelSize + destX;
                ker_s[destY][destX] = kernel2[ker_pos];
            }
            __syncthreads(); //奇怪，这个同步不能去

            if (threadIdx.y < inputSize && threadIdx.x < inputSize)
            {
                int in_pos = ic * inputSize * inputSize + threadIdx.y * inputSize + threadIdx.x;
                in_s[destY][destX] = output_pool[in_pos];
                int a = 1;
            }

            __syncthreads();

            if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                            ic * kernelSize * kernelSize + i * kernelSize + j;
                        // accum += input[ic * inputSize * inputSize + (srcY + i) * inputSize + srcX + j] * kernel[ker_pos];
                        accum += in_s[srcY + i][srcX + j] * ker_s[i][j];
                    }
                }

            }


        }



        if (destY < outputSize && destX < outputSize)
            in_pool_s[destY][destX] = accum + tmp_bias;

        __syncthreads();


        int output_pool_size = outputSize / 2;
        int kernel_pool_size = 2;

        if (srcY < output_pool_size && srcX < output_pool_size)
        {
            float tmp_max = 0;
            for (int i = 0; i < kernel_pool_size; i++)
                for (int j = 0; j < kernel_pool_size; j++)
                {

                    tmp_max = max(tmp_max, in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
                }
            int out_pos = oc * output_pool_size * output_pool_size + srcY * output_pool_size + srcX;
            if (tmp_max >= 0)
            {

                output_pool2[out_pos] = tmp_max;
            }
            else
            {
                output_pool2[out_pos] = 0;
            }
        }
    }
}


// void test_conv1()
// {
//     std::vector<float> input(inputRowSize * inputColSize * inputChannel, 0);
//     std::vector<float> kernel(kernelRowSize * kernelColSize * inputChannel * outputChannel, 0);
//     std::vector<float> output(outputRowSize * outputColSize * outputChannel, 0);
//     std::vector<float> output_pool(12 * 12 * 6, 0);
//     std::vector<float> kernelBias(outputChannel, 1.1);
//     init_ij(input, inputRowSize, inputColSize, inputChannel);
//     init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);
//     float* d_input, * d_output, * d_kernel, * d_kernelBias, * d_kernel2, * d_kernelBias2, * d_output_pool, * d_output_pool2;

//     outputRowSize = 24, outputColSize = 24, outputChannel = 6;
//     int inputSize = inputRowSize * inputColSize * inputChannel * sizeof(float);
//     int outputSize = outputRowSize * outputColSize * outputChannel * sizeof(float);
//     int kernelSize = kernelColSize * kernelRowSize * inputChannel * outputChannel * sizeof(float);
//     int kernelBiasSize = outputChannel * sizeof(float);
//     int kernelSize1 = 5 * 5 *  * sizeof(float);
//     int kernelBiasSize1 = outputChannel * sizeof(float);
//     checkCudaErrors(cudaMalloc(&d_input, inputSize));
//     checkCudaErrors(cudaMalloc(&d_output, outputSize));
//     checkCudaErrors(cudaMalloc(&d_kernel, kernelSize));
//     checkCudaErrors(cudaMalloc(&d_kernelBias, kernelBiasSize));
//     checkCudaErrors(cudaMalloc(&d_kernel2, kernelSize));
//     checkCudaErrors(cudaMalloc(&d_kernelBias2, kernelBiasSize));
//     checkCudaErrors(cudaMalloc(&d_output_pool, 12 * 12 * 6 * sizeof(float)));
//     checkCudaErrors(cudaMalloc(&d_output_pool2, 12 * 12 * 6 * sizeof(float)));

//     checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));
//     checkCudaErrors(cudaMemcpy(d_kernelBias, kernelBias.data(), kernelBiasSize, cudaMemcpyHostToDevice));
//     dim3 block(28, 28);
//     dim3 grid(6);
//     conv2d2 << < grid, block >> > (d_input, d_output_pool, d_kernel, d_kernelBias, 1, 6, 28, 5);
//     conv2d2 << < grid, block >> > (d_output_pool, d_output_pool2, d_kernel2, d_kernelBias2, 1, 6, 28, 5);
//     cudaDeviceSynchronize();
//     checkCudaErrors(cudaMemcpy(output.data(), d_output, 24 * 24 * 6 * sizeof(float), cudaMemcpyDeviceToHost));
//     checkCudaErrors(cudaMemcpy(output_pool.data(), d_output_pool, 12 * 12 * 6 * sizeof(float), cudaMemcpyDeviceToHost));

//     print_M(output, 24, 24, 6);
//     print_M(output_pool, 12, 12, 6);

// }

// void check_fusion()
// {


//     for (int c = 0; c < outputChannel; c++)
//     {
//         for (int i = 0; i < outputRowSize; i++)
//         {
//             for (int j = 0; j < outputColSize; j++)
//             {
//                 //elementwise + reduce
//                 double tmp = 0;
//                 for (int tc = 0; tc < inputChannel; tc++)
//                 {
//                     for (int row = i; row < i + kernelRowSize; row++)
//                     {
//                         for (int col = j; col < j + kernelColSize; col++)
//                         {
//                             tmp += kernel[c * kernelRowSize * kernelColSize * inputChannel + \
//                                 tc * kernelColSize * kernelRowSize + \
//                                 (row - i) * kernelColSize + (col - j)] * \
//                                 input[tc * inputRowSize * inputColSize + row * inputColSize + col];
//                         }
//                     }
//                 }

//                 output[c * outputRowSize * outputColSize + i * outputColSize + j] = tmp + kernelBias[c];

//             }
//         }
//     }
// }
void test_conv_fusion()
{
    int input_size1 = 28 * 28 * 1 * sizeof(float),
        input_size2 = 12 * 12 * 6 * sizeof(float);
    std::vector<float> input1(28 * 28 * 1, 0);
    std::vector<float> kernel1(5 * 5 * 6, 1);
    std::vector<float> output_pool(12 * 12 * 6, 1);
    std::vector<float> output(24 * 24 * 6, 0);
    std::vector<float> output1(24 * 24 * 6, 0);
    std::vector<float> output2(8 * 8 * 16, 0);
    std::vector<float> input2(12 * 12 * 6, 0);
    std::vector<float> kernel2(5 * 5 * 6 * 16, 1);
    std::vector<float> output_pool2(4 * 4 * 16, 0);
    std::vector<float> kernelBias1(6, 1);
    std::vector<float> kernelBias2(16, 1);
    init_ij(input1, 28, 28, 1);
    // init_ij(kernel1, 5, 5, 6);
    // init_ij(kernel2, 5, 5, 6 * 16);
    // init_ij(input2, inputRowSize, inputColSize, inputChannel);
    // init_one(kernel1, kernelRowSize, kernelColSize, inputChannel * outputChannel);
    float* d_input1, * d_output1, * d_kernel1, * d_kernelBias1,
        * d_kernel2, * d_kernelBias2, * d_output_pool, * d_output_pool2;

    int input1Size = 28 * 28 * 1 * sizeof(float);
    int output1Size = 12 * 12 * 6 * sizeof(float);
    int kernel1Size = 5 * 5 * 6 * 1 * sizeof(float);
    int kernelBias1Size = 6 * sizeof(float);
    int kernel2Size = 5 * 5 * 6 * 16 *sizeof(float);
    int kernelBias2Size = 16 * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input1, input1Size));
    checkCudaErrors(cudaMalloc(&d_output1, output1Size));
    checkCudaErrors(cudaMalloc(&d_kernel1, kernel1Size));
    checkCudaErrors(cudaMalloc(&d_kernelBias1, kernelBias1Size));
    checkCudaErrors(cudaMalloc(&d_kernel2, kernel2Size));
    checkCudaErrors(cudaMalloc(&d_kernelBias2, kernelBias2Size));
    checkCudaErrors(cudaMalloc(&d_output_pool, 12 * 12 * 6 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output_pool2, 4 * 4 * 16 * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_input1, input1.data(), input1Size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel1, kernel1.data(), kernel1Size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias1, kernelBias1.data(), kernelBias1Size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel2, kernel2.data(), kernel2Size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias2, kernelBias2.data(), kernelBias2Size, cudaMemcpyHostToDevice));
    dim3 block(28, 28);
    dim3 grid(16);
#if 0

    _conv2d_1<1> << < 1, block >> > (d_input1, d_output_pool, d_kernel1, d_kernelBias1);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {

        fprintf(stderr, "CUDA error111: %s %d\n", cudaGetErrorString(cudaError), inputRowSize);
        // 处理错误
    }
    // checkCudaErrors(cudaMemcpy(d_output_pool, output_pool.data(), output_pool.size() * sizeof(float), cudaMemcpyHostToDevice));
// checkCudaErrors(cudaMemcpy(output.data(), d_output, 24 * 24 * 6 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    _conv2d_1<2> << < 1, block >> > (d_output_pool, d_output_pool2, d_kernel2, d_kernelBias2);
#else
    _conv2d_fusion << < 1, block >> > (d_input1, d_output_pool, d_kernel1, d_kernelBias1, d_output_pool2, d_kernel2, d_kernelBias2);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {

        fprintf(stderr, "CUDA error111: %s %d\n", cudaGetErrorString(cudaError), inputRowSize);
        // 处理错误
    }
#endif
    cudaDeviceSynchronize();
    // checkCudaErrors(cudaMemcpy(output.data(), d_output, 24 * 24 * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(output_pool2.data(), d_output_pool2, 4 * 4 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

    // print_M(output, 24, 24, 6);
    print_M(output_pool2, 4, 4, 16);
    // check_fusion()
    for (int c = 0; c < 6; c++)
    {
        for (int i = 0; i < 24; i++)
        {
            for (int j = 0; j < 24; j++)
            {
                //elementwise + reduce
                double tmp = 0;
                for (int tc = 0; tc < 1; tc++)
                {
                    for (int row = i; row < i +5; row++)
                    {
                        for (int col = j; col < j + 5; col++)
                        {
                            tmp += kernel1[c * 5 * 5 * tc + \
                                tc * 5 * 5 + \
                                (row - i) * 5 + (col - j)] * \
                                input1[tc * 28 * 28 + row * 28 + col];
                        }
                    }
                }

                output1[c * 24 * 24 + i * 24 + j] = tmp + kernelBias1[c];

            }
        }
        for (int i = 0; i < 12; i++)
        {
            for (int j = 0; j < 12; j++)
            {
                //relu + maxpool
                double tmp = 0;
                {
                    for (int row = 2 * i; row < 2 * i + 2; row++)
                    {
                        for (int col = 2 * j; col < 2 * j + 2; col++)
                        {
                            if (output1[c * 24 * 24 + row * 24 + col] >= 0)
                                tmp = max(output1[c * 24 * 24 + row * 24 + col], tmp);
                        }
                    }
                }

                output_pool[c * 12 * 12 + i * 12 + j] = tmp;

            }
        }
    }
    // print_M(output1, 24, 24, 6);
    print_M(output_pool, 12, 12, 6);
    for (int c = 0; c < 16; c++)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                //elementwise + reduce
                double tmp = 0;
                for (int tc = 0; tc < 6; tc++)
                {
                    for (int row = i; row < i + 5; row++)
                    {
                        for (int col = j; col < j + 5; col++)
                        {
                            tmp += kernel2[c * 5 * 5 * tc + \
                                tc * 5 * 5 + \
                                (row - i) * 5 + (col - j)] * \
                                output_pool[tc * 12 * 12 + row * 12 + col];
                                // printf("tmp = %f, ker = %f, in = %f\n ", tmp, kernel2[c * 5 * 5 * tc + tc * 5 * 5 + \
                                // (row - i) * 5 + (col - j)], \
                                // output_pool[tc * 12 * 12 + row * 12 + col]);
                        }
                    }
                }

                output2[c * 8 * 8 + i * 8 + j] = tmp + kernelBias2[c];

            }
        }
        for (int i = 0; i < 4 ; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                //relu + maxpool
                double tmp = 0;
                {
                    for (int row = 2 * i; row < 2 * i + 2; row++)
                    {
                        for (int col = 2 * j; col < 2 * j + 2; col++)
                        {
                            if (output2[c * 8 * 8 + row * 8 + col] >= 0)
                                tmp = max(output2[c * 8 * 8 + row * 8 + col], tmp);
                        }
                    }
                }

                output_pool2[c * 4 * 4 + i * 4 + j] = tmp;

            }
        }
    }
    print_M(output2, 8 ,8, 16);
    print_M(output_pool2, 4, 4, 16);
}
void test_conv2()
{
    int inputRowSize = 12, inputColSize = 12, inputChannel = 6;

    outputRowSize = 8, outputColSize = 8, outputChannel = 16;
    std::vector<float> input(inputRowSize * inputColSize * inputChannel, 0);
    std::vector<float> kernel(kernelRowSize * kernelColSize * inputChannel * outputChannel, 0);
    std::vector<float> output(outputRowSize * outputColSize * outputChannel, 0);
    std::vector<float> output_pool(4 * 4 * 16, 0);
    std::vector<float> kernelBias(outputChannel, 1.1);
    init_ij(input, inputRowSize, inputColSize, inputChannel);
    print_M(input, 12, 12, 6);
    init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);
    float* d_input, * d_output, * d_kernel, * d_kernelBias, * d_output_pool;
    int inputSize = inputRowSize * inputColSize * inputChannel * sizeof(float);
    int outputSize = outputRowSize * outputColSize * outputChannel * sizeof(float);
    int kernelSize = kernelColSize * kernelRowSize * inputChannel * outputChannel * sizeof(float);
    int kernelBiasSize = outputChannel * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, inputSize));
    checkCudaErrors(cudaMalloc(&d_output, outputSize));
    checkCudaErrors(cudaMalloc(&d_kernel, kernelSize));
    checkCudaErrors(cudaMalloc(&d_kernelBias, kernelBiasSize));
    checkCudaErrors(cudaMalloc(&d_output_pool, 4 * 4 * 16 * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernelBias, kernelBias.data(), kernelBiasSize, cudaMemcpyHostToDevice));
    dim3 block(28, 28);
    dim3 grid(18);
    conv2d2 << < grid, block >> > (d_input, d_output_pool, d_kernel, d_kernelBias, 6, 16, 12, 5);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(output.data(), d_output, 12 * 12 * 6 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(output_pool.data(), d_output_pool, 4 * 4 * 16 * sizeof(float), cudaMemcpyDeviceToHost));

    // print_M(output, 24, 24, 6);
    print_M(output_pool, 4, 4,16);

}

int main()
{
    std::vector<float> input(inputRowSize * inputColSize * inputChannel, 0);
    std::vector<float> kernel(kernelRowSize * kernelColSize * inputChannel * outputChannel, 0);
    std::vector<float> output(outputRowSize * outputColSize * outputChannel, 0);
    std::vector<float> kernelBias(outputChannel, 1);
    init_ij(input, inputRowSize, inputColSize, inputChannel);
    init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);

    test_conv_fusion();
    // conv2d(input, inputRowSize, inputColSize, inputChannel, \
    //     kernel, kernelBias, kernelRowSize, kernelColSize, \
    //     output, outputRowSize, outputColSize, outputChannel);
    // print_M(output, outputRowSize, outputColSize, outputChannel);
    // conv2dCheck(input, inputRowSize, inputColSize, inputChannel, \
    //     kernel, kernelBias, kernelRowSize, kernelColSize, \
    //     output, outputRowSize, outputColSize, outputChannel);
    // printf("-------------------------check!-------------------------\n");
    // print_M(output, outputRowSize, outputColSize, outputChannel);

}
