
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


const int inputRowSize = 28, inputColSize = 28, inputChannel = 1;
const int outputRowSize = 24, outputColSize = 24, outputChannel = 6;
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

    v1_convolution<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT, KERNEL_WIDTH, MALLOC_TEMP_SIZE,
        MALLOC_KERNEL_HEIGHT, MALLOC_KERNEL_WIDTH, MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH>
        << <dimGrid, dimBlock >> > (d_input, d_output, d_kernel, d_kernelBias,
            N, inputChannel, inputRowSize, inputColSize, outputChannel, outputRowSize, outputColSize, kernelRowSize, kernelColSize);
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
                A[i * n * m + j * m + k] = k + j;
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
int main()
{
    std::vector<float> input(inputRowSize * inputColSize * inputChannel, 0);
    std::vector<float> kernel(kernelRowSize * kernelColSize * inputChannel * outputChannel, 0);
    std::vector<float> output(outputRowSize * outputColSize * outputChannel, 0);
    std::vector<float> kernelBias(outputChannel, 1);
    init_ij(input, inputRowSize, inputColSize, inputChannel);
    init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);


    conv2d(input, inputRowSize, inputColSize, inputChannel, \
        kernel, kernelBias, kernelRowSize, kernelColSize, \
        output, outputRowSize, outputColSize, outputChannel);
    print_M(output, outputRowSize, outputColSize, outputChannel);
    conv2dCheck(input, inputRowSize, inputColSize, inputChannel, \
        kernel, kernelBias, kernelRowSize, kernelColSize, \
        output, outputRowSize, outputColSize, outputChannel);
    printf("-------------------------check!-------------------------\n");
    print_M(output, outputRowSize, outputColSize, outputChannel);

}
