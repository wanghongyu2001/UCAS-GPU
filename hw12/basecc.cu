// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

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


// const int kernelRowSize = 5, kernelColSize = 5;
const int N  =1;
const int KERNEL_HEIGHT = 5, KERNEL_WIDTH = 5, BLOCK_HEIGHT = 8, BLOCK_WIDTH = 4, MALLOC_TEMP_SIZE = 16 * 4;  
#define type float
const int  KERNEL_SIZE = 2, TMP_SIZE = 16 * 4;
template<
    const int BLOCK_HEIGHT,
    const int BLOCK_WIDTH,
    const int KERNEL_SIZE,
    const int TMP_SIZE>
__global__ void _reluMaxPoll(type* input, type* output, int inputChannel, int inputRowSize, int inputColSize,
    int outputChannel, int outputRowSize, int outputColSize,
    int kernelSize)
{
    // printf("222222\n");
    // printf("%d %d %d %d %d %d %d\n", inputChannel, inputRowSize, inputColSize,
    //     outputChannel, outputRowSize, outputColSize,
    //     kernelSize);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int row_boundary = (outputRowSize / blockDim.y - 1);
    int col_boundary = (outputColSize / blockDim.x - 1);
    row_boundary = row_boundary < 0 ? 0 : row_boundary;
    col_boundary = col_boundary < 0 ? 0 : col_boundary;
    // printf("row_boundary = %d, col_boundary = %d \n", row_boundary, col_boundary);
    int row_edge = outputRowSize % blockDim.y, col_edge = outputColSize % blockDim.x;
    // printf("row_edge = %d, col_edge = %d\n", row_edge, col_edge);
    //开两倍防止边界条件
    __shared__ type s_in[BLOCK_HEIGHT * 2][BLOCK_WIDTH * 2];
    float load_reg[4]; //加载到寄存器
    //这个block元素开始的位置 对应input的元素位置，因为pool层所以*kernel_size
    int begin_pos = (blockIdx.y * blockDim.y * inputColSize + blockIdx.x * blockDim.x) * 2;
    #ifdef DEBUG
    if (threadIdx.x == 0 && threadIdx.y == 0 )
        printf("begin_pos %d blockIdx.x = %d, idy = %d\n", begin_pos, blockIdx.x, blockIdx.y);
    #endif
    int trans_ele_num = 4; //线程转移元素数
    int cur_in_block_height = blockDim.y * 2, cur_in_block_width = blockDim.x * 2;
    int in_tile_thread_per_row, in_tile_row_start, in_tile_col, in_tile_row_stride;

    if (blockIdx.y == row_boundary)
        cur_in_block_height += row_edge * 2;
    if (blockIdx.x == col_boundary)
        cur_in_block_width += col_edge * 2;
    //对应于input，应该是乘以 kernel_size
    in_tile_thread_per_row = blockDim.x * 2 / trans_ele_num;
    in_tile_row_start = tid / in_tile_thread_per_row;
    in_tile_col = tid % in_tile_thread_per_row * trans_ele_num;
    in_tile_row_stride = blockDim.x * blockDim.y / in_tile_thread_per_row;
    #ifdef DEBUG
    if (threadIdx.x == 0 && threadIdx.y == 0) 
        printf("in_per_row = %d, in_tile_row_stride = %d, cur_in_block_height = %d ,cur_in_block_width = %d\n",
         in_tile_thread_per_row, in_tile_row_stride, cur_in_block_height, cur_in_block_width);
    #endif
    // 预取数据
    for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height; i += in_tile_row_stride)
    {
        FETCH_FLOAT4(load_reg[0]) = FETCH_FLOAT4(input[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)]);
        #ifdef DEBUG
        if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) 
        {
            printf("input pos %d \n", begin_pos + (in_tile_row_start + i) * inputColSize + in_tile_col);
        }
        #endif
        s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
        s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
        s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
        s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
        if (in_tile_col + 2 * trans_ele_num > cur_in_block_width && in_tile_col + trans_ele_num < cur_in_block_width)
        {
            for (int j = in_tile_col + trans_ele_num; j < cur_in_block_width; j++)
            {
                s_in[in_tile_row_start + i][j] = input[begin_pos + (in_tile_row_start + i) * inputColSize + j];
            }
        }
    }
    //等待数据trans结束
    __syncthreads();
    #ifdef DEBUG
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) // 16 8
    {
        for (int i = 0; i < cur_in_block_height; i++)
        {
            for (int j = 0; j < cur_in_block_width; j++)
            {
                printf("(%d %d) %.2f|%.2f\n", i, j, s_in[i][j], input[begin_pos + OFFSET(i, j, inputColSize)]);
            }
        }
    }
    #endif
    // 直接除以kernelsize = 2
    int cur_out_block_height = cur_in_block_height / 2;
    int cur_out_block_width = cur_in_block_width / 2;
    // if (blockIdx.y == row_boundary)
    //     cur_out_block_height += row_edge / 2;

    // if (blockIdx.x == col_boundary)
    //     cur_out_block_width += col_edge / 2;
    #ifdef DEBUG
    if (threadIdx.x == 0 && threadIdx.y == 0)
        printf("cur_out_block_height %d cur_out_block_width %d\n",cur_out_block_height, cur_out_block_width);
    #endif
    //这里映射到outtile
    int out_tile_thread_per_row = cur_out_block_width;
    int out_tile_row_start = (tid / out_tile_thread_per_row);
    int out_tile_col = tid % out_tile_thread_per_row;
    int out_tile_row_stride = blockDim.x * blockDim.y / out_tile_thread_per_row;// 32 / 4 = 8

    
    float tmp[TMP_SIZE];
    for (int i = 0; i < 4 * outputChannel; i++)
    {
        tmp[i] = 0;
    }

    for (int oc = 0; oc < outputChannel; oc++)
    {

            for (int i = 0; i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height; i += out_tile_row_stride)
            {
                int tmp_pos = i / out_tile_row_stride + oc * (cur_out_block_height / out_tile_row_stride + 1);
                float tmp_max = 0; //register
                for (int ii = 0; ii < KERNEL_SIZE; ii++)
                    for (int jj = 0; jj < KERNEL_SIZE; jj++)
                    {
                        tmp_max = max(s_in[(out_tile_row_start + i) * 2 + ii][out_tile_col * 2 + jj], tmp_max);
                        // printf("row %d col %d in %f \n", (out_tile_row_start + i) * 2 + ii, out_tile_col * 2 + jj, s_in[(out_tile_row_start + i) * 2 + ii][out_tile_col * 2 + jj]);
                    }
                if (tmp_max > 0)
                {
                    tmp[tmp_pos] = tmp_max;
                    #if DEBUG
                    printf("tmp %f \n", tmp_max);
                    #endif
                }
                else tmp[tmp_pos] = 0;
            }

            //取下一个channellinput数据
            if (oc + 1 < outputChannel)
            {
                for (int i = 0; i < cur_in_block_height && (i + in_tile_row_start < cur_in_block_height); i += in_tile_row_stride)
                {
                    FETCH_FLOAT4(load_reg[0]) = 
                        FETCH_FLOAT4(input[begin_pos + (oc + 1) * inputColSize * inputRowSize + (in_tile_row_start + i) * inputColSize + in_tile_col]);
                    s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
                    s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
                    s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
                    s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
                    //如果有多余的，不足4个
                    if (in_tile_col + 2 * trans_ele_num > cur_in_block_width &&
                        in_tile_col + trans_ele_num < cur_in_block_width)
                    {
                        for (int k = in_tile_col + trans_ele_num; k < cur_in_block_width; k++)
                        {
                            s_in[in_tile_row_start + i][k] = input[begin_pos + (oc + 1) * inputRowSize * inputColSize + (in_tile_row_start + i) * inputColSize + k];
                        }
                    }

                }
                #if DEBUG
                if (oc + 1 < inputChannel)
                    if (blockIdx.x == 2 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) // 16 8
                    {
                        for (int i = 0; i < cur_in_block_height; i++)
                        {
                            for (int j = 0; j < cur_in_block_width; j++)
                            {
                                printf("bIdx %d (%d %d) %.2f|%.2f\n", blockIdx.x, i, j, s_in[i][j], input[begin_pos + (oc + 1) * inputRowSize * inputColSize + OFFSET(i, j, inputColSize)]);
                            }
                        }
                    }
                #endif
            }

            
        
        //数据写回去

        for (int i = 0; i < cur_out_block_height && (i + out_tile_row_start) < cur_out_block_height; i += out_tile_row_stride)
        {
            int tmp_pos = i / out_tile_row_stride + oc * (cur_out_block_height / out_tile_row_stride + 1);
            int out_pos = oc * outputRowSize * outputColSize +
                cur_out_block_height * blockIdx.y * outputColSize + 
                blockIdx.x * cur_out_block_width +
                (out_tile_row_start + i) * outputColSize + 
                out_tile_col;
            output[out_pos] = tmp[tmp_pos];
            #ifdef DEBUG
            printf("tmp[%d] = %f, output[%d]\n",tmp_pos, tmp[tmp_pos], out_pos);
            #endif
        }

        //prefetch the next inputchannel data, in fact in = 0
        __syncthreads();
    }
    
}

void reluMaxPool(float* d_input, int inputRowSize, int inputColSize, int inputChannel, \
    int kernelRowSize, int kernelColSize, \
    float* d_output, int outputRowSize, int outputColSize, int outputChannel, cudaStream_t stream)
{
#if 1
    // float * d_output;
    // int input_size = inputChannel * inputColSize * inputRowSize * sizeof(float);
    // int output_size = outputChannel * outputColSize * outputRowSize * sizeof(float);
    // checkCudaErrors(cudaMalloc(&d_input, input_size));
    // checkCudaErrors(cudaMalloc(&d_output, output_size));

    //memcpy
    // checkCudaErrors(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));
    int gridx = outputColSize / BLOCK_WIDTH, gridy = outputRowSize / BLOCK_HEIGHT;
    gridx = gridx <= 0 ? 1 : gridx;
    gridy = gridy <= 0 ? 1 : gridy;
    #ifdef DEBUG
    printf("grid %d %d, block %d %d\n", gridx, gridy, BLOCK_WIDTH, BLOCK_HEIGHT);
    #endif
    dim3 grid(gridx, gridy);
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    // printf("1111\n");
    // f1 << <1, 200 >> > (1);
    // cudaDeviceSynchronize();
    _reluMaxPoll< BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_SIZE, TMP_SIZE> << <grid, block, 0, stream >> > (d_input, d_output, inputChannel, inputRowSize, inputColSize,
        outputChannel, outputRowSize, outputColSize,
        2);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        
        fprintf(stderr, "CUDA error111: %s %d\n", cudaGetErrorString(cudaError), inputRowSize);
        // 处理错误
    }
    else {
        // kernel 调用成功
    }
    // cudaDeviceSynchronize();
    // checkCudaErrors(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));
    // cudaFree(d_input);
    // cudaFree(d_output);
#else
    for (int c = 0; c < outputChannel; c++)
    {
        for (int i = 0; i < outputRowSize; i++)
        {
            for (int j = 0; j < outputColSize; j++)
            {
                //relu + maxpool
                double tmp = 0;
                {
                    for (int row = kernelRowSize * i; row < kernelRowSize * i + kernelRowSize; row++)
                    {
                        for (int col = kernelColSize * j; col < kernelColSize * j + kernelColSize; col++)
                        {
                            if (input[c * inputRowSize * inputColSize + row * inputColSize + col] >= 0)
                                tmp = max(input[c * inputRowSize * inputColSize + row * inputColSize + col], tmp);
                        }
                    }
                }

                output[c * outputRowSize * outputColSize + i * outputColSize + j] = tmp;

            }
        }
    }
#endif
}

        
template <
    const int BLOCK_HEIGHT, const int BLOCK_WIDTH, const int KERNEL_HEIGHT, const int KERNEL_WIDTH, const int MALLOC_TEMP_SIZE>
__global__ void _conv2d(element_type* in, element_type* out, element_type* kernel, element_type* kernelBias, int batch_size,
    int inputChannel, int inputRowSize, int inputColSize,
    int outputChannel, int outputRowSize, int outputColSize,
    int kernelH, int kernelW)
{
    // block id 与 thread id的读取与计算 分块是对target矩阵去分的
    // 目前按一个线程负责一个in的计算
    int thread_num_per_block = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
    // 分块边界 boundary是限制正常范围 edge是需要特殊处理的范围
    int row_boundary = outputRowSize / BLOCK_HEIGHT - 1,
        col_boundary = outputColSize / BLOCK_WIDTH - 1;
    int row_edge = outputRowSize % BLOCK_HEIGHT, col_edge = outputColSize % BLOCK_WIDTH;
    // 线程计算值暂存区大小 动态分配不是很方便 需要在外部分配并带进来
    // 一般取单个计算元素和oc之积的2倍即可 因为block比较小
    const int temp_size = MALLOC_TEMP_SIZE;

    // if (tid==0)
    //     printf("(%d %d)\n", blockIdx.y, blockIdx.x);

    
    // __shared__ float s_in[BLOCK_HEIGHT + KERNEL_HEIGHT - 1][BLOCK_WIDTH + KERNEL_WIDTH - 1];
    __shared__ float s_kernel[1 + KERNEL_HEIGHT][1 + KERNEL_WIDTH]; // 开奇数内存会出错
    __shared__ float s_in[(BLOCK_HEIGHT + KERNEL_HEIGHT) * 2][(BLOCK_WIDTH + KERNEL_WIDTH) * 2];       // 要满足修正的尺寸
    float load_reg[4];

    // 当前block的起始位置
    // int begin_pos = (blockIdx.y + threadIdx.y) * BLOCK_HEIGHT + (blockIdx.x) * BLOCK_WIDTH + threadIdx.x;
    //记录in矩阵的起始位置，他是根据in矩阵进行划分的
    int begin_pos = blockIdx.y * blockDim.y * inputColSize + blockIdx.x * BLOCK_WIDTH;

    int single_trans_ele_num = 4;                               // 线程一次转移的数据数
    int cur_in_block_height = BLOCK_HEIGHT + KERNEL_HEIGHT - 1, // 读入in的block height，读入in的block的row大小
        cur_in_block_width = BLOCK_WIDTH + KERNEL_WIDTH - 1,    // 读入in的block width，读入in的block的col大小
        in_tile_thread_per_row,                                 
        in_tile_row_start,                                      
        in_tile_col,                                            
        in_tile_row_stride;                                    

    // 如果是in边缘的block，需要多读几个数据，相当于处理边界情况
    if (blockIdx.y == row_boundary)
    {
        cur_in_block_height = BLOCK_HEIGHT + row_edge + kernelH - 1;
    }
    if (blockIdx.x == col_boundary)
    {
        cur_in_block_width = BLOCK_WIDTH + col_edge + kernelW - 1;
    }

    in_tile_thread_per_row = cur_in_block_width / single_trans_ele_num; //每个线程读取single_trans_ele_num个数据，则一行需要的线程数
    in_tile_row_start = tid / in_tile_thread_per_row; //就是说这个tid对应的行是多少，我理解的tile是一行
    in_tile_col = tid % in_tile_thread_per_row * single_trans_ele_num; // 获得这个tid对应的这行第几个然后*4就知道他从哪一列开始读取
    in_tile_row_stride = thread_num_per_block / in_tile_thread_per_row; // 每个thread需要跳跃的大小

    // 下方都是读取第一个channel的数据
    for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
        i += in_tile_row_stride)
    {
        // if (blockIdx.y == 0 && blockIdx.x == 0)
        // {
        //     printf("%d (%d %d) %d %d\n", tid, in_tile_row_start + i, in_tile_col, cur_in_block_height, cur_in_block_width);
        // }
        FETCH_FLOAT4(load_reg[0]) =
            FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)]);
        s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
        s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
        s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
        s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
        if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
            cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) // 余量不足一次转移数
        {
            for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
            {
                s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, inputColSize)];
            }
        }
    }

   
    if ( threadIdx.y < KERNEL_HEIGHT && threadIdx.x == 0)
    {
        for (int j = 0; j < KERNEL_WIDTH; j++)
        {
            s_kernel[threadIdx.y][j] = kernel[OFFSET(threadIdx.y, j, KERNEL_WIDTH)];
        }
    }

    __syncthreads();
   


    // 逐个channel计算 一个线程负责block中的一个元素计算
 
    int cur_out_block_height = blockDim.y,
        cur_out_block_width = blockDim.x,  
        out_tile_thread_per_row,             
        out_tile_row_start,                  
        out_tile_col,                       
        out_tile_row_stride;                 
    if (blockIdx.y == row_boundary)
    {
        cur_out_block_height = BLOCK_HEIGHT + row_edge;
    }
    if (blockIdx.x == col_boundary)
    {
        cur_out_block_width = BLOCK_WIDTH + col_edge;
    }

    out_tile_thread_per_row = cur_out_block_width ;
    out_tile_row_start = tid / out_tile_thread_per_row;
    out_tile_col = tid % out_tile_thread_per_row ;
    out_tile_row_stride = thread_num_per_block / out_tile_thread_per_row;

    float tmp[temp_size]; 
    for (int i = 0; i < temp_size; i++)
        tmp[i] = 0;

    int out_pos, temp_pos;

    for (int oc = 0; oc < outputChannel; oc++)
    {
        for (int ic = 0; ic < inputChannel; ic++)
        {
            // i,j 是相当于当前block起始位置而言
            // 用ic的每个block去对oc的kernel进行计算
            for (int i = 0; i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height;
                i += out_tile_row_stride)
            {
                
                // 计算线程负责的元素 同一个oc的缓存顺序排列
                // 不同oc偏移一个cur_out_block_height / out_tile_row_stride + 1的位置
                temp_pos = i / out_tile_row_stride +
                    oc * (cur_out_block_height / out_tile_row_stride + 1);
                for (int ii = 0; ii < KERNEL_HEIGHT; ii++)
                {
                    for (int jj = 0; jj < KERNEL_WIDTH; jj++) // 更换的是SMEM中的内容，相对位置不变
                    {
                        tmp[temp_pos] += s_in[out_tile_row_start + i + ii][out_tile_col + jj] * s_kernel[ii][jj];
                    }
                }
                
            }
            // 读取下一个in channel和对应kernel的数据
            if (ic + 1 < inputChannel)
            {
                for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
                    i += in_tile_row_stride)
                {
                    FETCH_FLOAT4(load_reg[0]) =
                        FETCH_FLOAT4(in[begin_pos + (ic + 1) * inputRowSize * inputColSize + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)]);
                    s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
                    s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
                    s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
                    s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
                    if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
                        in_tile_col + 1 * single_trans_ele_num < cur_in_block_width)
                    {
                        for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
                        {
                            s_in[in_tile_row_start + i][j] = in[begin_pos + (ic + 1) * inputRowSize * inputColSize + OFFSET(in_tile_row_start + i, j, inputColSize)];
                        }
                    }
                }
                if ( threadIdx.y < KERNEL_HEIGHT && threadIdx.x == 0)
                {
                    for (int j = 0; j < KERNEL_WIDTH; j++)
                    {
                        s_kernel[threadIdx.y][j] = kernel[(oc * inputChannel + ic + 1) * kernelH * kernelW + OFFSET(threadIdx.y, j, KERNEL_WIDTH)];
                    }
                }
            }

            __syncthreads();

        }
        // 读取下一个kernel channel数据
        if (oc + 1 < outputChannel)
        {
            if ( threadIdx.y < KERNEL_HEIGHT && threadIdx.x == 0)
            {
                for (int j = 0; j < KERNEL_WIDTH; j++)
                {
                    s_kernel[threadIdx.y][j] = kernel[(oc + 1) * inputChannel * kernelH * kernelW + OFFSET(threadIdx.y, j, KERNEL_WIDTH)];
                }
            }
        }
        __syncthreads();
        // 写回 利用线程id计算写回位置
        int i = 0;
        while (i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height)
        {
                out_pos = oc * outputRowSize * outputColSize +
                    blockIdx.y * BLOCK_HEIGHT * outputColSize + blockIdx.x * BLOCK_WIDTH +
                    OFFSET(out_tile_row_start + i, out_tile_col, outputColSize);
                temp_pos = i / out_tile_row_stride +
                    oc * (cur_out_block_height / out_tile_row_stride + 1);
                out[out_pos] = tmp[temp_pos] + kernelBias[oc];
            
            i += out_tile_row_stride;
        }
        // 读取下一个in channel数据
        for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height;
            i += in_tile_row_stride)
        {

            FETCH_FLOAT4(load_reg[0]) =
                FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)]);
            s_in[in_tile_row_start + i][in_tile_col] = load_reg[0];
            s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
            s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
            s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
            if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width &&
                cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) 
            {
                for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++)
                {
                    s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, inputColSize)];
                }
            }
        }
    }
}

void conv2d(float*  d_input, int inputRowSize, int inputColSize, int inputChannel, \
    float*  d_kernel, float* d_kernelBias, int kernelRowSize, int kernelColSize, \
    float* d_output, int outputRowSize, int outputColSize, int outputChannel, cudaStream_t stream)
{
#if 1
    // float * d_kernel, *d_kernelBias;
    // int inputSize = inputChannel * inputRowSize * inputColSize * sizeof(float);
    // int outputSize = outputChannel * outputRowSize * outputRowSize * sizeof(float);
    // int kernelSize = kernelRowSize * kernelColSize * inputChannel * outputChannel * sizeof(float);
    // int kernelBiasSize = outputChannel * sizeof(float);
    dim3 dimGrid(outputColSize / BLOCK_WIDTH, outputRowSize / BLOCK_HEIGHT);
    dim3 dimBlock(BLOCK_WIDTH , BLOCK_HEIGHT );
    // checkCudaErrors(cudaMalloc(&d_input, inputSize));
    // checkCudaErrors(cudaMalloc(&d_output, outputSize));
    // checkCudaErrors(cudaMalloc(&d_kernel, kernelSize));
    // checkCudaErrors(cudaMalloc(&d_kernelBias, kernelBiasSize));
    
    // checkCudaErrors(cudaMemcpy(d_input, input.data(), inputSize, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_kernelBias, kernelBias.data(), kernelBiasSize, cudaMemcpyHostToDevice));

    _conv2d<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT, KERNEL_WIDTH, MALLOC_TEMP_SIZE>
        << <dimGrid, dimBlock, 0, stream >> > (d_input, d_output, d_kernel, d_kernelBias,
            N, inputChannel, inputRowSize, inputColSize, outputChannel, outputRowSize, outputColSize, kernelRowSize, kernelColSize);
    // cudaDeviceSynchronize();
    // checkCudaErrors(cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));

    //free
    // checkCudaErrors(cudaFree(d_input));
    // checkCudaErrors(cudaFree(d_output));
    // checkCudaErrors(cudaFree(d_kernel));
    // checkCudaErrors(cudaFree(d_kernelBias));
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
// 读取MNIST数据集
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << path << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
        ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
        ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
        ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
        ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);
    // std::cout << "magic_number " << magic_number << "num_images" << num_images << "num_rows" << num_rows << "num_cols" << num_cols << std::endl;
    int image_size = num_rows * num_cols;
    std::vector<std::vector<float>> images(num_images, std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));

            images[i][j] = static_cast<float>(pixel) / 255.0f;
            images[i][j] = 2 * images[i][j] - 1;
            // if (i == 0)
            // {
            //     std::cout << static_cast<float>(pixel) << " ";
            // }
        }
    }

    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!" << path << std::endl;
        return {};
    }

    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_items, sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
        ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
        ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

// 读取模型参数
std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param) {
        params.push_back(param);
    }
    return params;
}

// 范例kernel函数，无实际作用
__global__ void add_arrays(int* a, int* b, int* c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void printTensor(std::vector<float> A, int rowS, int colS, int chaS)
{
    for (int c = 0; c < chaS; c++)
    {

        std::cout << "channel : " << c << std::endl;
        for (int i = 0; i < rowS; i++)
        {

            for (int j = 0; j < colS; j++)
            {
                std::cout << A[c * rowS * colS + i * colS + j] << " ";
            }
            std::cout << std::endl;
        }
    }

}
void poolReluConv1(float* input, int inputRowSize, int inputColSize, \
    float* kernelWeight, int inputChannel, int outputChannel, int kernelConv1Size, \
    float* kernelConv1Bias, int kernelMaxPoolSize, \
    float* output, int outputRowSize, int outputColSize, float* outputTmp, cudaStream_t stream)
{
    // std::vector<float> outputTmp((inputRowSize - kernelConv1Size + 1) * (inputRowSize - kernelConv1Size + 1) * outputChannel, 0);
    // float* outputTmp;
    // cudaMalloc(&outputTmp, sizeof(float) * (inputRowSize - kernelConv1Size + 1) * (inputRowSize - kernelConv1Size + 1) * outputChannel);
    conv2d(input, inputRowSize, inputColSize, inputChannel, \
        kernelWeight, kernelConv1Bias, kernelConv1Size, kernelConv1Size, \
        outputTmp, inputRowSize - kernelConv1Size + 1, inputColSize - kernelConv1Size + 1, outputChannel, stream);
    
    inputRowSize = inputRowSize - kernelConv1Size + 1;
    inputColSize = inputColSize - kernelConv1Size + 1;
    outputRowSize = inputRowSize / kernelMaxPoolSize;
    outputColSize = inputColSize / kernelMaxPoolSize;
    outputChannel = outputChannel;
    inputChannel = outputChannel;
    // printTensor(outputTmp, 24, 24, 6);
    reluMaxPool(outputTmp, inputRowSize, inputColSize, inputChannel, \
        kernelMaxPoolSize, kernelMaxPoolSize, \
        output, outputRowSize, outputColSize, outputChannel, stream);
    // cudaFree(outputTmp);
    // printTensor(output, 12, 12, 6);

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

__global__ void reluGemv( float* __restrict__ A, float* __restrict__ ABias, float* __restrict__ x, float* __restrict__ y, const int M,const int N) {
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
        // if (tx == 0) printf("iter : %d, N %d\n", kIteration, N);
        if (kIteration == 0) kIteration = 1;
        A = &A[current_row * N];
#pragma unroll
        for (int i = 0; i < kIteration; i++) {
            int current_col_vec = (i * warp_size + laneId);
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0){
            res += ABias[current_row];
            if (res >= 0)
                y[current_row] = res;
            else
                y[current_row] = 0;
        }


    }
}

__global__ void reluGemv_final( float* __restrict__ A, float* __restrict__ ABias, float* __restrict__ x, float* __restrict__ y, const int M,const int N,
                                int* predict, int idx) {
    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = blockIdx.y * blockDim.y + blockIdx.x * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_size = 32;
    int laneId = tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if (current_row < M) {
        float res = 0;
        int kIteration = (N / warp_size) / 4;
        // if (tx == 0) printf("iter : %d, N %d\n", kIteration, N);
        if (kIteration == 0) kIteration = 1;
        A = &A[current_row * N];
#pragma unroll
        for (int i = 0; i < kIteration; i++) {
            int current_col_vec = (i * warp_size + laneId);
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0){
            res += ABias[current_row];
            if (res >= 0)
                y[current_row] = res;
            else
                y[current_row] = 0;
        }
        if (tid == 0)
        {
            int max_idx = 0, tmp_max = 0;
            for (int i = 0; i < 10; i ++ )
                if (tmp_max < y[i])
                {
                    tmp_max = y[i];
                    max_idx = i;
                }
            // printf("--------maxidx = %d\n", max_idx);
            predict[idx] = max_idx;
        }

    }
}



void reluSPMV(float* d_input, int inputRowSize, \
    float* d_kernel, int kernelRowSize, int kernelColSize, \
    float* d_kernelBias,\
    float* d_output, int outputRowSize, cudaStream_t stream)
{
    #if 1

    dim3 dimGrid((kernelRowSize + 3) / 4);
    dim3 dimBlock(32, 4);
    reluGemv<< < dimGrid, dimBlock, 0, stream>> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize);

    
#else
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


void reluSPMV_final(float* d_input, int inputRowSize, \
    float* d_kernel, int kernelRowSize, int kernelColSize, \
    float* d_kernelBias,\
    float* d_output, int outputRowSize, int* predict, int idx, cudaStream_t stream)
{
    #if 1

    dim3 dimGrid((kernelRowSize + 3) / 4);
    dim3 dimBlock(32, 4);
    reluGemv_final<< < dimGrid, dimBlock, 0, stream >> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize, predict, idx);

    
#else
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

int maxT(std::vector<float> A)
{
    int tmp = A[0], idx = 0;
    for (int i = 0; i < 10; i++)
    {
        if (tmp < A[i])
        {
            tmp = A[i], idx = i;
        }
    }
    return idx;

}
// #define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

__device__ void warpReduce(volatile float* cache, unsigned int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce(int *predict,int *labels, int *sum, int N){
    __shared__ float sdata[256];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // printf("predict[i] =%d  predict[i + blockDim.x] %d\n", predict[i], predict[i + blockDim.x]);
    if (i < N)
        sdata[tid] = (predict[i] == labels[i]) ;
    if (i + blockDim.x < N)
        sdata[tid] += (predict[i + blockDim.x] == labels[i + blockDim.x]);
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) sum[blockIdx.x] = sdata[0];
}

int check(int* d_predict, int* d_labels, int N)
{    
    int THREAD_PER_BLOCK = 256;
    int NUM_PER_BLOCK = 2* 256;
    // printf("N %d\n", N);
    int block_num = (N + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    int *d_sum, *sum = (int*)malloc(sizeof(int) * block_num);

    // printf("N %d. block_num %d\n", N, block_num);
    cudaMalloc(&d_sum, block_num * sizeof(int));
    
    dim3 Grid( block_num, 1);
    dim3 Block( THREAD_PER_BLOCK, 1);

    reduce<<<Grid,Block>>>(d_predict, d_labels, d_sum, N);
    cudaMemcpy(sum, d_sum, block_num*sizeof(int),cudaMemcpyDeviceToHost);
    int ans = 0;
    for (int i = 0; i < block_num; i ++ )
    {
        ans += sum[i];
        // printf("sum[%d] = %d\n", i, sum[i]);
    }
    return ans;
    
}
void init_ij(std::vector<float>& A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
                A[i * n * m + j * m + k] = k + j;
        }
}
int main(int argc, char* argv[]) {
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    // cout << dir;

    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
#if 0
    auto images = read_mnist_images(dir + "/../../data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    // 读取测试集标签
    auto labels = read_mnist_labels(dir + "/../../data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    // 读取模型参数
    auto conv1_weight = read_param(dir + "/conv1.weight.txt");
    auto conv1_bias = read_param(dir + "/conv1.bias.txt");
    auto conv2_weight = read_param(dir + "/conv2.weight.txt");
    auto conv2_bias = read_param(dir + "/conv2.bias.txt");
    auto fc1_weight = read_param(dir + "/fc1.weight.txt");
    auto fc1_bias = read_param(dir + "/fc1.bias.txt");
    auto fc2_weight = read_param(dir + "/fc2.weight.txt");
    auto fc2_bias = read_param(dir + "/fc2.bias.txt");
    auto fc3_weight = read_param(dir + "/fc3.weight.txt");
    auto fc3_bias = read_param(dir + "/fc3.bias.txt");
#else
    auto images = read_mnist_images(dir + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    // 读取测试集标签
    auto labels = read_mnist_labels(dir + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    // 读取模型参数
    // // std::cout << dir << std::endl;
    auto conv1_weight = read_param(dir + "/conv1.weight_epoch400.txt");
    auto conv1_bias = read_param(dir + "/conv1.bias_epoch400.txt");
    auto conv2_weight = read_param(dir + "/conv2.weight_epoch400.txt");
    auto conv2_bias = read_param(dir + "/conv2.bias_epoch400.txt");
    auto fc1_weight = read_param(dir + "/fc1.weight_epoch400.txt");
    auto fc1_bias = read_param(dir + "/fc1.bias_epoch400.txt");
    auto fc2_weight = read_param(dir + "/fc2.weight_epoch400.txt");
    auto fc2_bias = read_param(dir + "/fc2.bias_epoch400.txt");
    auto fc3_weight = read_param(dir + "/fc3.weight_epoch400.txt");
    auto fc3_bias = read_param(dir + "/fc3.bias_epoch400.txt");
    // auto conv1_weight = read_param(dir + "/conv1.weight.txt");
    // auto conv1_bias = read_param(dir + "/conv1.bias.txt");
    // auto conv2_weight = read_param(dir + "/conv2.weight.txt");
    // auto conv2_bias = read_param(dir + "/conv2.bias.txt");
    // auto fc1_weight = read_param(dir + "/fc1.weight.txt");
    // auto fc1_bias = read_param(dir + "/fc1.bias.txt");
    // auto fc2_weight = read_param(dir + "/fc2.weight.txt");
    // auto fc2_bias = read_param(dir + "/fc2.bias.txt");
    // auto fc3_weight = read_param(dir + "/fc3.weight.txt");
    // auto fc3_bias = read_param(dir + "/fc3.bias.txt");
#endif
    // 打印每一个标签，仅用于调试！

    // for (const auto& label : labels) {
    //     std::cout << label << " ";
    // }
    // std::cout<<std::endl;


    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    // 进行推理
    // std::cout << images.size() << std::endl; // 1w张图
    // std::cout << images[0].size() << std::endl; // 28 * 28 = 784

    // 参数加载
    // std::cout << fc3_bias.size() << std::endl;
        // std::vector<float> output1(6 * 24 * 24, 0);
    float* d_output1, *d_output2, *d_output3, *d_output4, *d_output5, *d_input;
    float* d_conv1_weight, *d_conv1_bias, *d_conv2_weight, *d_conv2_bias, *d_fc1_weight,
        *d_fc1_bias, *d_fc2_weight, *d_fc2_bias, *d_fc3_weight, *d_fc3_bias;
    float* outputTmp;
    int *d_predict, *d_labels;
    int *predict = (int*)malloc(sizeof(int) * labels.size());
    
    const int nStreams = 1; 
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    int d_input_size = 1 * 28 * 28 * sizeof(float);
    int d_output1_size = 6 * 24 * 24 * sizeof(float);
    int d_output2_size = 6 * 24 * 24 * sizeof(float);
    int d_output3_size = 120 * sizeof(float);
    int d_output4_size = 84 * sizeof(float);
    int d_output5_size = 10 * sizeof(float);
    int outputTmp_size = sizeof(float) * (24) * (24) * 16;
    // int d_input_size = 1 * 28 * 28 ;
    // int d_output1_size = 6 * 24 * 24 ;
    // int d_output2_size = 6 * 24 * 24 ;
    // int d_output3_size = 120 ;
    // int d_output4_size = 84 ;
    // int d_output5_size = 10 ;
    checkCudaErrors(cudaMalloc(&outputTmp, sizeof(float) * (24) * (24) * 16 * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_input, 1 * 28 * 28 * sizeof(float) * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_output1, 6 * 24 * 24 * sizeof(float) * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_output2, 6 * 24 * 24 * sizeof(float) * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_output3, 120 * sizeof(float) * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_output4, 84 * sizeof(float) * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_output5, 10 * sizeof(float) * nStreams * 4));
    checkCudaErrors(cudaMalloc(&d_predict, sizeof(int) * labels.size()));
    checkCudaErrors(cudaMalloc(&d_labels, sizeof(int) * labels.size()));
    checkCudaErrors(cudaMalloc(&d_conv1_weight, conv1_weight.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_bias, conv1_bias.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_weight, conv2_weight.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_bias, conv2_bias.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_weight, fc1_weight.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_bias, fc1_bias.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_weight, fc2_weight.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_bias, fc2_bias.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_weight, fc3_weight.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_bias, fc3_bias.size() * sizeof(float)));
    std::vector<float> output2(6 * 24 * 24, 0);
    std::vector<float> output3(120, 0);
    std::vector<float> output4(84, 0);
    std::vector<float> output5(10, 0);
    std::vector<float> input(28 * 28, 0);
    // init_ij(input, 28, 28, 1);

    checkCudaErrors(cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_weight, conv1_weight.data(), sizeof(float) * conv1_weight.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_bias, conv1_bias.data(), sizeof(float) * conv1_bias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_weight, conv2_weight.data(), sizeof(float) * conv2_weight.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_bias, conv2_bias.data(), sizeof(float) * conv2_bias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_weight, fc1_weight.data(), sizeof(float) * fc1_weight.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_weight, fc2_weight.data(), sizeof(float) * fc2_weight.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_weight, fc3_weight.data(), sizeof(float) * fc3_weight.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_bias, fc1_bias.data(), sizeof(float) * fc1_bias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_bias, fc2_bias.data(), sizeof(float) * fc2_bias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_bias, fc3_bias.data(), sizeof(float) * fc3_bias.size(), cudaMemcpyHostToDevice));

    int sum = 0;
    for (int t = 0; t < images.size(); t++) {
        // TODO ...在这里实现利用CUDA对图片进行深度学习的推理过程，当然，你也可以改进for循环以使用batch推理提速...

        // 打印每一张图片，仅用于调试！

        // for(int i = 0;i <28; i++)
        // {
        //     for(int j = 0; j<28; j++)
        //     {
        //         std::cout << images[t][i*28 + j] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        //conv2d1 input_channal = 1, output_channal = 6, kernel_size = 5 
        // TODO : 对每一个images[t]进行推理，这个是28*28的矩阵

        // x = self.pool(F.relu(self.conv1(x))), x = images[t] conv1 = nn.Conv2d(1, 6, 5)
        //pool = nn.MaxPool2d(2, 2)
#if 1

    // printf("input:\n");
    // printTensor(input, 28, 28, 1);
    int stream_tid = t % nStreams;
    cudaMemcpyAsync(d_input + d_input_size * stream_tid, images[t].data(), sizeof(float) * 28 * 28, cudaMemcpyHostToDevice, streams[stream_tid]);

    poolReluConv1(d_input + d_input_size * stream_tid, 28, 28, \
        d_conv1_weight, 1, 6, 5, \
        d_conv1_bias, 2, \
        d_output1 + d_output1_size * stream_tid, 12, 12,
        outputTmp + outputTmp_size * stream_tid,
        streams[stream_tid]);

    // printf("--------------output1--------------\n");
    // printTensor(output1, 12, 12, 6);

    poolReluConv1(d_output1 + d_output1_size * stream_tid, 12, 12, \
        d_conv2_weight, 6, 16, 5, \
        d_conv2_bias, 2, \
        d_output2 + d_output2_size * stream_tid, 4, 4,
        outputTmp + outputTmp_size * stream_tid,
        streams[stream_tid]);
    // cudaMemcpy(output2.data(), d_output2, sizeof(float) * 6 * 24 * 24, cudaMemcpyDeviceToHost);

    // printf("output2\n");
    // printTensor(output2, 4, 4, 16);

    reluSPMV(d_output2 + d_output2_size * stream_tid, 256, \
        d_fc1_weight, 120, 256, \
        d_fc1_bias, \
        d_output3 + d_output3_size * stream_tid, 120,
        streams[stream_tid]);

    // printf("output3\n");
    // printTensor(output3, 120, 1, 1);

    reluSPMV(d_output3 + d_output3_size * stream_tid, 120, \
        d_fc2_weight, 84, 120, \
        d_fc2_bias, \
        d_output4 + d_output4_size * stream_tid, 84,
        streams[stream_tid]);
    // printf("output4\n");
    // printTensor(output4, 84, 1, 1);

    reluSPMV_final(d_output4 + d_output4_size * stream_tid, 84, \
        d_fc3_weight, 10, 84, \
        d_fc3_bias, \
        d_output5 + d_output5_size * stream_tid, 10,
        d_predict, t,
        streams[stream_tid]);
    // if (t % nStreams == 0)
    //     for (int i = 0; i < nStreams; i ++ )
    //         cudaStreamSynchronize(streams[i]);
    // cudaMemcpy(output5.data(), d_output5, sizeof(float) *10, cudaMemcpyDeviceToHost);

    // printf("output5\n");
    // printTensor(output5, 10, 1, 1);


    // printf("\noutput2:\n");
    // printTensor(output5, 10, 1, 1);

        
        // if (labels[t] == maxT(output5))
        //     sum++;
#endif
        // std::cout << "real: " << labels[t]<< ", predict : "<<  maxT(output5) << std::endl;
    }

#if 0
    
    printf("input:\n");
    printTensor(input, 28, 28, 1);
    poolReluConv1(d_input, 28, 28, \
        d_conv1_weight, 1, 6, 5, \
        d_conv1_bias, 2, \
        d_output1, 12, 12, 
        outputTmp);

    // printf("--------------output1--------------\n");
    // printTensor(output1, 12, 12, 6);

    poolReluConv1(d_output1, 12, 12, \
        d_conv2_weight, 6, 16, 5, \
        d_conv2_bias, 2, \
        d_output2, 4, 4,
        outputTmp);
    // cudaMemcpy(output2.data(), d_output2, sizeof(float) * 6 * 24 * 24, cudaMemcpyDeviceToHost);

    // printf("output2\n");
    // printTensor(output2, 4, 4, 16);

    reluSPMV(d_output2, 256, \
        d_fc1_weight, 120, 256, \
        d_fc1_bias, \
        d_output3, 120);

    // printf("output3\n");
    // printTensor(output3, 120, 1, 1);

    reluSPMV(d_output3, 120, \
        d_fc2_weight, 84, 120, \
        d_fc2_bias, \
        d_output4, 84);
    // printf("output4\n");
    // printTensor(output4, 84, 1, 1);

    reluSPMV_final(d_output4, 84, \
        d_fc3_weight, 10, 84, \
        d_fc3_bias, \
        d_output5, 10,
        d_predict, 0);
    

   
    // printf("pre %d\n", tmp_pre[0]);
    // printf("output5\n");
    // printTensor(output5, 10, 1, 1);


    // printf("\noutput2:\n");
    // printTensor(output5, 10, 1, 1);


#endif
    // cudaDeviceSynchronize();
    // for (int i = 0; i < nStreams; i ++ )
    //     cudaStreamSynchronize(streams[i]);
    sum = check(d_predict, d_labels, labels.size());
    // cudaMemcpy(predict, d_predict, sizeof(int) *labels.size(), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaFree(d_conv1_weight));
    checkCudaErrors(cudaFree(d_conv1_bias));
    checkCudaErrors(cudaFree(d_conv2_weight));
    checkCudaErrors(cudaFree(d_conv2_bias));
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output1));
    checkCudaErrors(cudaFree(d_output2));
    checkCudaErrors(cudaFree(d_output3));
    checkCudaErrors(cudaFree(d_output4));
    checkCudaErrors(cudaFree(d_output5));
    checkCudaErrors(cudaFree(d_fc1_weight));
    checkCudaErrors(cudaFree(d_fc1_bias));
    checkCudaErrors(cudaFree(d_fc2_weight));
    checkCudaErrors(cudaFree(d_fc2_bias));
    checkCudaErrors(cudaFree(d_fc3_weight));
    checkCudaErrors(cudaFree(d_fc3_bias));
    checkCudaErrors(cudaFree(outputTmp));
    checkCudaErrors(cudaFree(d_predict));
    checkCudaErrors(cudaFree(d_labels));
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // printf("sum = %d\n", sum);

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(2) << diff.count() << ":" << std::setprecision(4) << (float)sum / (float)images.size() << std::endl;
    // std::cout << std::fixed << std::setprecision(2) << diff.count() << ":0.0001";

    return 0;
}
