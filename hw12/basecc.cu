// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

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
const int N = 1;
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
    int threadNumPerBlock = blockDim.x * blockDim.y;
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
    if (threadIdx.x == 0 && threadIdx.y == 0)
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
    in_tile_row_stride = threadNumPerBlock / in_tile_thread_per_row;
#ifdef DEBUG
    if (threadIdx.x == 0 && threadIdx.y == 0)
        printf("in_per_row = %d, in_tile_row_stride = %d, cur_in_block_height = %d ,cur_in_block_width = %d\n",
            in_tile_thread_per_row, in_tile_row_stride, cur_in_block_height, cur_in_block_width);
#endif
    // 预取数据
    for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height; i += in_tile_row_stride)
    {
        FETCH_FLOAT4(load_reg[0]) = FETCH_FLOAT4(input[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)]);
        if (begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize) >= inputRowSize * inputColSize * inputChannel)
        {

            // printf("out the boundary %d \n", 4 * (begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)));
            continue;
        }
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
                if (begin_pos + (in_tile_row_start + i) * inputColSize + j >= inputRowSize * inputColSize * inputChannel)
                    continue;

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
        printf("cur_out_block_height %d cur_out_block_width %d\n", cur_out_block_height, cur_out_block_width);
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
                if (((begin_pos + (oc + 1) * inputColSize * inputRowSize + (in_tile_row_start + i) * inputColSize + in_tile_col)) >= inputRowSize * inputColSize * inputChannel)
                {
                    // printf("out the boundary %d of %d\n", (begin_pos + (oc + 1) * inputColSize * inputRowSize + (in_tile_row_start + i) * inputColSize + in_tile_col), inputRowSize * inputColSize * inputChannel);
                    continue;
                }
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
                        if ((begin_pos + (oc + 1) * inputRowSize * inputColSize + (in_tile_row_start + i) * inputColSize + k) >= inputRowSize * inputColSize * inputChannel)
                            continue;

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
            if (out_pos >= outputRowSize * outputChannel * outputColSize) continue;
            output[out_pos] = tmp[tmp_pos];
#ifdef DEBUG
            printf("tmp[%d] = %f, output[%d]\n", tmp_pos, tmp[tmp_pos], out_pos);
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
    _reluMaxPoll< BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_SIZE, TMP_SIZE> << <grid, block, 2000, stream >> > (d_input, d_output, inputChannel, inputRowSize, inputColSize,
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

// 读取MNIST数据集
std::vector<float> read_mnist_images(const std::string& path) {
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
    std::vector<float> images(num_images * 28 * 28, 0);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));

            images[i * 28 * 28 + j] = static_cast<float>(pixel) / 255.0f;
            images[i * 28 * 28 + j] = 2 * images[i* 28 * 28 + j] - 1;
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

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

__global__ void reluGemv(float* __restrict__ A, float* __restrict__ ABias, float* __restrict__ x, float* __restrict__ y, const int M, const int N) {
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
            if (current_col_vec * 4 >= N) continue;
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0) {
            res += ABias[current_row];
            if (res >= 0)
                y[current_row] = res;
            else
                y[current_row] = 0;
        }


    }
}

__global__ void reluGemv_final(float* __restrict__ A, float* __restrict__ ABias, float* __restrict__ x, float* __restrict__ y, const int M, const int N,
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
            if (current_col_vec * 4 >= N) continue;
            float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
            float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
            res += current_val.x * current_x.x;
            res += current_val.y * current_x.y;
            res += current_val.z * current_x.z;
            res += current_val.w * current_x.w;
        }
        res = warpReduceSum<warp_size>(res);
        if (laneId == 0) {
            res += ABias[current_row];
            if (res >= 0)
                y[current_row] = res;
            else
                y[current_row] = 0;
        }
        if (tid == 0)
        {
            int max_idx = 0, tmp_max = 0;
            for (int i = 0; i < 10; i++)
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
    float* d_kernelBias, \
    float* d_output, int outputRowSize, cudaStream_t stream)
{
#if 1

    dim3 dimGrid((kernelRowSize + 3) / 4);
    dim3 dimBlock(32, 4);
    reluGemv << < dimGrid, dimBlock >> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize);


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
    float* d_kernelBias, \
    float* d_output, int outputRowSize, int* predict, int idx, cudaStream_t stream)
{
#if 1

    dim3 dimGrid((kernelRowSize + 3) / 4);
    dim3 dimBlock(32, 4);
    reluGemv_final << < dimGrid, dimBlock >> > (d_kernel, d_kernelBias, d_input, d_output, kernelRowSize, kernelColSize, predict, idx);


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

__device__ void warpReduce(volatile float* cache, unsigned int tid) {
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce(int* predict, int* labels, int* sum, int N) {
    __shared__ float sdata[256];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // printf("predict[i] =%d  predict[i + blockDim.x] %d\n", predict[i], predict[i + blockDim.x]);
    if (i < N)
        sdata[tid] = (predict[i] == labels[i]);
    if (i + blockDim.x < N)
        sdata[tid] += (predict[i + blockDim.x] == labels[i + blockDim.x]);
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
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
    int NUM_PER_BLOCK = 2 * 256;
    // printf("N %d\n", N);
    int block_num = (N + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    int* d_sum, * sum = (int*)malloc(sizeof(int) * block_num);

    // printf("N %d. block_num %d\n", N, block_num);
    cudaMalloc(&d_sum, block_num * sizeof(int));

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce << <Grid, Block >> > (d_predict, d_labels, d_sum, N);
    cudaMemcpy(sum, d_sum, block_num * sizeof(int), cudaMemcpyDeviceToHost);
    int ans = 0;
    for (int i = 0; i < block_num; i++)
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


__global__ void _conv2d(float* input,  float* output_pool, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias,
    int inputChannel, int outputChannel, int inputSize, int kernelSize) //是方形
{ //放入一个channle的大小，每个block处理一个output channel,大小是inputSize, 还是多个output channel？

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

                    tmp_max = max(tmp_max,in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
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
    for (int oc = 0; oc < outputChannel; oc++)
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
    int* predict, int t)
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
    // if (tid == 0)
    // {
    //     printf("------------------------------------y------------------------------------\n");
    //     print_d(x, 256);
    // }
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


__global__ void _lenet_fusion(float* input, float* t_output_pool, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias, float* t_output_pool2, const float* __restrict__ kernel2,
    const float* __restrict__ kernel_bias2,
    float* __restrict__ A,
    float* __restrict__ ABias,
    float* __restrict__ x,
    float* __restrict__ A1,
    float* __restrict__ ABias1,
    float* __restrict__ A2,
    float* __restrict__ ABias2,
    float* __restrict__ y2,
    int* predict, int t) //是方形
{
    t = blockIdx.x;
    // if (threadIdx.x ==0 && threadIdx.y ==0 ) printf("id %d\n", t);

    // printf("111111111 blockDImx.x%d y %d z %d\n", blockDim.x, blockDim.y, blockDim.z);
    input = &input[t * 28 * 28];
    int inputChannel, outputChannel, inputSize, kernelSize;
    inputChannel = 1, outputChannel = 6, inputSize = 28, kernelSize = 5;


    __shared__ float in_s[28][28];
    __shared__ float in_pool_s[28][28];
    __shared__ float ker_s[5][5];

    __shared__ float output_pool[12 * 12 * 6];
    __shared__ float output_pool2[4 * 4 * 16];


    //确定要处理哪个outputchannel, 2d grid
    // int oc = threadIdx.z;

    int outputSize = inputSize - kernelSize + 1;

    int destY = threadIdx.y, destX = threadIdx.x;
    int srcY = destY, srcX = destX;
    for (int oc = 0; oc < outputChannel; oc++)
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
        // 4 * 4 * 16
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
    __syncthreads();
    //------------------------------------------------relu+gemv--------------------------------------------------------------
    //                                                RELU + GEMV
    //------------------------------------------------relu+gemv--------------------------------------------------------------
    
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
        x_s[tid] = output_pool2[tid];
    __syncthreads();
    // if (tid == 0)
    // {
    //     printf("------------------------------------y------------------------------------\n");
    //     print_d(x, 256);
    // }
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
int main(int argc, char* argv[]) {
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    // cout << dir;

    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
#if 1
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

    cudaFree(0);
    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    // 进行推理
    // std::cout << images.size() << std::endl; // 1w张图
    // std::cout << images[0].size() << std::endl; // 28 * 28 = 784

    // 参数加载
    // std::cout << fc3_bias.size() << std::endl;
    float* d_output1, * d_output2, * d_output3, * d_output4, * d_output5, * d_input;
    float* d_conv1_weight, * d_conv1_bias, * d_conv2_weight, * d_conv2_bias, * d_fc1_weight,
        * d_fc1_bias, * d_fc2_weight, * d_fc2_bias, * d_fc3_weight, * d_fc3_bias;
    float* d_outputTmp;
    int* d_predict, * d_labels;
    int* predict = (int*)malloc(sizeof(int) * labels.size());

    int nStreams = 1;
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
    cudaMalloc(&d_outputTmp, sizeof(float) * (24) * (24) * 16 * nStreams * 4);
    cudaMalloc(&d_input, 10000 * 28 * 28 * sizeof(float) * nStreams * 4);
    cudaMalloc(&d_output1, 6 * 24 * 24 * sizeof(float) * nStreams * 4);
    cudaMalloc(&d_output2, 6 * 24 * 24 * sizeof(float) * nStreams * 4);
    cudaMalloc(&d_output3, 120 * sizeof(float) * nStreams * 4);
    cudaMalloc(&d_output4, 84 * sizeof(float) * nStreams * 4);
    cudaMalloc(&d_output5, 10 * sizeof(float) * nStreams * 4);
    cudaMalloc(&d_predict, sizeof(int) * labels.size());
    cudaMalloc(&d_labels, sizeof(int) * labels.size());
    cudaMalloc(&d_conv1_weight, conv1_weight.size() * sizeof(float));
    cudaMalloc(&d_conv1_bias, conv1_bias.size() * sizeof(float));
    cudaMalloc(&d_conv2_weight, conv2_weight.size() * sizeof(float));
    cudaMalloc(&d_conv2_bias, conv2_bias.size() * sizeof(float));
    cudaMalloc(&d_fc1_weight, fc1_weight.size() * sizeof(float));
    cudaMalloc(&d_fc1_bias, fc1_bias.size() * sizeof(float));
    cudaMalloc(&d_fc2_weight, fc2_weight.size() * sizeof(float));
    cudaMalloc(&d_fc2_bias, fc2_bias.size() * sizeof(float));
    cudaMalloc(&d_fc3_weight, fc3_weight.size() * sizeof(float));
    cudaMalloc(&d_fc3_bias, fc3_bias.size() * sizeof(float));
    std::vector<float> output2(16 * 4 * 4, 0);
    std::vector<float> output1(6 * 12 * 12, 0);
    std::vector<float> output3(120, 0);
    std::vector<float> output4(84, 0);
    std::vector<float> output5(10, 0);
    std::vector<float> input(28 * 28, 0), outputTmp1(24 * 24 * 6, 0), outputTmp(8 * 8 * 16, 0);
    
    // init_ij(input, 28, 28, 1);

    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_weight, conv1_weight.data(), sizeof(float) * conv1_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, conv1_bias.data(), sizeof(float) * conv1_bias.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_weight, conv2_weight.data(), sizeof(float) * conv2_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_bias, conv2_bias.data(), sizeof(float) * conv2_bias.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_weight, fc1_weight.data(), sizeof(float) * fc1_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weight, fc2_weight.data(), sizeof(float) * fc2_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_weight, fc3_weight.data(), sizeof(float) * fc3_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, fc1_bias.data(), sizeof(float) * fc1_bias.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, fc2_bias.data(), sizeof(float) * fc2_bias.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_bias, fc3_bias.data(), sizeof(float) * fc3_bias.size(), cudaMemcpyHostToDevice);

    int sum = 0;
    cudaMemcpy(d_input, images.data(), sizeof(float) * 28 * 28 * 10000, cudaMemcpyHostToDevice);
    for (int t = 0; t < 1; t++) {
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

    // printf("input:\n");
    // printTensor(input, 28, 28, 1);
        int stream_tid = t % nStreams;
        // printf("--------------------------------input--------------------------------\n");
        // init_ij(input, 28, 28, 1);
        // printTensor(input, 28, 28, 1);
        // cudaMemcpy(d_input + d_input_size * stream_tid, &images[t * 28 * 28], sizeof(float) * 28 * 28, cudaMemcpyHostToDevice);

        dim3 block(32, 32);
        dim3 grid(10000);
#if 1
        _lenet_fusion << < grid, block >> > (d_input , d_output1 + d_output1_size * stream_tid, d_conv1_weight,
            d_conv1_bias, d_output2 + d_output2_size * stream_tid, d_conv2_weight,
            d_conv2_bias,
            d_fc1_weight, d_fc1_bias,
            d_output2 + d_output2_size * stream_tid,
            d_fc2_weight,
            d_fc2_bias,
            d_fc3_weight,
            d_fc3_bias,
            d_output5 + d_output5_size * stream_tid,
            d_predict, t);
        // cudaDeviceSynchronize();
#else
        _conv2d_fusion << < 1, block >> > (d_input + d_input_size * stream_tid, d_output1 + d_output1_size * stream_tid, d_conv1_weight,
            d_conv1_bias, d_output2 + d_output2_size * stream_tid, d_conv2_weight,
            d_conv2_bias);
        

        
        // cudaMemcpy(output2.data(), d_output2, sizeof(float) * output2.size(), cudaMemcpyDeviceToHost);
        // cudaMemcpy(outputTmp.data(), d_outputTmp, sizeof(float)* outputTmp.size(), cudaMemcpyDeviceToHost);

        // printf("--------------------------------output_conv--------------------------------\n");
        // printTensor(outputTmp, 8, 8, 16);
        // printf("--------------------------------output2_maxpool--------------------------------\n");
        // printTensor(output2, 4, 4, 16);

        relugemv_fusion << < 1, block >> > (d_fc1_weight, d_fc1_bias,
            d_output2 + d_output2_size * stream_tid,
            d_fc2_weight,
            d_fc2_bias,
            d_fc3_weight,
            d_fc3_bias,
            d_output5 + d_output5_size * stream_tid,
            d_predict, t);
        // cudaMemcpy(output5.data(), d_output5, sizeof(float) * output5.size(), cudaMemcpyDeviceToHost);
        // printf("--------------------------------output5--------------------------------\n");
        // printTensor(output5, 10, 1, 1);

   
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
    init_ij(input, 28, 28, 1);
    printTensor(input, 28, 28, 1);
    int stream_tid = 0;
    cudaMemcpyAsync(d_input + d_input_size * stream_tid, input.data(), sizeof(float) * 28 * 28, cudaMemcpyHostToDevice, streams[stream_tid]);
    dim3 block(28, 28);
    dim3 grid(16);
    _conv2d << < grid, block >> > (d_input + d_input_size * stream_tid, d_output1 + d_output1_size * stream_tid, d_conv1_weight,
        d_conv1_bias, 1, 6, 28, 5);
    // checkCudaErrors(cudaMemcpy(d_output_pool, output_pool.data(), output_pool.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaMemcpy(output1.data(), d_output1, sizeof(float) * 6 * 12 * 12, cudaMemcpyDeviceToHost);

    printf("output1\n");
    printTensor(output1, 12, 12, 6);
    _conv2d << < grid, block >> > (d_output1 + d_output1_size * stream_tid, d_output2 + d_output2_size * stream_tid, d_conv2_weight,
        d_conv2_bias, 6, 16, 12, 5);
    cudaDeviceSynchronize();


    cudaMemcpy(output2.data(), d_output2, sizeof(float) * 4 * 4 * 16, cudaMemcpyDeviceToHost);

    printf("output2\n");
    printTensor(output2, 4, 4, 16);

    reluSPMV(d_output2, 256, \
        d_fc1_weight, 120, 256, \
        d_fc1_bias, \
        d_output3, 120, streams[0]);

    // printf("output3\n");
    // printTensor(output3, 120, 1, 1);

    reluSPMV(d_output3, 120, \
        d_fc2_weight, 84, 120, \
        d_fc2_bias, \
        d_output4, 84, streams[0]);
    // printf("output4\n");
    // printTensor(output4, 84, 1, 1);

    reluSPMV(d_output4, 84, \
        d_fc3_weight, 10, 84, \
        d_fc3_bias, \
        d_output5, 10, streams[0]);
    cudaMemcpy(output5.data(), d_output5, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    printTensor(output5, 10, 1, 1);


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
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_weight);
    cudaFree(d_conv2_bias);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_output4);
    cudaFree(d_output5);
    cudaFree(d_fc1_weight);
    cudaFree(d_fc1_bias);
    cudaFree(d_fc2_weight);
    cudaFree(d_fc2_bias);
    cudaFree(d_fc3_weight);
    cudaFree(d_fc3_bias);
    cudaFree(d_outputTmp);
    cudaFree(d_predict);
    cudaFree(d_labels);
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // printf("sum = %d\n", sum);

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << std::setprecision(4) << (float)sum / (float)10000<< std::endl;
    // std::cout << std::fixed << std::setprecision(2) << diff.count() << ":0.0001";

    return 0;
}
