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


const int inputRowSize = 12, inputColSize = 12, inputChannel = 6;
const int outputRowSize = 8, outputColSize = 8, outputChannel = 16;
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
#if 1
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

void reluMaxPool(std::vector<float> input, int inputRowSize, int inputColSize, int inputChannel, \
    int kernelRowSize, int kernelColSize, \
    std::vector<float>& output, int outputRowSize, int outputColSize, int outputChannel)
{

    for (int c = 0; c < outputChannel; c++)
    {
        for (int i = 0; i < outputRowSize; i++)
        {
            for (int j = 0; j < outputColSize; j++)
            {
                //maxpool relu + 
                double tmp = 0;
                {
                    for (int row = kernelRowSize * i; row < kernelRowSize * i + kernelRowSize; row++)
                    {
                        for (int col = kernelColSize * j; col < kernelColSize * j + kernelColSize; col++)
                        {
                            if (input[c * inputRowSize * inputColSize + row * inputColSize + col] >= 0)
                                tmp = max(input[c * inputRowSize * inputColSize + row * inputColSize + col], tmp);
                        }
                        // std::cout << tmp << " ";
                    }
                }

                output[c * outputRowSize * outputColSize + i * outputColSize + j] = tmp;

            }
        }
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
void poolReluConv1(std::vector<float> input, int inputRowSize, int inputColSize, \
    std::vector<float> kernelWeight, int inputChannel, int outputChannel, int kernelConv1Size, \
    std::vector<float> kernelConv1Bias, int kernelMaxPoolSize, \
    std::vector<float>& output)
{
    std::vector<float> outputTmp((inputRowSize - kernelConv1Size + 1) * (inputRowSize - kernelConv1Size + 1) * outputChannel, 0);
    conv2d(input, inputRowSize, inputColSize, inputChannel, \
        kernelWeight, kernelConv1Bias, kernelConv1Size, kernelConv1Size, \
        outputTmp, inputRowSize - kernelConv1Size + 1, inputColSize - kernelConv1Size + 1, outputChannel);

    inputRowSize = inputRowSize - kernelConv1Size + 1;
    inputColSize = inputColSize - kernelConv1Size + 1;
    int outputRowSize = inputRowSize / kernelMaxPoolSize;
    int outputColSize = inputColSize / kernelMaxPoolSize;
    outputChannel = outputChannel;
    inputChannel = outputChannel;
    // printTensor(outputTmp, 24, 24, 6);
    reluMaxPool(outputTmp, inputRowSize, inputColSize, inputChannel, \
        kernelMaxPoolSize, kernelMaxPoolSize, \
        output, outputRowSize, outputColSize, outputChannel);
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

// if N>= 128
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


void reluSPMV(std::vector<float> input, int inputRowSize, \
    std::vector<float> kernel, int kernelRowSize, int kernelColSize, \
    std::vector<float> kernelBias,\
    std::vector<float>& output, int outputRowSize)
{
    #if 1
// printf("11111\n");
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
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost));
    
    //cudaFree
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_kernelBias);

    
    
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
        std::vector<float> output1(6 * 24 * 24, 0);
        std::vector<float> output2(6 * 24 * 24, 0);
        std::vector<float> output3(120, 0);
        std::vector<float> output4(84, 0);
        std::vector<float> output5(10, 0);
        // std::vector<float> input(28 * 28, 0);
        // init_ij(input, 28, 28, 1);
        // printf("input:\n");
        // printTensor(input, 28, 28, 1);

        poolReluConv1(images[t], 28, 28, \
            conv1_weight, 1, 6, 5, \
            conv1_bias, 2, \
            output1);

        // printf("output1\n");
        // printTensor(output1, 12, 12, 6);

        poolReluConv1(output1, 12, 12, \
            conv2_weight, 6, 16, 5, \
            conv2_bias, 2, \
            output2);

        // printf("output2\n");
        // printTensor(output2, 4, 4, 16);

        reluSPMV(output2, 256, \
            fc1_weight, 120, 256, \
            fc1_bias, \
            output3, 120);

        // printf("output3\n");
        // printTensor(output3, 120, 1, 1);

        reluSPMV(output3, 120, \
            fc2_weight, 84, 120, \
            fc2_bias, \
            output4, 84);
        // printf("output4\n");
        // printTensor(output4, 84, 1, 1);

        reluSPMV(output4, 84, \
            fc3_weight, 10, 84, \
            fc3_bias, \
            output5, 10);
        // printf("output5\n");
        // printTensor(output5, 10, 1, 1);

        if (labels[t] == maxT(output5))
            sum++;
        // std::cout << "real: " << labels[t]<< ", predict : "<<  maxT(output5) << std::endl;
    }

    // std::vector<float> output1(6 * 24 * 24, 0);
    // std::vector<float> output2(6 * 24 * 24, 0);
    // std::vector<float> output3(120, 0);
    // std::vector<float> output4(84, 0);
    // std::vector<float> output5(10, 0);
    // std::vector<float> input(28 * 28, 0);
    // init_ij(input, 28, 28, 1);
    // printf("input:\n");
    // printTensor(images[0], 28, 28, 1);
    // poolReluConv1(images[0], 28, 28, \
    //     conv1_weight, 1, 6, 5, \
    //     conv1_bias, 2, \
    //     output1);

    // printf("output1\n");
    // printTensor(output1, 12, 12, 6);

    // poolReluConv2(output1, 12, 12, \
    //     conv2_weight, 6, 16, 5, \
    //     conv2_bias, 2, \
    //     output2);

    // printf("output2\n");
    // printTensor(output2, 4, 4, 16);

    // reluSPMV(output2, 256, \
    //     fc1_weight, 120, 256, \
    //     fc1_bias, \
    //     output3, 120);

    // printf("output3\n");
    // printTensor(output3, 120, 1, 1);

    // reluSPMV(output3, 120, \
    //     fc2_weight, 84, 120, \
    //     fc2_bias, \
    //     output4, 84);
    // printf("output4\n");
    // printTensor(output4, 84, 1, 1);

    // reluSPMV(output4, 84, \
    //     fc3_weight, 10, 84, \
    //     fc3_bias, \
    //     output5, 10);
    // printf("output5\n");
    // printTensor(output5, 10, 1, 1);


    // printf("\noutput2:\n");
    // printTensor(output5, 10, 1, 1);
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    cudaDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // printf("sum = %d\n", sum);

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(2) << diff.count() << ":" << std::setprecision(4) << (float)sum / (float)images.size() << std::endl;
    // std::cout << std::fixed << std::setprecision(2) << diff.count() << ":0.0001";

    return 0;
}
