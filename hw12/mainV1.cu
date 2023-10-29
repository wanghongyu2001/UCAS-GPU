// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70
// nvcc test.cu -o train -Xcompiler "-O3 -std=c++14" -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -rdc=true
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

#include <cuda_runtime.h>
#include <cmath>

// #include <device_launch_parameters.h>
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

#if 0
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));

            images[i * 28 * 28 + j] = static_cast<float>(pixel) / 255.0f;
            images[i * 28 * 28 + j] = 2 * images[i * 28 * 28 + j] - 1;
            // if (i == 0)
            // {
            //     std::cout << static_cast<float>(pixel) << " ";
            // }
        }
    }
#else
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0, pos = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            int row = j / 28, col = j % 28;
            if (row < 25 && col < 25)
            {
                images[i * 28 * 28 + pos] = static_cast<float>(pixel) / 255.0f;
                images[i * 28 * 28 + pos] = 2 * images[i * 28 * 28 + pos] - 1;
                pos++;
            }
            // images[i * 28 * 28 + j] = static_cast<float>(pixel) / 255.0f;
            // images[i * 28 * 28 + j] = 2 * images[i * 28 * 28 + j] - 1;
            // if (i == 0)
            // {
            //     std::cout << static_cast<float>(pixel) << " ";
            // }
        }
    }
#endif
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
    unsigned int mask = 0xfffffff;

    if (WarpSize >= 32)sum += __shfl_down_sync(mask, sum, 16); // 0 + 16, 1 + 17, 2 + 18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(mask, sum, 8);// 0 + 8, 1 + 9, 2 + 10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(mask, sum, 4);// 0 + 4, 1 + 5, 2 + 6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(mask, sum, 2);// 0 + 2, 1 + 3, 4 + 6, 5 + 7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(mask, sum, 1);// 0 + 1, 2 + 3, 4 + 5, etc.
    return sum;
}


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
    cudaFree(d_sum);

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



__device__ void print_d(float* y, int len)
{
    for (int i = 0; i < len; i++)
        printf("%f\n", y[i]);
}

__device__ float _tanh(float x)
{

    return 1 - (2 * (1 / (1 + expf(x * 2))));

}


template<int set_size>
__global__ void _lenet_fusion_new(float* input, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias,
    float* __restrict__ A,
    int* labels, int* sum,
    int set_id) //是方形
{
    const int batch_size = 10000 / set_size;
    int t = blockIdx.x;
    int flag[batch_size];
    for (int b = 0; b < batch_size; b++)
        flag[b] = labels[t + b];
    // clock_t start_conv_time = clock();
    input = &input[(t) * 28 * 28];
    int inputChannel, outputChannel, kernelSize;
    inputChannel = 1, outputChannel = 3, kernelSize = 5;


    __shared__ float in_s[batch_size][25][25];
    __shared__ float in_pool_s[batch_size][108];
    __shared__ float ker_s[5][5];
    int destY = threadIdx.x / 6, destX = threadIdx.x % 6;

    int srcY = destY, srcX = destX;

    for (int b = 0; b < 2; b++)
        for (int i = 0; i < 5; i++)
        {
            int ins_pos = i * 128 + threadIdx.x * 4;
            int pos = b * 28 * 28 + ins_pos;
            float4 loaded_data = reinterpret_cast<float4*>(input + pos)[0];
            in_s[b][ins_pos / 25][(ins_pos++) % 25] = loaded_data.x;
            in_s[b][ins_pos / 25][(ins_pos++) % 25] = loaded_data.y;
            in_s[b][ins_pos / 25][(ins_pos++) % 25] = loaded_data.z;
            in_s[b][ins_pos / 25][ins_pos % 25] = loaded_data.w;

        }

    for (int oc = 0; oc < outputChannel; oc++)
    {

        float tmp_bias = kernel_bias[oc];
        float accum[batch_size], accum1[batch_size];
        for (int b = 0; b < batch_size; b++)
            accum[b] = 0, accum1[b] = 0;


        if (oc == 0 && destY < kernelSize && destX < kernelSize) // 5 不需要多取
        {
            int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                destY * kernelSize + destX;
            ker_s[destY][destX] = kernel[ker_pos];
        }

        // __syncthreads();

        {
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {

                    for (int b = 0; b < batch_size; b++)
                        accum[b] += ker_s[i][j] * in_s[b][srcY * 4 + i][srcX * 4 + j];
                }

            }

            for (int b = 0; b < batch_size; b++)
            {
                int tmp_pos = oc * 6 * 6 + destY * 6 + destX;
                in_pool_s[b][(tmp_pos % 4) * 27 + (tmp_pos / 4)] = (accum[b] + tmp_bias) > 0 ? (accum[b] + tmp_bias) : 0;
            }

        }
        if (destY == 4)
        {
            for (int i = 0; i < kernelSize; i++)
            {
                for (int j = 0; j < kernelSize; j++)
                {
                    for (int b = 0; b < batch_size; b++)
                    {

                        accum1[b] += ker_s[i][j] * in_s[b][5 * 4 + i][srcX * 4 + j];
                    }
                }
            }

            for (int b = 0; b < batch_size; b++)
            {

                int tmp_pos = oc * 6 * 6 + (destY + 1) * 6 + destX;
                in_pool_s[b][(tmp_pos % 4) * 27 + (tmp_pos / 4)] = (accum1[b] + tmp_bias) > 0 ? (accum1[b] + tmp_bias) : 0;
            }
        }
        if (oc < outputChannel - 1 && destY < kernelSize && destX < kernelSize) // 5 不需要多取
        {
            int ker_pos = (oc + 1) * kernelSize * kernelSize * inputChannel +
                destY * kernelSize + destX;
            ker_s[destY][destX] = kernel[ker_pos];
        }
        // __syncthreads();
    }

    //------------------------------------------------relu+gemv--------------------------------------------------------------
    //                                                 GEMV
    //------------------------------------------------relu+gemv--------------------------------------------------------------


    // __syncthreads();
    int  width = 108;
    int tid = threadIdx.x;
    int warp_id = (tid) / 32;
    int thread_warp_id = tid % 32;
    const int warp_size = 32;

    //warp要取的col的start
    int col_vec_start = thread_warp_id;
    float tmp_max[2] = { -1e9, -1e9 };
    int    id[2] = { 0, 0 };
    if (threadIdx.x < 27)
    {
        for (int k = 0; k < 10 && warp_id == 0; k++)
        {
            float tmp[batch_size];
            tmp[0] = 0, tmp[1] = 0;

            int row = k;
            float4 current_val1 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start];

            for (int b = 0; b < batch_size; b++)
            {
                tmp[b] += current_val1.x * in_pool_s[b][(col_vec_start)+0 * 27];
                tmp[b] += current_val1.y * in_pool_s[b][(col_vec_start)+1 * 27];
                tmp[b] += current_val1.z * in_pool_s[b][(col_vec_start)+2 * 27];
                tmp[b] += current_val1.w * in_pool_s[b][(col_vec_start)+3 * 27];
            }
            tmp[0] = warpReduceSum<warp_size>(tmp[0]);
            tmp[1] = warpReduceSum<warp_size>(tmp[1]);

            if (tid == 0)
            {
                if (tmp_max[tid] < tmp[tid])
                {
                    tmp_max[tid] = tmp[tid];
                    id[tid] = row;
                }
                if (tmp_max[tid + 1] < tmp[tid + 1])
                {
                    tmp_max[tid + 1] = tmp[tid + 1];
                    id[tid + 1] = row;
                }
            }
        }
    }

    if (tid == 0)
    {
        int acc = id[0] == flag[0];
        acc += id[1] == flag[1];
        atomicAdd(sum, acc);
    }
}

__global__ void simpleKernel(int* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = data[tid] * 2;
    }
}

void cudaWarpup(int arraySize) {
    int size = arraySize * sizeof(int);

    int* h_data = new int[arraySize];
    for (int i = 0; i < arraySize; ++i) {
        h_data[i] = i;
    }
    int* d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    simpleKernel << <gridSize, blockSize >> > (d_data, arraySize);

    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // 释放内存
    delete[] h_data;
    cudaFree(d_data);
}

int main(int argc, char* argv[]) {
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    // cout << dir;
    int gpuDevice = 1;

    cudaSetDevice(gpuDevice);
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
    auto fc4_weight = read_param(dir + "/fc1.weight.txt");
    auto fc4_bias = read_param(dir + "/fc1.bias.txt");
    float* d_fc4_weight, * d_fc4_bias;
    // printf("fc4 size %d bias %d\n", fc4_weight.size(), fc4_bias.size());
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


    float* d_input;
    float* d_conv1_weight, * d_conv1_bias, * d_conv2_weight, * d_conv2_bias;
    int* d_predict, * d_labels;
    int* predict = (int*)malloc(sizeof(int) * labels.size());
    const int set_size = 10000 / 2;
    int nStreams = 1;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }


    cudaMalloc(&d_fc4_weight, fc4_weight.size() * sizeof(float));
    cudaMalloc(&d_fc4_bias, fc4_bias.size() * sizeof(float));
    cudaMemcpy(d_fc4_weight, fc4_weight.data(), sizeof(float) * fc4_weight.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_input, 2 * 10000 * 28 * 28 * sizeof(float));
    cudaMalloc(&d_predict, sizeof(int) * labels.size());
    cudaMalloc(&d_labels, sizeof(int) * labels.size());
    cudaMalloc(&d_conv1_weight, conv1_weight.size() * sizeof(float));
    cudaMalloc(&d_conv1_bias, conv1_bias.size() * sizeof(float));
    cudaMalloc(&d_conv2_weight, conv2_weight.size() * sizeof(float));
    cudaMalloc(&d_conv2_bias, conv2_bias.size() * sizeof(float));

    // cudaMemcpy(d_input, images.data(), sizeof(float) * images.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_weight, conv1_weight.data(), sizeof(float) * conv1_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, conv1_bias.data(), sizeof(float) * conv1_bias.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_weight, conv2_weight.data(), sizeof(float) * conv2_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv2_bias, conv2_bias.data(), sizeof(float) * conv2_bias.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, &images[0],
        sizeof(float) * images.size(), cudaMemcpyHostToDevice);

    // int sum = 0;

    int N = labels.size();
    int THREAD_PER_BLOCK = 256;
    int NUM_PER_BLOCK = 2 * 256;
    // printf("N %d\n", N);
    int block_num = (N + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK;
    int* d_sum, * just_for_warmup, * sum = (int*)malloc(sizeof(int) * block_num);

    // printf("N %d. block_num %d\n", N, block_num);
    cudaMalloc(&d_sum, block_num * sizeof(int));
    cudaMalloc(&just_for_warmup, block_num * sizeof(int));

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);


    cudaWarpup(1000);
    // 开始计时，使用chrono计时，不支持其它计时方式
    //--------------------------------开始执行--------------------------------
    int t = 0;
    int stream_tid = t % nStreams;
    // dim3 block(7 * 32);
    dim3 block(32);
    // dim3 grid(set_size);
    // set_size = 5000;
    dim3 grid(set_size);
    // printf("set_size %d\n", set_size);


    //仅仅就是warmup，计算的结果放在了just_for_warmup的临时变量里。和后面的计算没关系，统计的时间更接近这个nsight的prof的时间
    _lenet_fusion_new<set_size> << < grid, block >> > (d_input, \
        d_conv1_weight, \
        d_conv1_bias, \
        d_fc4_weight, \
        d_labels, \
        just_for_warmup, \
        t);
    cudaDeviceSynchronize();

    //--------------------------------开始执行--------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    _lenet_fusion_new<set_size> << < grid, block >> > (d_input, \
        d_conv1_weight, \
        d_conv1_bias, \
        d_fc4_weight, \
        d_labels, \
        d_sum, \
        t);


    cudaMemcpy(sum, d_sum, block_num * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    //--------------------------------执行结束--------------------------------
    std::chrono::duration<double> diff = end - start;
    std::cout << std::fixed << std::setprecision(6) << diff.count() << ":" << std::setprecision(4) << (float)sum[0] / (float)10000 << std::endl;

    // cudaMemcpy(predict, d_predict, sizeof(int) *labels.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_fc4_weight);
    cudaFree(d_fc4_bias);
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv2_weight);
    cudaFree(d_conv2_bias);
    cudaFree(d_input);
    cudaFree(d_predict);
    cudaFree(d_labels);
    // 向主机端同步以等待所有异步调用的GPU kernel执行完毕，这句必须要有
    // 结束计时
    // printf("sum = %d\n", sum);

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    // std::cout << std::fixed << std::setprecision(2) << diff.count() << ":0.0001";

    return 0;
}
