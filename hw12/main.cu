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
            images[i * 28 * 28 + j] = 2 * images[i * 28 * 28 + j] - 1;
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

template<int set_size>
__global__ void _lenet_fusion_new(float* input, const float* __restrict__ kernel,
    const float* __restrict__ kernel_bias, const float* __restrict__ kernel2,
    const float* __restrict__ kernel_bias2,
    float* __restrict__ A,
    int* labels, int* sum,
    int set_id) //是方形
{
    int t = set_id * set_size + blockIdx.x;
    int flag = labels[t];
    // clock_t start_conv_time = clock();
    input = &input[(t) * 28 * 28];
    int inputChannel, outputChannel, inputSize, kernelSize;
    inputChannel = 1, outputChannel = 6, inputSize = 28, kernelSize = 5;


    __shared__ float in_s[6][28][28];
    __shared__ float in_pool_s[28][28];
    __shared__ float ker_s[16][5][5];

    __shared__ float output_pool[6][12][12];
    __shared__ float output_pool2[16 * 4 * 4];

    int outputSize = inputSize - kernelSize + 1;

    int destY = threadIdx.y, destX = threadIdx.x;
    int srcY = destY, srcX = destX;
    for (int ic = 0; ic < inputChannel; ic++)
    {

        if (threadIdx.y < inputSize && threadIdx.x < inputSize)
        {
            int in_pos = ic * inputSize * inputSize + threadIdx.y * inputSize + threadIdx.x;
            in_s[ic][destY][destX] = input[in_pos];

        }
    }
    // __syncthreads();
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
                ker_s[ic][destY][destX] = kernel[ker_pos];
            }
        }
        __syncthreads(); //奇怪，这个同步不能去
        for (int ic = 0; ic < inputChannel; ic++)
        {
            if (destY < kernelSize && destX < kernelSize)
            {
                int ker_pos = oc * kernelSize * kernelSize * inputChannel +
                    ic * kernelSize * kernelSize + destY * kernelSize + destX;
                ker_s[ic][destY][destX] = kernel[ker_pos];
            }


            if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
            {
                for (int i = 0; i < kernelSize; i++)
                {
#pragma unroll
                    for (int j = 0; j < kernelSize; j++)
                    {
                        accum += in_s[ic][srcY + i][srcX + j] * ker_s[ic][i][j];
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
#pragma unroll
                for (int j = 0; j < kernel_pool_size; j++)
                {

                    tmp_max = max(tmp_max, in_pool_s[srcY * kernel_pool_size + i][srcX * kernel_pool_size + j]);
                }
            output_pool[oc][srcY][srcX] = tmp_max >= 0 ? tmp_max : 0;
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
                ker_s[ic][destY][destX] = kernel2[ker_pos];
            }
        }
        __syncthreads(); //奇怪，这个同步不能去
        for (int ic = 0; ic < inputChannel; ic++)
        {
            if (srcY + kernelSize - 1 < inputSize && srcX + kernelSize - 1 < inputSize)
            {
                for (int i = 0; i < kernelSize; i++)
                {
#pragma unroll
                    for (int j = 0; j < kernelSize; j++)
                    {
                        accum += output_pool[ic][srcY + i][srcX + j] * ker_s[ic][i][j];
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
#pragma unroll
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

    // clock_t end_conv_time = clock();
    // clock_t start_fc_time = clock();
    //------------------------------------------------relu+gemv--------------------------------------------------------------
    //                                                 GEMV
    //------------------------------------------------relu+gemv--------------------------------------------------------------

    //一个warp算y的一个元素
    int height = 10, width = 256;
    int warp_id = threadIdx.y;
    int warp_num = blockDim.y;
    const int warp_size = 32;

    //warp要取的col的start
    int col_vec_start = threadIdx.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // __shared__ Arow_s[width];  
    float* x_s = output_pool2;
    __shared__ float out[10];




#pragma unroll
    for (int row = warp_id / 2; row < height; row += warp_num)
    {
        out[row] = 0;
        float tmp = 0;
        col_vec_start += warp_id % 2 * 32;
        float4 current_val1 = reinterpret_cast<float4*>(A)[row * width / 4 + col_vec_start];
        tmp += current_val1.x * x_s[col_vec_start * 4];
        tmp += current_val1.y * x_s[col_vec_start * 4 + 1];
        tmp += current_val1.z * x_s[col_vec_start * 4 + 2];
        tmp += current_val1.w * x_s[col_vec_start * 4 + 3];
        tmp = warpReduceSum<warp_size>(tmp);
        if (threadIdx.x == 0)
        {
            atomicAdd(&out[row], tmp);
        }

    }
    __syncthreads();

    if (tid == 0)
    {
        float tmp_max = -1e9, id = 0;
        for (int i = 0; i < 10; i++)
        {
            if (tmp_max < out[i])
            {
                tmp_max = out[i], id = i;
            }
        }
        int acc = (id == flag);
        atomicAdd(sum, acc);
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
    const int set_size = 10000 / 40;
    int nStreams = 20;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }



    cudaMalloc(&d_fc4_weight, fc4_weight.size() * sizeof(float));
    cudaMalloc(&d_fc4_bias, fc4_bias.size() * sizeof(float));
    cudaMemcpy(d_fc4_weight, fc4_weight.data(), sizeof(float) * fc4_weight.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_input, 10000 * 28 * 28 * sizeof(float));
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
        int* d_sum, * sum = (int*)malloc(sizeof(int) * block_num);

        // printf("N %d. block_num %d\n", N, block_num);
        cudaMalloc(&d_sum, block_num * sizeof(int));

        dim3 Grid(block_num, 1);
        dim3 Block(THREAD_PER_BLOCK, 1);
    
        auto start = std::chrono::high_resolution_clock::now();
    // 开始计时，使用chrono计时，不支持其它计时方式
        for (int t = 0; t < 10000 / set_size; t++) {
        int stream_tid = t % nStreams;
        dim3 block(32, 32);
        dim3 grid(set_size);

        _lenet_fusion_new<set_size> << < grid, block, 400, streams[stream_tid] >> > (d_input,
            d_conv1_weight,
            d_conv1_bias,
            d_conv2_weight,
            d_conv2_bias,
            d_fc4_weight,
            d_labels,
            d_sum,
            t);

        // cudaDeviceSynchronize();

        // std::cout << "real: " << labels[t]<< ", predict : "<<  maxT(output5) << std::endl;
    }

    cudaMemcpy(sum, d_sum, block_num * sizeof(int), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << std::fixed << std::setprecision(10) << diff.count() << ":" << std::setprecision(10) << (float)sum[0]/ (float)10000 << std::endl;

    cudaMemcpy(sum, d_sum, block_num * sizeof(int), cudaMemcpyDeviceToHost);
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
