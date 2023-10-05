// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc program_2.cu -o program_2 -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

// 读取MNIST数据集
std::vector<std::vector<float>> read_mnist_images(const std::string& path)
{
    std::ifstream file(path , std::ios::binary);
    if (!file)
    {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0 , num_images = 0 , num_rows = 0 , num_cols = 0;
    file.read((char*)&magic_number , sizeof(magic_number));
    file.read((char*)&num_images , sizeof(num_images));
    file.read((char*)&num_rows , sizeof(num_rows));
    file.read((char*)&num_cols , sizeof(num_cols));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
        ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_images = ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
        ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
    num_rows = ((num_rows & 0xff000000) >> 24) | ((num_rows & 0x00ff0000) >> 8) |
        ((num_rows & 0x0000ff00) << 8) | ((num_rows & 0x000000ff) << 24);
    num_cols = ((num_cols & 0xff000000) >> 24) | ((num_cols & 0x00ff0000) >> 8) |
        ((num_cols & 0x0000ff00) << 8) | ((num_cols & 0x000000ff) << 24);

    int image_size = num_rows * num_cols;
    std::vector<std::vector<float>> images(num_images , std::vector<float>(image_size));

    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < image_size; ++j)
        {
            unsigned char pixel = 0;
            file.read((char*)&pixel , sizeof(pixel));
            images[i][j] = (static_cast<float>(pixel) / 255.0f - 0.5) * 2;
        }
    }

    return images;
}

// 读取MNIST label数据集
std::vector<int> read_mnist_labels(const std::string& path)
{
    std::ifstream file(path , std::ios::binary);
    if (!file)
    {
        std::cout << "Cannot open file!" << std::endl;
        return {};
    }

    int magic_number = 0 , num_items = 0;
    file.read((char*)&magic_number , sizeof(magic_number));
    file.read((char*)&num_items , sizeof(num_items));

    // Reverse Integers (MNIST data is in big endian format)
    magic_number = ((magic_number & 0xff000000) >> 24) | ((magic_number & 0x00ff0000) >> 8) |
        ((magic_number & 0x0000ff00) << 8) | ((magic_number & 0x000000ff) << 24);
    num_items = ((num_items & 0xff000000) >> 24) | ((num_items & 0x00ff0000) >> 8) |
        ((num_items & 0x0000ff00) << 8) | ((num_items & 0x000000ff) << 24);

    std::vector<int> labels(num_items);
    for (int i = 0; i < num_items; ++i)
    {
        unsigned char label = 0;
        file.read((char*)&label , sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    return labels;
}

// 读取模型参数
std::vector<float> read_param(const std::string& path)
{
    std::ifstream file(path);
    std::vector<float> params;
    float param;
    while (file >> param)
    {
        params.push_back(param);
    }
    return params;
}


__device__ void conv_oneThread(float* inputTensor , float* outputTensor , float* kernel , float bias)
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            *outputTensor += inputTensor[i * 5 + j] * kernel[i * 5 + j];
        }
    }
    //if Cin is 1
    if (blockDim.z == 1)
    {
        *outputTensor += bias;
        if (*outputTensor < 0.0f)
        {
            *outputTensor = 0.0f;
        }
    }
}

__device__ void pooling_oneThread(float* inputTensor , float* outputTensor)
{
    for (int i = 0; i < 4; i++)
    {
        if (inputTensor[i] > *outputTensor)
        {
            *outputTensor = inputTensor[i];
        }
    }
}

__device__ void linear_oneTread(float* inputTensor , float* outputTensor , float* weight)
{
    for (int i = 0; i < 16; i++)
    {
        *outputTensor += inputTensor[i] * weight[i];
    }
}

__global__ void convLayer1(float* inputTensor , float outputTensor[6][24][24] , float* convWeight , float* convBias)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float s_convWeight_split[5][5];
    __shared__ float s_convBias_split;
    float t_inputTensor_split[5][5];
    float t_outputTensor_split = 0;
    if (z < 6 && y < 24 && x < 24)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                t_inputTensor_split[i][j] = inputTensor[(y + i) * 28 + x + j];
            }
        }
    }

    if (z < 6 && threadIdx.y < 5 && threadIdx.x < 5)
    {
        s_convWeight_split[threadIdx.y][threadIdx.x] = convWeight[blockIdx.z * 25 + threadIdx.y * 5 + threadIdx.x];
    }

    s_convBias_split = convBias[blockIdx.z];

    __syncthreads( );
    if (z < 6 && y < 24 && x < 24)
    {
        conv_oneThread(&t_inputTensor_split[0][0] , &t_outputTensor_split , &s_convWeight_split[0][0] , s_convBias_split);
        outputTensor[z][y][x] = t_outputTensor_split;
    }
}

__global__ void poolingLayer1(float inputTensor[6][24][24] , float outputTensor[6][12][12])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float t_inputTensor_split[4];
    float t_outputTensor_split = 0;

    if (z < 6 && x < 12 && y < 12)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                t_inputTensor_split[2 * i + j] = inputTensor[z][2 * y + i][2 * x + j];
            }
        }
        pooling_oneThread(t_inputTensor_split , &t_outputTensor_split);
        outputTensor[z][y][x] = t_outputTensor_split;
    }
}

__global__ void convLayer2(float inputTensor[6][12][12] , float outputTensor[16][8][8] , float* convWeight , float* convBias)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float s_convWeight_split[6][5][5];
    __shared__ float s_convBias_split;
    float t_inputTensor_split[5][5];
    __shared__ float s_outputTensor_split[6][8][8];


    s_outputTensor_split[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads( );
    if (z < 16 * 6 && threadIdx.y < 5 && threadIdx.x < 5)
    {
        s_convWeight_split[threadIdx.z][threadIdx.y][threadIdx.x] = convWeight[z * 25 + threadIdx.y * 5 + threadIdx.x];
        s_convBias_split = convBias[blockIdx.z];
    }
    __syncthreads( );

    if (z < 16 * 6 && y < 8 && x < 8)
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                t_inputTensor_split[i][j] = inputTensor[threadIdx.z][y + i][x + j];
            }
        }
        conv_oneThread(&t_inputTensor_split[0][0] , &s_outputTensor_split[threadIdx.z][threadIdx.y][threadIdx.x] ,
                       &s_convWeight_split[threadIdx.z][0][0] , s_convBias_split);
    }
    __syncthreads( );

    if (threadIdx.z == 0 && y < 8 && x < 8)
    {
        //summarize
        for (int i = 1; i < 6; i++)
        {
            s_outputTensor_split[0][threadIdx.y][threadIdx.x] += s_outputTensor_split[i][threadIdx.y][threadIdx.x];
        }
        s_outputTensor_split[0][threadIdx.y][threadIdx.x] += s_convBias_split;
        //ReLU
        if (s_outputTensor_split[0][threadIdx.y][threadIdx.x] < 0.0f)
        {
            s_outputTensor_split[0][threadIdx.y][threadIdx.x] = 0;
        }
    }
    __syncthreads( );
    outputTensor[blockIdx.z][y][x] = s_outputTensor_split[0][threadIdx.y][threadIdx.x];
}

__global__ void poolingLayer2(float inputTensor[16][8][8] , float outputTensor[16 * 4 * 4])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float t_inputTensor_split[2][2];
    float t_outputTensor_split = 0;

    if (z < 16 && x < 4 && y < 4)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                t_inputTensor_split[i][j] = inputTensor[z][2 * y + i][2 * x + j];
            }
        }
        pooling_oneThread(&t_inputTensor_split[0][0] , &t_outputTensor_split);
        outputTensor[z * 4 * 4 + y * 4 + x] = t_outputTensor_split;
    }
}

__global__ void linear1_ReLU(float inputTensor[16 * 4 * 4] , float outputTensor[120] , float fc1Weight[120][256] , float fc1Bias[120])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float  s_fc1Bias_split[8];
    float  t_fc1Weight_split[16] = { 0 };
    float t_inputTensor_split[16] = { 0 };
    __shared__ float s_temp[8][16];

    s_temp[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads( );

    if (x < 16 && y < 120)
    {
        s_fc1Bias_split[threadIdx.y] = fc1Bias[y];
        __syncthreads( );

        for (int i = 0; i < 16; i++)
        {
            t_fc1Weight_split[i] = fc1Weight[y][16 * x + i];
            t_inputTensor_split[i] = inputTensor[16 * x + i];
        }
        //Multiply split        
        linear_oneTread(t_inputTensor_split , &s_temp[threadIdx.y][threadIdx.x] , t_fc1Weight_split);
        __syncthreads( );

        //sum
        if (threadIdx.x == 0)
        {
            for (int i = 1; i < 16; i++)
            {
                s_temp[threadIdx.y][0] += s_temp[threadIdx.y][i];
            }
            //add bias
            s_temp[threadIdx.y][0] += s_fc1Bias_split[threadIdx.y];

            //ReLU
            if (s_temp[threadIdx.y][0] < 0.0f)
            {
                s_temp[threadIdx.y][0] = 0.0f;
            }
            //copy to global memory
            outputTensor[y] = s_temp[threadIdx.y][0];
        }
    }
}

__global__ void linear2_ReLU(float inputTensor[120] , float outputTensor[84] , float fc2Weight[84][120] , float fc2Bias[84])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__  float s_fc2Bias_split[8];
    float t_fc2Weight_split[16] = { 0 };
    float t_inputTensor_split[16] = { 0 };
    __shared__  float s_temp[8][8];


    s_temp[threadIdx.y][threadIdx.x] = 0.0f;
    if (y < 84)
        s_fc2Bias_split[threadIdx.y] = fc2Bias[y];
    __syncthreads( );

    if (x < 7 && y < 84)
    {
        for (int i = 0; i < 16; i++)
        {
            t_fc2Weight_split[i] = fc2Weight[y][16 * x + i];
            t_inputTensor_split[i] = inputTensor[16 * x + i];
        }
    }

    if (x == 7 && y < 84)
    {
        for (int i = 0; i < 8; i++)
        {
            t_fc2Weight_split[i] = fc2Weight[y][16 * x + i];
            t_inputTensor_split[i] = inputTensor[16 * 7 + i];
        }
    }

    if (x < 8 && y < 84)
    {
        linear_oneTread(t_inputTensor_split , &s_temp[threadIdx.y][threadIdx.x] , t_fc2Weight_split);
    }
    __syncthreads( );

    if (threadIdx.x == 0 && y < 84)
    {
        for (int i = 1; i < 8; i++)
        {
            s_temp[threadIdx.y][0] += s_temp[threadIdx.y][i];
        }
        //add bias
        s_temp[threadIdx.y][0] += s_fc2Bias_split[threadIdx.y];

        //ReLU
        if (s_temp[threadIdx.y][0] < 0.0f)
        {
            s_temp[threadIdx.y][0] = 0.0f;
        }
        //copy to global memory
        outputTensor[y] = s_temp[threadIdx.y][0];
    }
}

__global__ void linear3(float inputTensor[84] , float outputTensor[10] , float fc3Weight[10][84] , float fc3Bias[10])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //split into 4*8 per block
    __shared__ float s_fc3Bias_split[4];
    float t_fc3Weight_split[8] = { 0 };
    float t_inputTensor_split[8] = { 0 };
    __shared__ float s_temp[4][11];

    s_temp[threadIdx.y][threadIdx.x] = 0.0f;
    if (y < 10)
        s_fc3Bias_split[threadIdx.y] = fc3Bias[y];
    __syncthreads( );

    if (x < 10 && y < 10)
    {
        for (int i = 0; i < 8; i++)
        {
            t_fc3Weight_split[i] = fc3Weight[y][8 * x + i];
            t_inputTensor_split[i] = inputTensor[8 * x + i];
        }
    }

    if (x == 10 && y < 10)
    {
        for (int i = 0; i < 4; i++)
        {
            t_fc3Weight_split[i] = fc3Weight[y][8 * 10 + i];
            t_inputTensor_split[i] = inputTensor[8 * 10 + i];
        }

    }

    if (x < 11 && y < 10)
    {
        for (int i = 0; i < 8; i++)
        {
            s_temp[threadIdx.y][threadIdx.x] += t_inputTensor_split[i] * t_fc3Weight_split[i];
        }
    }
    __syncthreads( );

    if (threadIdx.x == 0 && y < 10)
    {
        for (int i = 1; i < 11; i++)
        {
            s_temp[threadIdx.y][0] += s_temp[threadIdx.y][i];
        }
        //add bias 
        s_temp[threadIdx.y][0] += s_fc3Bias_split[threadIdx.y];
        outputTensor[y] = s_temp[threadIdx.y][0];
    }
}


int main(int argc , char* argv[ ])
{
    std::string dir = argv[1];  // 第一个参数是程序所在的目录，这个目录是存放前一步训练模型参数文件的目录，从这个目录下读取模型参数文件，相对于这个目录读取测试集图片和标签
    //cleastd::cout << dir;

    // 读取测试集，对于想实现CUDA C/C++训练的同学，参考训练集文件名为train-images-idx3-ubyte和train-labels-idx1-ubyte
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


    float* d_p_image , * d_p_conv1_weight , * d_p_conv1_bias , * d_p_conv2_weight , * d_p_conv2_bias , \
        * d_p_fc1_weight , * d_p_fc1_bias , * d_p_fc2_weight , * d_p_fc2_bias , * d_p_fc3_weight , * d_p_fc3_bias , \
        (*d_conv1topooling1)[24][24] , (*d_pooling1toconv2)[12][12] , (*d_conv2topooling2)[8][8] , * d_pooling2tofc1 , \
        * d_fc1tofc2 , * d_fc2tofc3 , * d_predict;



    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now( );

    cudaMalloc((void**)&d_p_image , images[0].size( ) * 10 * sizeof(float));
    cudaMalloc((void**)&d_p_conv1_weight , conv1_weight.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_conv1_bias , conv1_bias.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_conv2_weight , conv2_weight.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_conv2_bias , conv2_bias.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_fc1_weight , fc1_weight.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_fc1_bias , fc1_bias.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_fc2_weight , fc2_weight.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_fc2_bias , fc2_bias.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_fc3_weight , fc3_weight.size( ) * sizeof(float));
    cudaMalloc((void**)&d_p_fc3_bias , fc3_bias.size( ) * sizeof(float));

    cudaMalloc((void**)&d_conv1topooling1 , 6 * 24 * 24 * sizeof(float));
    cudaMalloc((void**)&d_pooling1toconv2 , 6 * 12 * 12 * sizeof(float));
    cudaMalloc((void**)&d_conv2topooling2 , 16 * 8 * 8 * sizeof(float));
    cudaMalloc((void**)&d_pooling2tofc1 , 16 * 4 * 4 * sizeof(float));
    cudaMalloc((void**)&d_fc1tofc2 , 120 * sizeof(float));
    cudaMalloc((void**)&d_fc2tofc3 , 84 * sizeof(float));
    cudaMalloc((void**)&d_predict , 10 * sizeof(float));

    cudaMemcpy(d_p_conv1_weight , &conv1_weight[0] , conv1_weight.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_conv1_bias , &conv1_bias[0] , conv1_bias.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_conv2_weight , &conv2_weight[0] , conv2_weight.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_conv2_bias , &conv2_bias[0] , conv2_bias.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_fc1_weight , &fc1_weight[0] , fc1_weight.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_fc1_bias , &fc1_bias[0] , fc1_bias.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_fc2_weight , &fc2_weight[0] , fc2_weight.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_fc2_bias , &fc2_bias[0] , fc2_bias.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_fc3_weight , &fc3_weight[0] , fc3_weight.size( ) * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_fc3_bias , &fc3_bias[0] , fc3_bias.size( ) * sizeof(float) , cudaMemcpyHostToDevice);


    float predict[10] = { 0 };
    int correct = 0;
    for (int t = 0; t < 10000; t++)
    {
        cudaMemcpy(d_p_image , &images[t][0] , sizeof(float) * 28 * 28 , cudaMemcpyHostToDevice);

        dim3 block_dim(8 , 8 , 1);
        dim3 grid_dim(3 , 3 , 6);
        convLayer1 << < grid_dim , block_dim >> > (d_p_image , d_conv1topooling1 , d_p_conv1_weight , d_p_conv1_bias);
        block_dim = { 4 , 4 , 6 };
        grid_dim = { 3 , 3 , 1 };
        poolingLayer1 << < grid_dim , block_dim >> > (d_conv1topooling1 , d_pooling1toconv2);
        block_dim = { 8,8,6 };
        grid_dim = { 1,1,16 };
        convLayer2 << <grid_dim , block_dim >> > (d_pooling1toconv2 , d_conv2topooling2 , d_p_conv2_weight , d_p_conv2_bias);
        block_dim = { 4,4,16 };
        grid_dim = { 1,1,1 };
        poolingLayer2 << <grid_dim , block_dim >> > (d_conv2topooling2 , d_pooling2tofc1);
        block_dim = { 16,8,1 };
        grid_dim = { 1,15,1 };
        linear1_ReLU << <grid_dim , block_dim >> > (d_pooling2tofc1 , d_fc1tofc2 , (float(*)[256])d_p_fc1_weight , d_p_fc1_bias);
        block_dim = { 8,8,1 };
        grid_dim = { 1,11,1 };
        linear2_ReLU << <grid_dim , block_dim >> > (d_fc1tofc2 , d_fc2tofc3 , (float(*)[120])d_p_fc2_weight , d_p_fc2_bias);
        block_dim = { 11,4,1 };
        grid_dim = { 1,3,1 };
        linear3 << <grid_dim , block_dim >> > (d_fc2tofc3 , d_predict , (float(*)[84]) d_p_fc3_weight , d_p_fc3_bias);

        cudaMemcpy(predict , d_predict , 10 * sizeof(float) , cudaMemcpyDeviceToHost);

        int predictLabel = 0;
        float maxValue = 0;
        for (int i = 0; i < 10; i++)
        {
            if (predict[i] > maxValue)
            {
                maxValue = predict[i];
                predictLabel = i;
            }
        }
        if (labels[t] == predictLabel)
            correct++;
    }

    cudaFree(d_p_image);
    cudaFree(d_p_conv1_weight);
    cudaFree(d_p_conv1_bias);
    cudaFree(d_p_conv2_weight);
    cudaFree(d_p_fc1_weight);
    cudaFree(d_p_fc1_bias);
    cudaFree(d_p_fc2_weight);
    cudaFree(d_p_fc2_bias);
    cudaFree(d_p_fc3_weight);
    cudaFree(d_p_fc3_bias);

    cudaFree(d_conv1topooling1);
    cudaFree(d_pooling1toconv2);
    cudaFree(d_conv2topooling2);
    cudaFree(d_pooling2tofc1);
    cudaFree(d_fc1tofc2);
    cudaFree(d_fc2tofc3);
    cudaFree(d_predict);

    cudaDeviceSynchronize( );
    // 结束计时
    auto end = std::chrono::high_resolution_clock::now( );
    std::chrono::duration<double> diff = end - start;

    // 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    std::cout << std::fixed << std::setprecision(2) << diff.count( ) << ":" << std::setprecision(4) << ((float)correct) / labels.size( );

    return 0;
}
