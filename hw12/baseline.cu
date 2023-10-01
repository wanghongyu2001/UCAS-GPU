// 这是程序二的模板程序，我们已经准备好了加载数据集和加载程序一模型参数的部分，请实现CUDA的深度学习推理过程，请严格保持输出格式输出
// 编译的命令为：nvcc test.cu -o test -Xcompiler "-O3 -std=c++14" -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

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
void conv2d(std::vector<float>  input, int inputRowSize, int inputColSize, int inputChannel, \
    std::vector<float>  kernel, std::vector<float> kernelBias, int kernelRowSize, int kernelColSize, \
    std::vector<float>&  output, int outputRowSize, int outputColSize, int outputChannel)
{

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

void reluSPMV(std::vector<float> input, int inputRowSize, \
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
    // auto conv1_weight = read_param(dir + "/conv1.weight_epoch400.txt");
    // auto conv1_bias = read_param(dir + "/conv1.bias_epoch400.txt");
    // auto conv2_weight = read_param(dir + "/conv2.weight_epoch400.txt");
    // auto conv2_bias = read_param(dir + "/conv2.bias_epoch400.txt");
    // auto fc1_weight = read_param(dir + "/fc1.weight_epoch400.txt");
    // auto fc1_bias = read_param(dir + "/fc1.bias_epoch400.txt");
    // auto fc2_weight = read_param(dir + "/fc2.weight_epoch400.txt");
    // auto fc2_bias = read_param(dir + "/fc2.bias_epoch400.txt");
    // auto fc3_weight = read_param(dir + "/fc3.weight_epoch400.txt");
    // auto fc3_bias = read_param(dir + "/fc3.bias_epoch400.txt");
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
