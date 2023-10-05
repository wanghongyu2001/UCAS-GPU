#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
using namespace std;
const int inputRowSize = 8, inputColSize = 8, inputChannel = 16;
const int outputRowSize = 4, outputColSize = 4, outputChannel = 16;
const int kernelRowSize = 2, kernelColSize = 2, kernelSize = 2;
// const int kernelOCSize = kernelColSize * kernelRowSize * inputChannel;


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
#define type float
const int BLOCK_HEIGHT = 8, BLOCK_WIDTH = 4, KERNEL_SIZE = 2, TMP_SIZE = 16 * 4;
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
        if (begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, inputColSize)  >= inputRowSize * inputColSize * inputChannel)
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
                    if (((begin_pos + (oc + 1) * inputColSize * inputRowSize + (in_tile_row_start + i) * inputColSize + in_tile_col))  >= inputRowSize * inputColSize * inputChannel)
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
            printf("tmp[%d] = %f, output[%d]\n",tmp_pos, tmp[tmp_pos], out_pos);
            #endif
        }

        //prefetch the next inputchannel data, in fact in = 0
        __syncthreads();
    }
    
}

void reluMaxPool(std::vector<float>& input, int inputRowSize, int inputColSize, int inputChannel, \
    int kernelRowSize, int kernelColSize, \
    std::vector<float>& output, int outputRowSize, int outputColSize, int outputChannel)
{
#if 1
    float* d_input, * d_output;
    int input_size = inputChannel * inputColSize * inputRowSize * sizeof(float);
    int output_size = outputChannel * outputColSize * outputRowSize * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, input_size));
    checkCudaErrors(cudaMalloc(&d_output, output_size));

    //memcpy
    checkCudaErrors(cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice));
    int gridx = outputColSize / BLOCK_WIDTH, gridy = outputRowSize / BLOCK_HEIGHT;
    gridx = gridx <= 0 ? 1 : gridx;
    gridy = gridy <= 0 ? 1 : gridy;
    printf("grid %d %d, block %d %d\n",gridx, gridy,BLOCK_WIDTH, BLOCK_HEIGHT );
    dim3 grid(gridx,  gridy);
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    // printf("1111\n");
    // f1 << <1, 200 >> > (1);
    // cudaDeviceSynchronize();
    _reluMaxPoll< BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_SIZE, TMP_SIZE> << <grid, block >> > (d_input, d_output, inputChannel, inputRowSize, inputColSize,
        outputChannel, outputRowSize, outputColSize,
        2);
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
        // 处理错误
    } else {
        // kernel 调用成功
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));
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


    
    void init_ij(std::vector<float>& A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
                A[i * n * m + j * m + k] = k + j + i;
        }
}
void init_one(std::vector<float>& A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
                A[i * n * m + j * m + k] = 1;
        }
}
void init_zero(std::vector<float>& A, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            A[i * n + j] = 0;
        }
}
void print_M(std::vector<float>& A, int n)
{
    for (int i = 0; i < n; i++)
    {

        for (int j = 0; j < n; j++)
        {
            cout << A[i * n + j] << " ";
        }
        cout << endl;
    }
}

void print_M(std::vector<float>& A, int rowS, int colS, int chaS)
{
    for (int c = 0; c < chaS; c++)
    {

        cout << "channel : " << c << endl;
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
class Block {
public:
    __device__ int thdId() {
        return threadIdx.x;
    }

    __device__ void print() {
        printf("threadId is: %d\n", thdId());
    }

};
__global__ void f1(int i) {
    Block block;
    block.print();
    printf("f1%d\n", i);

}
void reluMaxPoolCheck(std::vector<float>& input, int inputRowSize, int inputColSize, int inputChannel, \
    int kernelRowSize, int kernelColSize, \
    std::vector<float>& output, int outputRowSize, int outputColSize, int outputChannel)
{

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
}

int main()
{

    vector<float> input(inputRowSize * inputColSize * inputChannel, 0);
    vector<float> output(outputRowSize * outputColSize * outputChannel, 0);
    init_ij(input, inputRowSize, inputColSize, inputChannel);
    // init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);
    //check
    // print_M(input, inputRowSize, inputColSize, inputChannel);
    reluMaxPool(input, inputRowSize, inputColSize, inputChannel, kernelRowSize, kernelColSize, \
        output, outputRowSize, outputColSize, outputChannel);
    print_M(output, outputRowSize, outputColSize, outputChannel);
    reluMaxPoolCheck(input, inputRowSize, inputColSize, inputChannel, kernelRowSize, kernelColSize, \
        output, outputRowSize, outputColSize, outputChannel);
    printf("------------------------------------------\n");
    print_M(output, outputRowSize, outputColSize, outputChannel);

}
