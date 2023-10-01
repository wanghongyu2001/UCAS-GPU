#include <iostream>

using namespace std;
const int inputRowSize = 12, inputColSize = 12, inputChannel = 6;
const int outputRowSize = 8, outputColSize = 8, outputChannel = 16;
const int kernelRowSize = 5, kernelColSize = 5;
const int kernelOCSize = kernelColSize * kernelRowSize * inputChannel;


void init_ij(double* A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k++)
                A[i * n * m + j * m + k] = k + j;
        }
}
void init_one(double* A, int n, int m, int c)
{
    for (int i = 0; i < c; i++)
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < m; k ++ )
                A[i * n * m + j * m + k] = 1;
        }
}
void init_zero(double* A, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            A[i * n + j] = 0;
        }
}
void conv2d(double* input, int inputRowSize, int inputColSize, int inputChannel, \
    double* kernel, int kernelRowSize, int kernelColSize,  \
    double* output, int outputRowSize, int outputColSize, int outputChannel)
{
    
    for (int c = 0; c < outputChannel; c++)
    {
        for (int i = 0; i < outputRowSize; i++)
        {
            for (int j = 0; j < outputColSize; j ++ )
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
                                (row - i)* kernelColSize + (col - j)] * \
                                input[tc * inputRowSize * inputColSize + row * inputColSize + col];
                        }
                    }
                }
                    
                output[c * outputRowSize * outputColSize + i * outputColSize + j] = tmp;

            }
        }
    }
}

void print_M(double* A, int rowS, int colS, int chaS)
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
    double* input = (double*)malloc(inputRowSize * inputColSize * inputChannel * sizeof(double));
    double* kernel = (double*)malloc(kernelRowSize * kernelColSize * inputChannel * outputChannel * sizeof(double));
    double* output = (double*)malloc(outputRowSize * outputColSize * outputChannel * sizeof(double));
    init_ij(input, inputRowSize, inputColSize, inputChannel);
    init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);

    conv2d(input, inputRowSize, inputColSize, inputChannel, kernel, kernelRowSize, kernelColSize, \
         output, outputRowSize, outputColSize, outputChannel);
    print_M(output, outputRowSize, outputColSize, outputChannel);

}
