#include <iostream>

using namespace std;
const int inputRowSize = 12, inputColSize = 12, inputChannel = 6;
const int outputRowSize = 6, outputColSize = 6, outputChannel = 6;
const int kernelRowSize = 2, kernelColSize = 2;
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
            for (int k = 0; k < m; k++)
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

void reluMaxPool(double* input, int inputRowSize, int inputColSize, int inputChannel, \
    int kernelRowSize, int kernelColSize, \
    double* output, int outputRowSize, int outputColSize, int outputChannel)
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

void print_M(double* A, int n)
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

void print_M(double* A, int rowS, int colS, int chaS)
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
int main()
{

    double* input = (double*)malloc(inputRowSize * inputColSize * inputChannel * sizeof(double));
    double* kernel = (double*)malloc(kernelRowSize * kernelColSize * inputChannel * outputChannel * sizeof(double));
    double* output = (double*)malloc(outputRowSize * outputColSize * outputChannel * sizeof(double));
    init_ij(input, inputRowSize, inputColSize, inputChannel);
    init_one(kernel, kernelRowSize, kernelColSize, inputChannel * outputChannel);

    reluMaxPool(input, inputRowSize, inputColSize, inputChannel, kernelRowSize, kernelColSize, \
        output, outputRowSize, outputColSize, outputChannel);
    print_M(output, outputRowSize, outputColSize, outputChannel);

}
