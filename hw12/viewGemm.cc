#include <iostream>

using namespace std;
const int inputRowSize = 4, inputColSize = 5;
const int kernelRowSize = 5, kernelColSize = 4;
const int outputRowSize = 4, outputColSize = 4;


void init_ij(double* A, int n, int m)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < m; k++)
            A[j * m + k] = k + j;
    }
}

void init_one(double* A, int n, int m)
{
    for (int j = 0; j < n; j++)
    {
        for (int k = 0; k < m; k++)
            A[j * m + k] = 1;
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

void GEMM(double* input, int inputRowSize, int inputColSize, \
    double* kernel, int kernelRowSize, int kernelColSize, \
    double* output, int outputRowSize, int outputColSize)
{
    for (int i = 0; i < outputRowSize; i++)
    {
        for (int j = 0; j < outputColSize; j++)
        {

            double tmp = 0;
            for (int k = 0; k < inputColSize; k++)
                tmp += input[i * inputColSize + k] * kernel[k * kernelColSize + j];
            output[i * outputColSize + j] = tmp;
        }
    }
}
void reluGEMM(double* input, int inputRowSize, int inputColSize, \
    double* kernel, int kernelRowSize, int kernelColSize, \
    double* output, int outputRowSize, int outputColSize)
{
    for (int i = 0; i < outputRowSize; i++)
    {
        for (int j = 0; j < outputColSize; j++)
        {

            double tmp = 0;
            for (int k = 0; k < inputColSize; k++)
                tmp += input[i * inputColSize + k] * kernel[k * kernelColSize + j];
            if (tmp >= 0) output[i * outputColSize + j] = tmp;
            else output[i * outputColSize + j] = 0;
        }
    }
}

void print_M(double* A, int rowS, int colS)
{
    for (int i = 0; i < rowS; i++)
    {
        for (int j = 0; j < colS; j++)
        {
            cout << A[i * colS + j] << " ";
        }
        cout << endl;
    }

}
int main()
{
    
    double* input = (double*)malloc(inputRowSize * inputColSize * sizeof(double));
    double* kernel = (double*)malloc(kernelColSize * kernelRowSize * sizeof(double));
    double* output = (double*)malloc(outputRowSize * outputColSize * sizeof(double));
    init_ij(input, inputRowSize, inputColSize);
    init_ij(kernel, kernelRowSize, kernelColSize);
    // GEMM(input, inputRowSize, inputColSize,kernel, kernelRowSize, kernelColSize, output, outputRowSize, outputColSize);
    reluGEMM(input, inputRowSize, inputColSize,kernel, kernelRowSize, kernelColSize, output, outputRowSize, outputColSize);

    print_M(output, outputRowSize, outputColSize);

}
