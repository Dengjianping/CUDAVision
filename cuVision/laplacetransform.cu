#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define K_SIZE 3
#define MAX_THREADS 32
#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { std::cout << std::endl << "CUDA Error: " << cudaGetErrorString(a) << ", error number: " << a << std::endl; cudaDeviceReset(); assert(0);}}

enum shape { full, same, valid };

//#define K_SIZE 3
__constant__ uchar lapalceKernel[K_SIZE][K_SIZE] = { { 0,1,0 },{ 1,-4,1 },{ 0,1,0 } };

__global__ void laplace(uchar *input, size_t inputPitch, int inputRow, int inputCol, uchar *output, size_t outputPitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockDim.x + threadIdx.x;

    if (row < inputRow&&col < inputCol) {
        for(size_t i = 0; i < K_SIZE; i++)
            for (size_t j = 0; j < K_SIZE; j++) {
                uchar *inputValue = (uchar*)((char*)input + row*inputPitch) + col;
                uchar *outputValue = (uchar*)((char*)output + (row + i)*outputPitch) + (col + j);
                *outputValue += (*inputValue)*lapalceKernel[i][j];
            }
    }

}

extern "C"
cudaError_t cudaLaplace(cv::Mat & input, cv::Mat & output, shape mode = full) {
    cudaError_t cudaStatus;
    switch (mode)
    {
    case full:
        output = cv::Mat(cv::Size(input.cols + K_SIZE - 1, input.rows + K_SIZE - 1), CV_8U, cv::Scalar(0));
        break;
    case same:
        output = cv::Mat(cv::Size(input.cols, input.rows), CV_8U, cv::Scalar(0));
        break;
    case valid:
        output = cv::Mat(cv::Size(input.cols - K_SIZE + 1, input.rows - K_SIZE + 1), CV_8U, cv::Scalar(0));
        break;
    default:
        break;
    }

    size_t inputPitch, outputPitch;
    uchar *d_input, *d_output;
    cudaStatus = cudaMallocPitch(&d_input, &inputPitch, input.cols * sizeof(uchar), input.rows);
    cudaStatus = cudaMallocPitch(&d_output, &outputPitch, output.cols * sizeof(uchar), output.rows);

    cudaStream_t inputCopy, outputCopy;
    cudaStatus = cudaStreamCreate(&inputCopy); cudaStatus = cudaStreamCreate(&outputCopy);

    cudaStatus = cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopy);
    cudaStatus = cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopy);

    CUDA_CALL(cudaStreamSynchronize(inputCopy)); CUDA_CALL(cudaStreamSynchronize(outputCopy));

    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    laplace<<<blockSize, threadSize>>> (d_input, inputPitch, input.rows, input.cols, d_output, outputPitch);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, outputPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaStreamDestroy(inputCopy)); CUDA_CALL(cudaStreamDestroy(outputCopy));
    cudaFree(d_input); cudaFree(d_output);
    return cudaStatus;
}