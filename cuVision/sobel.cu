#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { std::cout << std::endl << "CUDA Error: " << cudaGetErrorString(a) << ", error number: " << a << std::endl; cudaDeviceReset(); assert(0);}}
#define MAX_THREADS 32
#define K_SIZE 3

__constant__ char sobelKernelXC[K_SIZE][K_SIZE] = { { -1.0,0.0,1.0 },{ -2.0,0.0,2.0 },{ -1.0,0.0,1.0 } };
__constant__ char sobelKernelYC[K_SIZE][K_SIZE] = { { -1.0,-2.0,-1.0 },{ 0.0,0.0,0.0 },{ 1.0,2.0,1.0 } };

__global__ void sobel(uchar *input, int rows, int cols, size_t inputPitch, uchar *gx, size_t gxPitch, uchar *gy, size_t gyPitch, uchar *output, size_t outputPitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        // convolving
        for (size_t i = 0; i < K_SIZE; i++)
            for (size_t j = 0; j < K_SIZE; j++) {
                uchar *inputValue = (uchar*)((char*)input + row*inputPitch) + col;
                // convolving gx
                uchar *gxValue = (uchar*)((char*)gx + (row + i - K_SIZE / 2)*outputPitch) + (col + j - K_SIZE / 2);
                *gxValue += sobelKernelXC[i][j] * (*inputValue);

                // convolving gy
                uchar *gyValue = (uchar*)((char*)gy + (row + i - K_SIZE / 2)*outputPitch) + (col + j - K_SIZE / 2);
                *gyValue += sobelKernelYC[i][j] * (*inputValue);
        }

        uchar *gxValue = (uchar*)((char*)gx + row*outputPitch) + col;
        uchar *gyValue = (uchar*)((char*)gy + row*outputPitch) + col;
        uchar *outputValue = (uchar*)((char*)output + row*outputPitch) + col;
        *outputValue = *gxValue + *gyValue;
    }
}

extern "C"
void cudaSobel(cv::Mat & input, cv::Mat & output) {
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    uchar *d_input, *d_output, *gx, *gy;
    size_t inputPitch, outputPitch, gxPitch, gyPitch;

    cudaStream_t inputStream, outputStream, gxStream, gyStream;
    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream)); CUDA_CALL(cudaStreamCreate(&gxStream)); CUDA_CALL(cudaStreamCreate(&gyStream));

    CUDA_CALL(cudaMallocPitch(&d_input, &inputPitch, sizeof(uchar)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &outputPitch, sizeof(uchar)*output.cols, output.rows));
    CUDA_CALL(cudaMallocPitch(&gx, &gxPitch, sizeof(uchar)*output.cols, output.rows));
    CUDA_CALL(cudaMallocPitch(&gy, &gyPitch, sizeof(uchar)*output.cols, output.rows));

    CUDA_CALL(cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream));
    CUDA_CALL(cudaMemcpy2DAsync(gx, gxPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, gxStream));
    CUDA_CALL(cudaMemcpy2DAsync(gy, gyPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, gyStream));

    CUDA_CALL(cudaStreamSynchronize(inputStream)); CUDA_CALL(cudaStreamSynchronize(outputStream)); CUDA_CALL(cudaStreamSynchronize(gxStream)); CUDA_CALL(cudaStreamSynchronize(gyStream));

    cudaMemset(gx, 0, sizeof(uchar)*output.rows*output.cols);
    cudaMemset(gy, 0, sizeof(uchar)*output.rows*output.cols);

    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    sobel<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, inputPitch, gx, gxPitch, gy, gyPitch, d_output, outputPitch);
    CUDA_CALL(cudaDeviceSynchronize());

    // get data back
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, outputPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output); cudaFree(gx); cudaFree(gy);
    CUDA_CALL(cudaStreamDestroy(inputStream)); CUDA_CALL(cudaStreamDestroy(outputStream)); CUDA_CALL(cudaStreamDestroy(gxStream)); CUDA_CALL(cudaStreamDestroy(gyStream));
}