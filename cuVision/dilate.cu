//#include "headers.h"
#include "cumath.cuh"

#define K_SIZE 3

__constant__ uchar erodeKernel[K_SIZE][K_SIZE] = { { 0,1,0 },{ 1,1,1 },{ 0,1,0 } };

__global__ void dilate(uchar *input, size_t inputPitch, int inputRow, int inputCol, uchar *output, size_t outputPitch, int interation) {
    extern __shared__ uchar kernel[];
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;


    if (row < inputRow&&col < inputCol) {
        //convolve2D<uchar>(row, col, input, inputRow, inputCol, inputPitch, K_SIZE, kernel, output, outputPitch);
        for (size_t inter = 0; inter < interation; inter++)
            for (size_t i = 0; i < K_SIZE; i++)
                for (size_t j = 0; j < K_SIZE; j++) {
                    uchar *inputValue = (uchar*)((char*)input + row*inputPitch) + col;
                    uchar *outputValue = (uchar*)((char*)output + (row + i - K_SIZE / 2)*outputPitch) + (col + j - K_SIZE / 2);
                    *outputValue += erodeKernel[i][j] * (*inputValue);
                }
    }
}

extern "C"
void cudaDilate(cv::Mat & input, cv::Mat & output, int iteration = 1) {
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    if (input.type() != CV_8U) {
        cv::cvtColor(input, input, CV_8U);
    }

    uchar *d_input, *d_output;
    size_t inputPitch, outputPitch;

    CUDA_CALL(cudaMallocPitch(&d_input, &inputPitch, sizeof(uchar)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &outputPitch, sizeof(uchar)*output.cols, output.rows));

    cudaStream_t inputCopy, outputCopy;
    CUDA_CALL(cudaStreamCreate(&inputCopy)); CUDA_CALL(cudaStreamCreate(&outputCopy));

    CUDA_CALL(cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopy));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopy));

    CUDA_CALL(cudaStreamSynchronize(inputCopy)); CUDA_CALL(cudaStreamSynchronize(outputCopy));

    dim3 blockSize(input.cols / (MAX_THREADS) + 1, input.rows / (MAX_THREADS/2) + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS/2);

    int sharedSize = sizeof(uchar)*K_SIZE*K_SIZE;
    for (int iter = 0; iter < iteration; iter++) {
        dilate <<<blockSize, threadSize, sharedSize>>> (d_input, inputPitch, input.rows, input.cols, d_output, outputPitch, iteration);
    }
    CUDA_CALL(cudaDeviceSynchronize());
    

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, outputPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaStreamDestroy(inputCopy); cudaStreamDestroy(outputCopy);
    cudaFree(d_input); cudaFree(d_output);
}