#include "headers.h"

__global__ void threshold(uchar *input, size_t inputPitch, int imageRows, int imageCols, uchar *output, size_t outputPitch, uchar thresholdValue) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < imageRows&&col < imageCols) {
        uchar *pixelValue = (uchar*)((char*)input + row*inputPitch) + col;
        uchar *outputPixelValue = (uchar*)((char*)output + row*outputPitch) + col;
        if (*pixelValue < thresholdValue) {
            *outputPixelValue = 0;
        }
        else {
            *outputPixelValue = 255;
        }
    }
}

extern "C"
cudaError_t cudaThreshold(cv::Mat & input, cv::Mat & output, uchar thresholdValue) {
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    cudaError_t cudaStatus;

    // make sure the image is a gray image
    if (input.type() != CV_8U) {
        cv::cvtColor(input, input, CV_8U);
    }

    uchar *d_input, *d_output;
    size_t inputPitch, outputPitch;

    // get pitch
    cudaStatus = cudaMallocPitch(&d_input, &inputPitch, sizeof(uchar)*input.cols, input.rows);
    cudaStatus = cudaMallocPitch(&d_output, &outputPitch, sizeof(uchar)*output.cols, output.rows);

    // use stream to accelerate copy operation
    cudaStream_t inputCopy, outputCopy;
    cudaStatus = cudaStreamCreate(&inputCopy); cudaStatus = cudaStreamCreate(&outputCopy);

    // copy data to device
    cudaStatus = cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopy);
    cudaStatus = cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopy);

    // block until data copy is complete
    cudaStatus = cudaStreamSynchronize(inputCopy); cudaStatus = cudaStreamSynchronize(outputCopy);

    // define block size and
    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    threshold <<<blockSize, threadSize>>> (d_input, inputPitch, input.rows, input.cols, d_output, outputPitch, thresholdValue);
    cudaStatus = cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, outputPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    // resource releasing
    cudaStreamDestroy(inputCopy); cudaStreamDestroy(outputCopy);
    cudaFree(d_input); cudaFree(d_output);

    return cudaStatus;
}