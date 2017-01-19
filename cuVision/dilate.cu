#include "headers.h"

__global__ void dilate() {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;


}

extern "C"
cudaError_t cudaDilate(cv::Mat & input, cv::Mat & output) {
    cudaError_t cudaStatus;

    return cudaStatus;
}