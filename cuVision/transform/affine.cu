#include "..\cumath\cumath.cuh"

#define K_SIZE 3
#define FACTOR 0.5

__constant__ float formType[K_SIZE][K_SIZE] = { {1 * FACTOR,0,0},{0,1 * FACTOR,0},{0,0,1 * FACTOR} };

template<typename type>
__device__ void matrixMul(type *input1, type *input2, type *output) {

}

__global__ void affine(int imageRows, int imageCols) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    // coodinator
    int x = col - imageCols / 2;
    int y = imageRows / 2 - row;
    int a[] = { x,y,1 };

    if (row < imageRows&&col < imageCols) {

    }
}

void cudaAffine(cv::Mat & input, cv::Mat & output) {

}