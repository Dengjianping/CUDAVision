/*
    sobel operation just uses a sobel operator to convolve a target matrix, so the key to improve performance
    is how to tune the convolution. well, either you can write your own convolution algorithm or use cufft lib.
    1. naive convolution algorithm. refer to this link: https://www.evl.uic.edu/sjames/cs525/final.html
    2. cufft. This is a fine tuned algorithm, a little complicated to apply FFT to convolution.
       a. Apply API cufftExecR2C to kernel and taregt matrix;
       b. mulplication both FFTed kernel and target matrix;
       c. Inverse the result from step b.
       d. 
*/

#include "..\cumath\cumath.cuh"

#define K_SIZE 3
#define TILE_H 32
#define TILE_W 32
#define RADIUS 8
//#define LOOP_UNROLLING

__constant__ int sobelKernelXC[K_SIZE][K_SIZE] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
__constant__ int sobelKernelYC[K_SIZE][K_SIZE] = { { -1,-2,-1 },{ 0,0,0 },{ 1,2,1 } };

__global__ void sobel(int *input, int height, int width, int radius, int R, size_t pitch, int *output)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int shared[TILE_H][TILE_W+16]; // add extra 16 columns to eliminate bank confict, because there is 2-way bank conflict if define share memory like __shared__ int shared[TILE_H][TILE_W]
    if (row < height && col < width)
    {
        /*
        actually, there will be some branch divergence when thread face if-else statement, which will affect the performance.
        */
        // upper left corner in the block
        if (row - R < 0 || col - R < 0)
            shared[threadIdx.y][threadIdx.x] = 0;
        else
            shared[threadIdx.y][threadIdx.x] = *((int*)((char*)input + (row - R) * pitch) + (col - R));

        // upper right
        if (row - R < 0 || col + R > width - 1)
            shared[threadIdx.y][threadIdx.x + blockDim.x] = 0;
        else
            shared[threadIdx.y][threadIdx.x + blockDim.x] = *((int*)((char*)input + (row - R) * pitch) + (col + R));

        //bottom left
        if (row + R > height - 1 || col - R < 0)
            shared[threadIdx.y + blockDim.y][threadIdx.x] = 0;
        else
            shared[threadIdx.y + blockDim.y][threadIdx.x] = *((int*)((char*)input + (row + R) * pitch) + (col - R));

        // bottom right
        if (row + R > height - 1 || col - R > width - 1)
            shared[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = 0;
        else
            shared[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = *((int*)((char*)input + (row + R) * pitch) + (col + R));
        __syncthreads();

        int sumx = 0, sumy = 0;
#ifdef LOOP_UNROLLING
        // use loop unrolling to improve performance, it can avoid branching.
        sumx = sobelKernelXC[radius - 1][radius - 1] * shared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS - 1] +
            sobelKernelXC[radius - 1][radius] * shared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS] +
            sobelKernelXC[radius - 1][radius + 1] * shared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS + 1] +
            sobelKernelXC[radius][radius - 1] * shared[threadIdx.y + RADIUS][threadIdx.x + RADIUS - 1] +
            sobelKernelXC[radius][radius] * shared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] +
            sobelKernelXC[radius][radius + 1] * shared[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 1] +
            sobelKernelXC[radius + 1][radius - 1] * shared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS - 1] +
            sobelKernelXC[radius + 1][radius] * shared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS] +
            sobelKernelXC[radius + 1][radius + 1] * shared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS + 1];
        sumy = sobelKernelXC[radius - 1][radius - 1] * shared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS - 1] +
            sobelKernelYC[radius - 1][radius] * shared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS] +
            sobelKernelYC[radius - 1][radius + 1] * shared[threadIdx.y + RADIUS - 1][threadIdx.x + RADIUS + 1] +
            sobelKernelYC[radius][radius - 1] * shared[threadIdx.y + RADIUS][threadIdx.x + RADIUS - 1] +
            sobelKernelYC[radius][radius] * shared[threadIdx.y + RADIUS][threadIdx.x + RADIUS] +
            sobelKernelYC[radius][radius + 1] * shared[threadIdx.y + RADIUS][threadIdx.x + RADIUS + 1] +
            sobelKernelYC[radius + 1][radius - 1] * shared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS - 1] +
            sobelKernelYC[radius + 1][radius] * shared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS] +
            sobelKernelYC[radius + 1][radius + 1] * shared[threadIdx.y + RADIUS + 1][threadIdx.x + RADIUS + 1];
#else
        for (int i = -radius; i <= radius; i++)
            for (int j = -radius; j <= radius; j++)
            {
                sumx += sobelKernelXC[radius + i][radius + j] * shared[threadIdx.y + R - i][threadIdx.x + R - j];
                sumy += sobelKernelYC[radius + i][radius + j] * shared[threadIdx.y + R - i][threadIdx.x + R - j];
            }
#endif
        //__syncthreads(); // wait current thread job done
        int *out = (int*)((char*)output + row*pitch) + col;
        *out = sqrtf(powf(sumx, 2) + powf(sumy, 2));
    }
}

extern "C"
void cudaSobel(cv::Mat & input, cv::Mat & output) 
{
    input.convertTo(input, CV_32S);
    output = cv::Mat(input.size(), CV_32S, cv::Scalar(0));

    int *d_input, *d_output;
    size_t pitch;

    cudaStream_t inputStream, outputStream;
    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream));

    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(int)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(int)*output.cols, output.rows));
  
    CUDA_CALL(cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(int)*input.cols, sizeof(int)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(int)*output.cols, sizeof(int)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream));

    dim3 threadSize(16, 16);
    dim3 blockSize(input.cols / threadSize.x, input.rows / threadSize.y);

    sobel<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, 1, 8, pitch, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    // get data back
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(int)*output.cols, d_output, pitch, sizeof(int)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output);
    CUDA_CALL(cudaStreamDestroy(inputStream)); CUDA_CALL(cudaStreamDestroy(outputStream));

    output.convertTo(output, CV_8U);
    input.convertTo(input, CV_8U);
}