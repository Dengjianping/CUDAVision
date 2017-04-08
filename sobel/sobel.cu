#include "..\cumath\cumath.cuh"

#define K_SIZE 3
#define TILE_H 4
#define TILE_W 30
#define LOOP_UNROLLING

__constant__ int sobelKernelXC[K_SIZE][K_SIZE] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
__constant__ int sobelKernelYC[K_SIZE][K_SIZE] = { { -1,-2,-1 },{ 0,0,0 },{ 1,2,1 } };

__global__ void sobel(int *input, int height, int width, int radius, int tile_h, int tile_w, size_t pitch, int *output)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ int localData[TILE_H + K_SIZE - 1][TILE_W + K_SIZE - 1];
    if (row < height && col < width) {
        if (row - radius < 0 || col - radius < 0) 
        {
            localData[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            localData[threadIdx.y][threadIdx.x] = *((float*)((char*)input + (blockIdx.y*TILE_H + threadIdx.y - radius)*pitch) + (blockIdx.x*TILE_W + threadIdx.x - radius));
            //localData[threadIdx.y][threadIdx.x] = *((float*)((char*)input + (row - radius)*pitch) + (col - radius));
        }
        if (row > height - radius - 1 || col > width - radius - 1)
        {
            localData[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        
        if ((threadIdx.y >= radius && threadIdx.y <= blockDim.y - radius - 1) && (threadIdx.x >= radius && threadIdx.x <= blockDim.x - radius - 1)) 
        {
            int sumx = 0, sumy = 0;
#ifdef LOOP_UNROLLING
            // use loop unrolling to improve performance, it can avoid branching.
            sumx = sobelKernelXC[radius - 1][radius - 1] * localData[threadIdx.y - 1][threadIdx.x - 1] +
                   sobelKernelXC[radius - 1][radius] * localData[threadIdx.y - 1][threadIdx.x] +
                   sobelKernelXC[radius - 1][radius + 1] * localData[threadIdx.y - 1][threadIdx.x + 1] +
                   sobelKernelXC[radius][radius - 1] * localData[threadIdx.y][threadIdx.x - 1] +
                   sobelKernelXC[radius][radius] * localData[threadIdx.y][threadIdx.x] +
                   sobelKernelXC[radius][radius + 1] * localData[threadIdx.y][threadIdx.x + 1] +
                   sobelKernelXC[radius + 1][radius - 1] * localData[threadIdx.y + 1][threadIdx.x - 1] +
                   sobelKernelXC[radius + 1][radius] * localData[threadIdx.y + 1][threadIdx.x] +
                   sobelKernelXC[radius + 1][radius + 1] * localData[threadIdx.y + 1][threadIdx.x + 1];
            sumy = sobelKernelXC[radius - 1][radius - 1] * localData[threadIdx.y - 1][threadIdx.x - 1] +
                   sobelKernelYC[radius - 1][radius] * localData[threadIdx.y - 1][threadIdx.x] +
                   sobelKernelYC[radius - 1][radius + 1] * localData[threadIdx.y - 1][threadIdx.x + 1] +
                   sobelKernelYC[radius][radius - 1] * localData[threadIdx.y][threadIdx.x - 1] +
                   sobelKernelYC[radius][radius] * localData[threadIdx.y][threadIdx.x] +
                   sobelKernelYC[radius][radius + 1] * localData[threadIdx.y][threadIdx.x + 1] +
                   sobelKernelYC[radius + 1][radius - 1] * localData[threadIdx.y + 1][threadIdx.x - 1] +
                   sobelKernelYC[radius + 1][radius] * localData[threadIdx.y + 1][threadIdx.x] +
                   sobelKernelYC[radius + 1][radius + 1] * localData[threadIdx.y + 1][threadIdx.x + 1];
#else
            for (int i = -radius; i <= radius; i++)
                for (int j = -radius; j <= radius; j++)
                {
                    sumx += sobelKernelXC[radius + i][radius + j] * localData[threadIdx.y - i][threadIdx.x - j];
                    sumy += sobelKernelYC[radius + i][radius + j] * localData[threadIdx.y - i][threadIdx.x - j];
                }
#endif
            __syncthreads(); // wait current thread job done
            //int *out = (int*)((char*)output + (blockIdx.y*TILE_H + threadIdx.y - radius)*pitch) + (blockIdx.x*TILE_W + threadIdx.x - radius);
            int *out = (int*)((char*)output + (row - radius)*pitch) + (col - radius);
            *out = sqrtf(powf(sumx, 2) + powf(sumy, 2));
        }
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

    dim3 threadSize(32, 6);
    dim3 blockSize(input.cols / TILE_W + 1, input.rows / TILE_H + 1);

    size_t shared = threadSize.x*threadSize.y * sizeof(int);
    sobel<<<blockSize, threadSize>>>(d_input, blockSize.y*threadSize.y, blockSize.x*threadSize.x, 1, 4, 30, pitch, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    // get data back
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(int)*output.cols, d_output, pitch, sizeof(int)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output);
    CUDA_CALL(cudaStreamDestroy(inputStream)); CUDA_CALL(cudaStreamDestroy(outputStream));

    output.convertTo(output, CV_8U);
    input.convertTo(input, CV_8U);
}