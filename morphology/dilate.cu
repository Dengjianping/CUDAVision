#include "..\cumath\cumath.cuh"

#define K_SIZE 3
#define RADIUS 8

__constant__ float erodeKernel[K_SIZE][K_SIZE] = { { 0,1,0 },{ 1,1,1 },{ 0,1,0 } };

__global__ void dilate(float *input, size_t pitch, int height, int width, int radius, float *output) 
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ float shared[32][32];

    if (row < height&&col < width) 
    {
        if (row - RADIUS < 0 || col - RADIUS < 0)
            shared[threadIdx.y][threadIdx.x] = 0;
        else
            shared[threadIdx.y][threadIdx.x] = *((float*)((char*)input + (row - RADIUS)*pitch) + (col - RADIUS));

        if (row - RADIUS<0 || col + RADIUS>width - 1)
            shared[threadIdx.y][threadIdx.x + blockDim.x] = 0;
        else
            shared[threadIdx.y][threadIdx.x + blockDim.x] = *((float*)((char*)input + (row - RADIUS)*pitch) + (col + RADIUS));

        if (row + RADIUS > height - 1 || col - RADIUS < 0)
            shared[threadIdx.y + blockDim.y][threadIdx.x] = 0;
        else
            shared[threadIdx.y + blockDim.y][threadIdx.x] = *((float*)((char*)input + (row + RADIUS)*pitch) + (col - RADIUS));

        if (row + RADIUS > height - 1 || col + RADIUS > width - 1)
            shared[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = 0;
        else
            shared[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = *((float*)((char*)input + (row + RADIUS)*pitch) + (col + RADIUS));
        __syncthreads();

        float sum = 0;
        for (int i = -radius; i <= radius; i++)
            for (int j = -radius; j <= radius; j++)
            {
                sum += shared[threadIdx.y + RADIUS - i][threadIdx.x + RADIUS - j] * erodeKernel[radius + i][radius + j];
            }
        float *out = (float*)((char*)output + row*pitch) + col;
        *out = sum;
    }
}

extern "C"
void cudaDilate(cv::Mat & input, cv::Mat & output) 
{
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));

    if (input.type() != CV_32F) input.convertTo(input, CV_32F);

    float *d_input, *d_output;
    size_t pitch;

    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(float)*output.cols, output.rows));

    cudaStream_t inputCopy, outputCopy;
    CUDA_CALL(cudaStreamCreate(&inputCopy)); CUDA_CALL(cudaStreamCreate(&outputCopy));

    CUDA_CALL(cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopy));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopy));

    dim3 threadSize(16, 16);
    dim3 blockSize(input.cols / threadSize.x + 1, input.rows / threadSize.y + 1);

    int radius = K_SIZE / 2;
    dilate <<<blockSize, threadSize>>> (d_input, pitch, input.rows, input.cols, radius, d_output);

    CUDA_CALL(cudaDeviceSynchronize());
    

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, pitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaStreamDestroy(inputCopy); cudaStreamDestroy(outputCopy);
    cudaFree(d_input); cudaFree(d_output);
    output.convertTo(output, CV_8U);
}