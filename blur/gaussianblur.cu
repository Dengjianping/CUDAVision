#include "..\cumath\cumath.cuh"

#define TILE_H 32
#define TILE_W 32
#define RADIUS 8

__global__ void gaussianBlur(float *input, size_t pitch, int height, int width, float *output, int radius, float theta = 1.0)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int p = 2 * radius + 1; // gaussian kernel length for address shared array, bias

    extern __shared__ float gaussian[];
    extern __shared__ float shared[];
    if (row < height && col < width)
    {
        // generate a gaussian kernel, and load it to shared memory
        if (threadIdx.x < p && threadIdx.y < p)
            gaussian[threadIdx.y*p + threadIdx.x] = twoDimGaussian(threadIdx.x - radius, radius - threadIdx.y, theta);
        __syncthreads();
        /*
            Do not load share memory like this, which will load incompletely,
            if (threadIdx.x < 2 * radius + 1 && threadIdx.y < 2 * radius + 1)
            {
                gaussian[threadIdx.y*(2 * radius + 1) + threadIdx.x] = twoDimGaussian(threadIdx.x - radius, radius - threadIdx.y, theta);
                __syncthreads();
            }
        */
        

        // upper left corner in the block
        if (row - RADIUS < 0 || col - RADIUS < 0)
            shared[threadIdx.y*TILE_W + threadIdx.x +p*p] = 0;
        else
            shared[threadIdx.y*TILE_W + threadIdx.x + p*p] = *((float*)((char*)input + (row - RADIUS) * pitch) + (col - RADIUS));

        // upper right
        if (row - RADIUS < 0 || col + RADIUS > width - 1)
            shared[threadIdx.y*TILE_W + (threadIdx.x + blockDim.x) + p*p] = 0;
        else
            shared[threadIdx.y*TILE_W + (threadIdx.x + blockDim.x) + p*p] = *((float*)((char*)input + (row - RADIUS) * pitch) + (col + RADIUS));

        //bottom left
        if (row + RADIUS > height - 1 || col - RADIUS < 0)
            shared[(threadIdx.y + blockDim.y)*TILE_W + threadIdx.x + p*p] = 0;
        else
            shared[(threadIdx.y + blockDim.y)*TILE_W + threadIdx.x + p*p] = *((float*)((char*)input + (row + RADIUS) * pitch) + (col - RADIUS));

        // bottom right
        if (row + RADIUS > height - 1 || col - RADIUS > width - 1)
            shared[(threadIdx.y + blockDim.y)*TILE_W + (threadIdx.x + blockDim.x) + p*p] = 0;
        else
            shared[(threadIdx.y + blockDim.y)*TILE_W + (threadIdx.x + blockDim.x) + p*p] = *((float*)((char*)input + (row + RADIUS) * pitch) + (col + RADIUS));
        __syncthreads();

        float sum = 0;
        for (int i = -radius; i <= radius; i++)
            for (int j = -radius; j <= radius; j++)
            {
                sum += gaussian[(radius + i)*p + radius + j] * shared[(threadIdx.y + RADIUS - i)*TILE_W + (threadIdx.x + RADIUS - j) + p*p];
            }
        float *out = (float *)((char *)output + row*pitch) + col;
        *out = sum;
    }
}

extern "C"
void cudaGaussianBlur(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0)
{
    input.convertTo(input, CV_32F);
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));

    // my video card is nvs4200, if I want to use max occupanc(100%)y to this kernel, I need to use 32 * 6 threads in each block.
    dim3 threadSize(16, 16);
    dim3 blockSize(input.cols / threadSize.x + 1, input.rows / threadSize.y + 1);

    // create stream to asynchronously copy data to device
    cudaStream_t in_stream, out_stream; 
    CUDA_CALL(cudaStreamCreate(&in_stream));
    CUDA_CALL(cudaStreamCreate(&out_stream));

    // copy data to device
    float *d_input,*d_output; 
    size_t pitch;
    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(float)*output.cols, output.rows));
    CUDA_CALL(cudaMemcpy2DAsync(d_input, pitch, input.data, input.cols * sizeof(float), input.cols * sizeof(float), input.rows, cudaMemcpyHostToDevice, in_stream));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, pitch, output.data, output.cols * sizeof(float), output.cols * sizeof(float), output.rows, cudaMemcpyHostToDevice, in_stream));

    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float) + TILE_H*TILE_W*sizeof(float);
    gaussianBlur<<<blockSize, threadSize, dynamicSize, in_stream>>>(d_input, pitch, input.rows, input.cols, d_output, radius, theta);
    CUDA_CALL(cudaDeviceSynchronize());

    // copy data back to host
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, pitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output);
    cudaStreamDestroy(in_stream); cudaStreamDestroy(out_stream);

    input.convertTo(input, CV_8U);
    output.convertTo(output, CV_8U);
}