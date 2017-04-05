#include "cumath.cuh"
#include "cuda_fp16.h"

#define K_SIZE 3
#define TILE_H 4
#define TILE_W 30
#define LOOP_UNROLLING

texture<uchar, 2, cudaReadModeElementType> text2D;

__constant__ int sobelKernelXC[K_SIZE][K_SIZE] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
__constant__ int sobelKernelYC[K_SIZE][K_SIZE] = { { -1,-2,-1 },{ 0,0,0 },{ 1,2,1 } };

__global__ void sobel(float *input, int height, int width, size_t pitch, float *output) 
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    //extern __shared__ uchar localData[];

    if (row > 0 && row < height - 1 && col>0 && col < width - 1) {
        // load data to share memory
        //localData[threadIdx.y*blockDim.x + threadIdx.x] = *(uchar*)((char*)input + row*pitch) + col;
        //__syncthreads();

        // convolving
        float sumx = 0;
        float sumy = 0;
        for (int i = -K_SIZE / 2; i <= K_SIZE/2; i++)
            for (int j = -K_SIZE / 2; j <= K_SIZE / 2; j++) 
            {
                float *inputValue = (float*)((char*)input + (row - i)*pitch) + (col - j);
                //float in = tex2D(text2D, col - j, row - i);
                // convolving gx
                sumx += sobelKernelXC[K_SIZE / 2 + i][K_SIZE / 2 + j] * (*inputValue);
                //sumx += sobelKernelXC[K_SIZE / 2 + i][K_SIZE / 2 + j] * in;
                //sumx += sobelKernelXC[K_SIZE / 2 + i][K_SIZE / 2 + j] * localData[threadIdx.y*blockDim.x + threadIdx.x];
                //__syncthreads();

                // convolving gy
                //sumy += sobelKernelYC[K_SIZE / 2 + i][K_SIZE / 2 + j] * localData[threadIdx.y*blockDim.x + threadIdx.x];
                //__syncthreads();
                sumy += sobelKernelYC[K_SIZE / 2 + i][K_SIZE / 2 + j] * (*inputValue);
                //sumy += sobelKernelYC[K_SIZE / 2 + i][K_SIZE / 2 + j] * in;
            }

        float *outputValue = (float*)((char*)output + row*pitch) + col;
        //*outputValue = sqrtf(powf((float)*gxValue, 2) + powf((float)*gyValue, 2));
        *outputValue = (sqrtf(powf(sumx, 2) + powf(sumy, 2)));
    }
}

__global__ void sobel1(float *input, int height, int width, int radius, int tile_h, int tile_w, int max_blockIdx_x, int max_blockIdx_y, size_t pitch, float *output)
{
    int row = blockIdx.y*TILE_H + threadIdx.y - radius;
    int col = blockIdx.x*TILE_W + threadIdx.x - radius;

    __shared__ float localData[TILE_H + K_SIZE - 1][TILE_W + K_SIZE - 1];
    if (row < height && col < width) {
        if (row < 0 || col < 0) 
        {
            localData[threadIdx.y][threadIdx.x] = 0;
        }
        /*if (row > height + radius-1 || col > width +radius - 1)
        {
            localData[threadIdx.y][threadIdx.x] = 0;
        }*/
        /*if (blockIdx.y == max_blockIdx_y - 1)
        {
            int remain = height % tile_h != 0 ? tile_h : height % tile_h;
            if (threadIdx.y > remain + radius - 1)
            {
                localData[threadIdx.y][threadIdx.x] = 0;
            }
        }
        if (blockIdx.x == max_blockIdx_x - 1)
        {
            int remain = width % tile_w != 0 ? tile_h : width % tile_w;
            if (threadIdx.x > remain + radius - 1)
            {
                localData[threadIdx.y][threadIdx.x] = 0;
            }
        }*/
        /*if (row == height - 1 || col == width - 1)
        {
            localData[threadIdx.y][threadIdx.x] = 0;
        }*/
        /*if (blockIdx.y == 5)
        {
            if (threadIdx.y > 1)
            {
                localData[threadIdx.y][threadIdx.x] = 0;
            }
            else
            {
                localData[threadIdx.y][threadIdx.x] = *((float*)((char*)input + (blockIdx.y*TILE_H + threadIdx.y - radius)*pitch) + (blockIdx.x*TILE_W + threadIdx.x - radius));
            }
        }*/
        else 
        {
            localData[threadIdx.y][threadIdx.x] = *((float*)((char*)input + row*pitch) + col);
            //localData[threadIdx.y][threadIdx.x] = *((float*)((char*)input + (row - radius)*pitch) + (col - radius));
            //localData[threadIdx.y][threadIdx.x] = *((float*)((char*)input + (225)*pitch) + (300));
        }
        __syncthreads();
        
        if ((threadIdx.y >= radius && threadIdx.y <= blockDim.y - radius - 1) && (threadIdx.x >= radius && threadIdx.x <= blockDim.x - radius - 1)) 
        {
            /*int sumx = 0, sumy = 0;
            #pragma unroll
            for (int i = -radius; i <= radius; i++)
                for (int j = -radius; j <= radius; j++) 
                {
                    sumx += sobelKernelXC[radius + i][radius + j] * localData[threadIdx.y - i][threadIdx.x - j];
                    sumy += sobelKernelYC[radius + i][radius + j] * localData[threadIdx.y - i][threadIdx.x - j];
                }*/
            // use loop unrolling to improve performance, it can avoid branching.
            float sumx = 0, sumy = 0;
#ifdef LOOP_UNROLLING
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
            __syncthreads();
            float *out = (float*)((char*)output + row*pitch) + col;
            //float *out = (float*)((char*)output + (row - radius)*pitch) + (col - radius);
            *out = sqrtf(powf(sumx, 2) + powf(sumy, 2));
        }
    }
}

extern "C"
void cudaSobel(cv::Mat & input, cv::Mat & output) 
{
    input.convertTo(input, CV_32F);
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));

    float *d_input, *d_output;
    size_t pitch;

    cudaStream_t inputStream, outputStream;
    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream));

    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(float)*output.cols, output.rows));
  
    CUDA_CALL(cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream));

    // setup texture
    //text2D.filterMode = cudaFilterModePoint;
    //text2D.addressMode[0] = cudaAddressModeWrap;
    //text2D.addressMode[1] = cudaAddressModeWrap;

    //cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
    //cudaBindTexture2D(NULL, &text2D, d_input, &desc, input.cols, input.rows, pitch);

    /* int max_blockIdx_x, int max_blockIdx_y,
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */ 
    //dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);
    dim3 threadSize(32, 6);
    //dim3 blockSize(input.cols / threadSize.x + 1, input.rows / threadSize.y + 1);
    dim3 blockSize(input.cols / TILE_W + 1, input.rows / TILE_H + 1);

    size_t shared = threadSize.x*threadSize.y * sizeof(float);
    sobel1<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, 1, 4, 30, blockSize.x, blockSize.y, pitch, d_output);
    //sobel1 <<<blockSize, threadSize>>>(d_input, input.rows, input.cols, K_SIZE/2, pitch, d_output);
    //sobel<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, pitch, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    // get data back
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, pitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output);
    CUDA_CALL(cudaStreamDestroy(inputStream)); CUDA_CALL(cudaStreamDestroy(outputStream));
    //cudaUnbindTexture(&text2D);
    output.convertTo(output, CV_8U);
    input.convertTo(input, CV_8U);
}