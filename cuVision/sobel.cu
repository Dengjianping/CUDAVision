#include "cumath.cuh"

#define K_SIZE 3

__constant__ char sobelKernelXC[K_SIZE][K_SIZE] = { { -1.0,0.0,1.0 },{ -2.0,0.0,2.0 },{ -1.0,0.0,1.0 } };
__constant__ char sobelKernelYC[K_SIZE][K_SIZE] = { { -1.0,-2.0,-1.0 },{ 0.0,0.0,0.0 },{ 1.0,2.0,1.0 } };

__global__ void sobel(uchar *input, int height, int width, size_t pitch, char *gx, char *gy, uchar *output) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ uchar localData[];

    if (row < height && col < width) {
        // load data to share memory
        localData[threadIdx.y*blockDim.x+threadIdx.x] = *(uchar*)((char*)input + row*pitch) + col;
        __syncthreads();

        // convolving
        for (size_t i = 0; i < K_SIZE; i++)
            for (size_t j = 0; j < K_SIZE; j++) {
                uchar *inputValue = (uchar*)((char*)input + row*pitch) + col;
                // convolving gx
                if (row + i - K_SIZE / 2 >= 0 && col + j - K_SIZE / 2 >= 0) {
                    char *gxValue = (char*)((char*)gx + (row + i - K_SIZE / 2)*pitch) + (col + j - K_SIZE / 2);
                    *gxValue += sobelKernelXC[i][j] * (*inputValue);
                    //*gxValue += sobelKernelXC[i][j] * localData[threadIdx.y*blockDim.x + threadIdx.x];
                    //__syncthreads();

                    // convolving gy
                    char *gyValue = (char*)((char*)gy + (row + i - K_SIZE / 2)*pitch) + (col + j - K_SIZE / 2);
                    //*gyValue += sobelKernelYC[i][j] * localData[threadIdx.y*blockDim.x + threadIdx.x];
                    //__syncthreads();
                    *gyValue += sobelKernelYC[i][j] * (*inputValue);
                }
        }

        char *gxValue = (char*)((char*)gx + row*pitch) + col;
        char *gyValue = (char*)((char*)gy + row*pitch) + col;
        uchar *outputValue = (uchar*)((char*)output + row*pitch) + col;
        *outputValue = sqrtf(powf((float)*gxValue,2) + powf((float)*gyValue,2));
    }
}

extern "C"
void cudaSobel(cv::Mat & input, cv::Mat & output) {
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    uchar *d_input, *d_output; char *gx, *gy;
    size_t pitch;

    cudaStream_t inputStream, outputStream, gxStream, gyStream;
    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream)); CUDA_CALL(cudaStreamCreate(&gxStream)); CUDA_CALL(cudaStreamCreate(&gyStream));

    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(uchar)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(uchar)*output.cols, output.rows));
    CUDA_CALL(cudaMallocPitch(&gx, &pitch, sizeof(char)*output.cols, output.rows));
    CUDA_CALL(cudaMallocPitch(&gy, &pitch, sizeof(char)*output.cols, output.rows));

    CUDA_CALL(cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream));
    CUDA_CALL(cudaMemcpy2DAsync(gx, pitch, output.data, sizeof(char)*output.cols, sizeof(char)*output.cols, output.rows, cudaMemcpyHostToDevice, gxStream));
    CUDA_CALL(cudaMemcpy2DAsync(gy, pitch, output.data, sizeof(char)*output.cols, sizeof(char)*output.cols, output.rows, cudaMemcpyHostToDevice, gyStream));

    CUDA_CALL(cudaStreamSynchronize(inputStream)); CUDA_CALL(cudaStreamSynchronize(outputStream)); CUDA_CALL(cudaStreamSynchronize(gxStream)); CUDA_CALL(cudaStreamSynchronize(gyStream));

    cudaMemset(gx, 0, sizeof(char)*output.rows*output.cols);
    cudaMemset(gy, 0, sizeof(char)*output.rows*output.cols);

    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    dim3 blockSize(input.cols / (MAX_THREADS/2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS/2, MAX_THREADS);

    size_t shared = threadSize.x*threadSize.y * sizeof(uchar);
    sobel<<<blockSize, threadSize, shared>>>(d_input, input.rows, input.cols, pitch, gx, gy, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    // get data back
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output); cudaFree(gx); cudaFree(gy);
    CUDA_CALL(cudaStreamDestroy(inputStream)); CUDA_CALL(cudaStreamDestroy(outputStream)); CUDA_CALL(cudaStreamDestroy(gxStream)); CUDA_CALL(cudaStreamDestroy(gyStream));
}