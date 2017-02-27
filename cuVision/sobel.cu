#include "cumath.cuh"

#define K_SIZE 3

__constant__ char sobelKernelXC[K_SIZE][K_SIZE] = { { -1.0,0.0,1.0 },{ -2.0,0.0,2.0 },{ -1.0,0.0,1.0 } };
__constant__ char sobelKernelYC[K_SIZE][K_SIZE] = { { -1.0,-2.0,-1.0 },{ 0.0,0.0,0.0 },{ 1.0,2.0,1.0 } };

__global__ void sobel(uchar *input, int height, int width, size_t inputPitch, uchar *gx, size_t gxPitch, uchar *gy, size_t gyPitch, uchar *output, size_t outputPitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ uchar localData[];

    if (row < height && col < width) {
        //int globalIndex = threadIdx.y*blockDim.x + threadIdx.x;
        //int localIndex = row*width + col;
        //localData[localIndex] = input[globalIndex];
        __syncthreads();
        // convolving
        for (size_t i = 0; i < K_SIZE; i++)
            for (size_t j = 0; j < K_SIZE; j++) {
                uchar *inputValue = (uchar*)((char*)input + row*inputPitch) + col;
                // convolving gx
                uchar *gxValue = (uchar*)((char*)gx + (row + i - K_SIZE / 2)*gxPitch) + (col + j - K_SIZE / 2);
                *gxValue += sobelKernelXC[i][j] * (*inputValue);

                // convolving gy
                uchar *gyValue = (uchar*)((char*)gy + (row + i - K_SIZE / 2)*gyPitch) + (col + j - K_SIZE / 2);
                *gyValue += sobelKernelYC[i][j] * (*inputValue);
        }

        uchar *gxValue = (uchar*)((char*)gx + row*gxPitch) + col;
        uchar *gyValue = (uchar*)((char*)gy + row*gyPitch) + col;
        uchar *outputValue = (uchar*)((char*)output + row*outputPitch) + col;
        *outputValue = *gxValue + *gyValue;
    }
}

extern "C"
void cudaSobel(cv::Mat & input, cv::Mat & output) {
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    uchar *d_input, *d_output, *gx, *gy;
    size_t inputPitch, outputPitch, gxPitch, gyPitch;
    size_t srcPitch = input.step;
    size_t dstPitch = output.step;

    cudaStream_t inputStream, outputStream, gxStream, gyStream;
    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream)); CUDA_CALL(cudaStreamCreate(&gxStream)); CUDA_CALL(cudaStreamCreate(&gyStream));

    CUDA_CALL(cudaMallocPitch(&d_input, &inputPitch, sizeof(uchar)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &outputPitch, sizeof(uchar)*output.cols, output.rows));
    CUDA_CALL(cudaMallocPitch(&gx, &gxPitch, sizeof(uchar)*output.cols, output.rows));
    CUDA_CALL(cudaMallocPitch(&gy, &gyPitch, sizeof(uchar)*output.cols, output.rows));

    std::cout << inputPitch << std::endl;

    CUDA_CALL(cudaMemcpy2DAsync(d_input, srcPitch, input.data, dstPitch, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, srcPitch, output.data, dstPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream));
    CUDA_CALL(cudaMemcpy2DAsync(gx, srcPitch, output.data, dstPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, gxStream));
    CUDA_CALL(cudaMemcpy2DAsync(gy, srcPitch, output.data, dstPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, gyStream));

    CUDA_CALL(cudaStreamSynchronize(inputStream)); CUDA_CALL(cudaStreamSynchronize(outputStream)); CUDA_CALL(cudaStreamSynchronize(gxStream)); CUDA_CALL(cudaStreamSynchronize(gyStream));

    cudaMemset(gx, 0, sizeof(uchar)*output.rows*output.cols);
    cudaMemset(gy, 0, sizeof(uchar)*output.rows*output.cols);

    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    dim3 blockSize(input.cols / (MAX_THREADS/2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS/2, MAX_THREADS);

    size_t sharedSize = blockSize.x*blockSize.y * sizeof(uchar);
    sobel<<<blockSize, threadSize, sharedSize>>>(d_input, input.rows, input.cols, dstPitch, gx, dstPitch, gy, dstPitch, d_output, dstPitch);
    CUDA_CALL(cudaDeviceSynchronize());

    // get data back
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, dstPitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output); cudaFree(gx); cudaFree(gy);
    CUDA_CALL(cudaStreamDestroy(inputStream)); CUDA_CALL(cudaStreamDestroy(outputStream)); CUDA_CALL(cudaStreamDestroy(gxStream)); CUDA_CALL(cudaStreamDestroy(gyStream));
}