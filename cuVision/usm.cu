#include "cumath.cuh"

#define K_SIZE 3

//__device__ float blurredArray[3][3];

__device__ float twoDimGaussians(int x, int y, float theta) {
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}


__global__ void usm(float *d_input, size_t inputPitch, int imageRows, int imageCols, float *blurred, size_t blurredPitch, float *d_output, size_t outputPitch, int radius, float theta, float weight) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ float gaussianKernel[];

    if (row < imageRows&&col < imageCols) {
        float *inputPixel = (float*)((char*)d_input + row*inputPitch) + col;      

        // gaussian kernel
        if (row < 2 * radius + 1 && col < 2 * radius + 1) {
            gaussianKernel[row*(2 * radius + 1) + col] = twoDimGaussians(col - radius, radius - row, theta);
            __syncthreads();
        }

        // get gaussian blurring data, convolving
        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
                float *blurredPixel = (float*)((char*)blurred + (row + i - radius)*blurredPitch) + (col + j - radius);
                *blurredPixel += gaussianKernel[i*(2 * radius + 1) + j] * (*inputPixel);
            }

        float *outputPixel = (float*)((char*)d_output + row*outputPitch) + col;
        float *sblurredPixel = (float*)((char*)blurred + row*blurredPitch) + col;
        *outputPixel = ((*inputPixel) - weight*(*sblurredPixel)) / (1.0 - weight);
    }
}

extern "C"
void cudaUSM(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0, float weight = 0.6) {
    input.convertTo(input, CV_32F);
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));

    //dim3 blockSize(2 * (input.cols / MAX_THREADS + 1), input.rows / MAX_THREADS + 1);
    dim3 blockSize(input.cols / (MAX_THREADS/2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS/2, MAX_THREADS);

    float *d_input, *d_blurred, *d_output;
    size_t inputPitch, blurredPitch, outputPitch;
    cudaStream_t inputStream, blurredStream, outputStream;

    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&blurredStream)); CUDA_CALL(cudaStreamCreate(&outputStream));
    CUDA_CALL(cudaMallocPitch(&d_input, &inputPitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_blurred, &blurredPitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &outputPitch, sizeof(float)*output.cols, output.rows));

    CUDA_CALL((cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream)));
    CUDA_CALL((cudaMemcpy2DAsync(d_blurred, blurredPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, blurredStream)));
    CUDA_CALL((cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream)));

    CUDA_CALL(cudaStreamSynchronize(inputStream)); CUDA_CALL(cudaStreamSynchronize(blurredStream)); CUDA_CALL(cudaStreamSynchronize(outputStream));

    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
    usm <<<blockSize, threadSize, dynamicSize>>> (d_input, inputPitch, input.rows, input.cols, d_blurred, blurredPitch, d_output, outputPitch, radius, theta, weight);
    
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, outputPitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    cudaStreamDestroy(inputStream); cudaStreamDestroy(blurredStream); cudaStreamDestroy(outputStream);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_blurred);

    output.convertTo(output, CV_8U);
    input.convertTo(input, CV_8U);
}