#include "..\cumath\cumath.cuh"

#define K_SIZE 3
#define FACTOR 0.5

__constant__ float P = 3.1415;

__global__ void usmcolor(float3 *d_input, size_t inputPitch, int imageRows, int imageCols, float3 *blurred, size_t blurredPitch, float3 *d_output, size_t outputPitch, int radius, float theta, float weight) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ float gaussianKernel[];

    if (row < imageRows&&col < imageCols) {
        float3 *inputPixel = (float3*)((char*)d_input + row*inputPitch) + col;

        // gaussian kernel
        if (row < 2 * radius + 1 && col < 2 * radius + 1) {
            gaussianKernel[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
            __syncthreads();
        }

        // get gaussian blurring data, convolving
        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
                float3 *blurredPixel = (float3*)((char*)blurred + (row + i - radius)*blurredPitch) + (col + j - radius);

                blurredPixel->x += gaussianKernel[i*(2 * radius + 1) + j] * inputPixel->x; // r channel
                blurredPixel->y += gaussianKernel[i*(2 * radius + 1) + j] * inputPixel->y; // g channel
                blurredPixel->z += gaussianKernel[i*(2 * radius + 1) + j] * inputPixel->z; // b channel
            }

        float3 *outputPixel = (float3*)((char*)d_output + row*outputPitch) + col;
        float3 *sblurredPixel = (float3*)((char*)blurred + row*blurredPitch) + col;

        outputPixel->x = (inputPixel->x - weight*sblurredPixel->x) / (1.0 - weight); // r channel
        outputPixel->y = (inputPixel->y - weight*sblurredPixel->y) / (1.0 - weight); // g channel
        outputPixel->z = (inputPixel->z - weight*sblurredPixel->z) / (1.0 - weight); // b channel
    }
}

extern "C"
void cudaUSMColor(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0, float weight = 0.6) {
    input.convertTo(input, CV_32FC3);
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0, 0, 0));

    //dim3 blockSize(2 * (input.cols / MAX_THREADS + 1), input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS, 1);
    //dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 blockSize(ceil((float)input.cols / threadSize.x), ceil((float)input.rows / threadSize.y), 1);


    float3 *d_input, *d_blurred, *d_output;
    size_t inputPitch, blurredPitch, outputPitch;
    cudaStream_t inputStream, blurredStream, outputStream;

    CUDA_CALL(cudaMallocPitch(&d_input, &inputPitch, sizeof(float3)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_blurred, &blurredPitch, sizeof(float3)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &outputPitch, sizeof(float3)*output.cols, output.rows));

    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&blurredStream)); CUDA_CALL(cudaStreamCreate(&outputStream));

    CUDA_CALL((cudaMemcpy2DAsync(d_input, inputPitch, input.data, sizeof(float3)*input.cols, sizeof(float3)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream)));
    CUDA_CALL((cudaMemcpy2DAsync(d_blurred, blurredPitch, output.data, sizeof(float3)*output.cols, sizeof(float3)*output.cols, output.rows, cudaMemcpyHostToDevice, blurredStream)));
    CUDA_CALL((cudaMemcpy2DAsync(d_output, outputPitch, output.data, sizeof(float3)*output.cols, sizeof(float3)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream)));

    CUDA_CALL(cudaStreamSynchronize(inputStream)); CUDA_CALL(cudaStreamSynchronize(blurredStream)); CUDA_CALL(cudaStreamSynchronize(outputStream));

    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
    usmcolor << <blockSize, threadSize, dynamicSize >> > (d_input, inputPitch, input.rows, input.cols, d_blurred, blurredPitch, d_output, outputPitch, radius, theta, weight);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float3)*output.cols, d_output, outputPitch, sizeof(float3)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    cudaStreamDestroy(inputStream); cudaStreamDestroy(blurredStream); cudaStreamDestroy(outputStream);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_blurred);

    output.convertTo(output, CV_8UC3);
    input.convertTo(input, CV_8UC3);
}