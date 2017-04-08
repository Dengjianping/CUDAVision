#include "..\cumath\cumath.cuh"

__global__ void gaussianBlur(float *input, size_t inputPitch, int imageRows, int imageCols, float *output, size_t outputPitch, int radius, float theta = 1.0) {
    // use a 1-dim array to store gaussian matrix
    extern __shared__ float gaussian[];

    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < imageRows&&col < imageCols) {
        // generate a gaussian kernel, and load it to shared memory
        if (row < 2 * radius + 1 && col < 2 * radius + 1) {
            float sum = 0.0;
            gaussian[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
            __syncthreads();
        }

        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
                // convolving, about how addressing matrix in device, 
                // see this link http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
                float *inputValue = (float *)((char *)input + row*inputPitch) + col;
                float *outputValue = (float *)((char *)output + (row + i - radius)*outputPitch) + (col + j - radius);
                *outputValue += (float)(*inputValue) * gaussian[i*(2 * radius + 1) + j];
            }
    }
}


extern "C"
void cudaGaussianBlur(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0)
{
    /*
    my sample image size is 600 * 450, so we need 600 * 450 threads to process this image on device at least,
    each block can contain 1024 threads at most in my device, so ,I can define block size as 600 * 450 / 1024 = 263 (20 * 15)
    */
    input.convertTo(input, CV_32F);
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));

    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);

    // create 2 streams to asynchronously copy data to device
    cudaStream_t inputStream, outputStream;
    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream));

    // copy data to device
    float *d_input, *d_output;

    size_t inputPitch;
    CUDA_CALL(cudaMallocPitch(&d_input, &inputPitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMemcpy2DAsync(d_input, inputPitch, input.data, input.cols * sizeof(float), input.cols * sizeof(float), input.rows, cudaMemcpyHostToDevice, inputStream));

    size_t outputPitch;
    CUDA_CALL(cudaMallocPitch(&d_output, &outputPitch, sizeof(float)*output.cols, output.rows));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, outputPitch, output.data, output.cols * sizeof(float), output.cols * sizeof(float), output.rows, cudaMemcpyHostToDevice, outputStream));
    cudaStreamSynchronize(inputStream); cudaStreamSynchronize(outputStream);

    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
    gaussianBlur<<<blockSize, threadSize, dynamicSize>>> (d_input, inputPitch, input.rows, input.cols, d_output, outputPitch, radius, theta);

    CUDA_CALL(cudaDeviceSynchronize());

    // copy data back to host
    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, outputPitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaFree(d_input); cudaFree(d_output);
    cudaStreamDestroy(inputStream); cudaStreamDestroy(outputStream);

    input.convertTo(input, CV_8U);
    output.convertTo(output, CV_8U);
}