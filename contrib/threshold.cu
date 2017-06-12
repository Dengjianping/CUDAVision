#include "..\cumath\cumath.cuh"

__global__ void threshold(float *input, size_t pitch, int height, int width, float *output, float thresholdValue) 
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    __shared__ float shared[32][32];

    if (row < height && col < width) 
    {
        // load share memory
        shared[threadIdx.y][threadIdx.x] = *((float*)((char*)input + row*pitch) + col);
        __syncthreads();

        // thresholding
        float t = shared[threadIdx.y][threadIdx.x];
        t = t < thresholdValue ? 0 : 255;

        float *out = (float*)((char*)output + row*pitch) + col;
        *out = t;
    }
}

extern "C"
void cudaThreshold(cv::Mat & input, cv::Mat & output, float thresholdValue) 
{
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));

    // make sure the image is a gray image
    if (input.type() != CV_32F) 
        input.convertTo(input, CV_32F);

    float *d_input, *d_output;
    size_t pitch;

    // get pitch
    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(float)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(float)*output.cols, output.rows));

    // use stream to accelerate copy operation
    cudaStream_t inputCopy, outputCopy;
    CUDA_CALL(cudaStreamCreate(&inputCopy)); CUDA_CALL(cudaStreamCreate(&outputCopy));

    // copy data to device
    CUDA_CALL(cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, inputCopy));
    CUDA_CALL(cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, outputCopy));

    // define block size and
    dim3 threadSize(MAX_THREADS, 6);
    dim3 blockSize(input.cols / threadSize.x + 1, input.rows / threadSize.y + 1);

    threshold <<<blockSize, threadSize>>> (d_input, pitch, input.rows, input.cols, d_output, thresholdValue);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, pitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    cudaStreamDestroy(inputCopy); cudaStreamDestroy(outputCopy);
    cudaFree(d_input); cudaFree(d_output);

    output.convertTo(output, CV_8U);
    input.convertTo(input, CV_8U);
}