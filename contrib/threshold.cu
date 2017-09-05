#include "..\cumath\cumath.cuh"

#define VALUE(x) x < thresholdValue ? 0 : 255

__device__ void handle_uchar4(uchar4* input, uchar thresholdValue)
{
    input->x = VALUE(input->x);
    input->y = VALUE(input->y);
    input->z = VALUE(input->z);
    input->w = VALUE(input->w);
}


__global__ void threshold(uchar *input, int height, int width, uchar *output, uchar thresholdValue)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    // use blockDim.y*gridDim.y or blockDim.x*gridDim.x to retrieve elements, increasing ILP level, and memory accessing is coalesced.
    // visit this post, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (uint i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            // because use uchar, I just get 25% loading efficiency on global memory, a warp only request only 32 bytes, but ideally 128 bytes.
            // There's a post from CUDA blog, https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
            // as result, it benefit a lot performance improvement.
            uchar4 temp = reinterpret_cast<uchar4*>(input)[i*width + j];
            handle_uchar4(&temp, thresholdValue);
            reinterpret_cast<uchar4*>(output)[i*width + j] = temp;
        }
}


extern "C"
void cudaThreshold(cv::Mat & input, cv::Mat & output, uchar thresholdValue, bool zero_copy=false)
{
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
    // define block size and
    dim3 threadSize(MAX_THREADS, 6);
    dim3 blockSize(input.cols / (4 * threadSize.x), input.rows / (4 * threadSize.y)); // I divide the image into 16 grid to increase ILP level.
    if (!zero_copy)
    {
        uchar *d_input, *d_output;
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows));
        CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows));

        threshold <<<blockSize, threadSize, 0, stream>>> (d_input, input.rows, input.cols, d_output, thresholdValue);
        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

        // resources releasing
        CUDA_CALL(cudaStreamDestroy(stream));
        CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
    }
    else
    {
        CUDA_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));

        // bind input data
        uchar *h_input, *d_input;
        CUDA_CALL(cudaHostAlloc(&h_input, sizeof(uchar)*input.cols*input.rows, cudaHostAllocMapped));
        memcpy(h_input, input.data, sizeof(uchar)*input.cols*input.rows);
        CUDA_CALL(cudaHostGetDevicePointer(&d_input, d_input, 0));

        // bind output data
        uchar *h_output, *d_output;
        CUDA_CALL(cudaHostAlloc(&h_output, sizeof(uchar)*input.cols*input.rows, cudaHostAllocMapped));
        memset(h_output, 0, sizeof(uchar)*input.cols*input.rows);
        CUDA_CALL(cudaHostGetDevicePointer(&d_output, h_output, 0));

        threshold <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, d_output, thresholdValue);
        CUDA_CALL(cudaDeviceSynchronize());

        memcpy(output.data, h_input, sizeof(uchar)*input.cols*input.rows);
        CUDA_CALL(cudaFreeHost(h_input)); CUDA_CALL(cudaFreeHost(h_output));
    }

}