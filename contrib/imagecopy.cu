#include "..\cumath\cumath.cuh"

struct __align__(8) uchar8 { unsigned char a, b, c, d, m, n, p, q; };

__global__ void copyTo(uchar *input, int height, int width, uchar *output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    //int row = blockDim.y*blockIdx.y + 2 * threadIdx.y; // each thread handle 8 pixels
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (uint i = row; i < height / 4; i += blockDim.y*gridDim.y) // stride by 4 byte
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            reinterpret_cast<uchar4*>(output)[i*width + j] = reinterpret_cast<uchar4*>(input)[i*width + j];
            //reinterpret_cast<uchar4*>(output)[(i + 1)*width + j] = reinterpret_cast<uchar4*>(input)[(i + 1)*width + j];
            //reinterpret_cast<uchar4*>(output)[(i + 2)*width + j] = reinterpret_cast<uchar4*>(input)[(i + 2)*width + j];
            //reinterpret_cast<uchar4*>(output)[(i + 3)*width + j] = reinterpret_cast<uchar4*>(input)[(i + 3)*width + j];
        }
}

extern "C"
void cudaCopyTo(const cv::Mat & input, cv::Mat & output)
{
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    uchar *d_input, *d_output;
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows);
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows);

    // define block size and
    dim3 threadSize(MAX_THREADS, 6);
    dim3 blockSize(input.cols / (4 * threadSize.x), input.rows / (8 * threadSize.y)); // I divide the image into 16 grid to increase ILP level.

    copyTo<<<blockSize, threadSize, 0, stream>>>(d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resources releasing
    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output);
}