#include "..\cumath\cumath.cuh"

struct __align__(16) float3_aligned { float x, y, z; };
struct __align__(4) uchar3_aligned { uchar x, y, z; };

__global__ void usmcolor(uchar3_aligned *d_input, // target data
                         size_t pitch, // width allocated by cudaMallocPitch
                         int height, int width, // image rows and cols
                         uchar3_aligned *d_output, // output
                         int radius, float theta, float weight) 
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ float gaussianKernel[];
    //__shared__ uchar3_aligned shared_pixels[6][32];
    __shared__ float shared_pixels[6*3][32];
    if (row < height && col < width) 
    {
        uchar3_aligned inputPixel = *((uchar3_aligned*)((char*)d_input + row*pitch) + col);
        // use share memory to fix global memory coalescing
        shared_pixels[threadIdx.y][threadIdx.x] = inputPixel.x;
        shared_pixels[threadIdx.y+6][threadIdx.x] = inputPixel.y;
        shared_pixels[threadIdx.y+12][threadIdx.x] = inputPixel.z;

        // gaussian kernel
        if (threadIdx.x < 2 * radius + 1 && threadIdx.y < 2 * radius + 1)
            gaussianKernel[threadIdx.y*(2 * radius + 1) + threadIdx.x] = twoDimGaussian(col - radius, radius - row, theta);
        __syncthreads();

        // get gaussian blurring data, convolving
        float3_aligned blurredPixel = { 0,0,0 };
        for (int i = -radius; i <= radius; i++)
            for (int j = -radius; j <= radius; j++) 
            {
                blurredPixel.x += gaussianKernel[(radius + i)*(2 * radius + 1) + (radius + j)] * shared_pixels[threadIdx.y][threadIdx.x]; // r channel
                blurredPixel.y += gaussianKernel[(radius + i)*(2 * radius + 1) + (radius + j)] * shared_pixels[threadIdx.y+6][threadIdx.x]; // g channel
                blurredPixel.z += gaussianKernel[(radius + i)*(2 * radius + 1) + (radius + j)] * shared_pixels[threadIdx.y+12][threadIdx.x]; // b channel
            }

        uchar3_aligned *outputPixel = (uchar3_aligned*)((char*)d_output + row*pitch) + col;
        uchar3_aligned blurredPixel1 = { 0,0,0 };
        blurredPixel1.x = (shared_pixels[threadIdx.y][threadIdx.x] - weight*blurredPixel.x) / (1.0 - weight); // r channel
        blurredPixel1.y = (shared_pixels[threadIdx.y+6][threadIdx.x] - weight*blurredPixel.y) / (1.0 - weight); // g channel
        blurredPixel1.z = (shared_pixels[threadIdx.y+12][threadIdx.x] - weight*blurredPixel.z) / (1.0 - weight); // b channel
        *outputPixel = blurredPixel1;
    }
}

extern "C"
void cudaUSMColor(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0, float weight = 0.6) 
{
    if (input.channels() != 3)
    {
        std::cout << "this image is not a 3-ch image" << std::endl;
        return;
    }

    //input.convertTo(input, CV_32FC3);
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0, 0, 0));

    dim3 threadSize(32, 6);
    dim3 blockSize(input.cols / threadSize.x + 1, input.rows / threadSize.y + 1);

    uchar3_aligned *d_input, *d_output;
    size_t pitch;
    cudaStream_t inputStream, outputStream;

    CUDA_CALL(cudaMallocPitch(&d_input, &pitch, sizeof(uchar3_aligned)*input.cols, input.rows));
    CUDA_CALL(cudaMallocPitch(&d_output, &pitch, sizeof(uchar3_aligned)*output.cols, output.rows));

    CUDA_CALL(cudaStreamCreate(&inputStream)); CUDA_CALL(cudaStreamCreate(&outputStream));
    CUDA_CALL((cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(uchar3)*input.cols, sizeof(uchar3)*input.cols, input.rows, cudaMemcpyHostToDevice, inputStream)));
    CUDA_CALL((cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(uchar3)*output.cols, sizeof(uchar3)*output.cols, output.rows, cudaMemcpyHostToDevice, outputStream)));

    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
    usmcolor<<<blockSize, threadSize, dynamicSize>>> (d_input, pitch, input.rows, input.cols, d_output, radius, theta, weight);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar3)*output.cols, d_output, pitch, sizeof(uchar3)*output.cols, output.rows, cudaMemcpyDeviceToHost));

    cudaStreamDestroy(inputStream); cudaStreamDestroy(outputStream);
    cudaFree(d_input); cudaFree(d_output);

    //output.convertTo(output, CV_8UC3);
    //input.convertTo(input, CV_8UC3);
}