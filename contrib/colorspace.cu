#include "..\cumath\cumath.cuh"

// 0.0 is double, 0.0f is float. This will affect the performance a lot.
#define RED_WEIGHT 0.2989f
#define GREEN_WEIGHT 0.5870f
#define BLUE_WEIGHT 0.1140f
#define MAX_PIXEL(r, g, b) fmaxf(r, fmaxf(g, b))
#define MIN_PIXEL(r, g, b) fminf(r, fminf(g, b))
#define BLOCK_DIM 32


dim3 ni()
{
    dim3 threadSize;
    #if (__CUDA__ARCH__>=200&&__CUDA__ARCH__ <=210)
    threadSize = dim3(32, 6); // 1536 / 8 = 32 * 6, more detials, see CUDA_Occupancy_Calculator.xls
    #elseif (__CUDA__ARCH__>=300&&__CUDA__ARCH__ <=370)
        threadSize = dim3(32, 4); // 2048 / 16 = 32 * 4
    #elseif(__CUDA__ARCH__ >= 500 && __CUDA__ARCH__ <= 610)
        threadSize = dim3(32, 2); // 2048 / 32 = 32 * 2
    #elseif(__CUDA__ARCH__ == 620 )
        threadSize = dim3(32, 4); // 4096 / 32 = 32 * 4
    #endif
    return threadSize;
}


__global__ void grayscale(uchar *d_input, int height, int width, uchar *d_output)
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x;

    __shared__ float smem[6][32 * 3];
    for (uint i = row; i < height; i += blockDim.y*gridDim.y)
        for (uint j = col; j + threadIdx.x < width; j += blockDim.x*gridDim.x)
        {
            if (threadIdx.x < 24) // 24 * 4 = 32 * 3
            {
                uint index = 3 * (i*width + j) / 4 + threadIdx.x;
                uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[index];
                /*smem[threadIdx.y][4 * threadIdx.x] = p0.x;
                smem[threadIdx.y][4 * threadIdx.x+1] = p0.y;
                smem[threadIdx.y][4 * threadIdx.x+2] = p0.z;
                smem[threadIdx.y][4 * threadIdx.x+3] = p0.w;*/
                reinterpret_cast<float4*>(smem)[24 * threadIdx.y + threadIdx.x] = make_float4((float)p0.x, (float)p0.y, (float)p0.z, (float)p0.w);
            }
            __syncthreads();

            float gray = smem[threadIdx.y][3 * threadIdx.x] * RED_WEIGHT + GREEN_WEIGHT*smem[threadIdx.y][3 * threadIdx.x + 1] + smem[threadIdx.y][3 * threadIdx.x + 2] * BLUE_WEIGHT;
            d_output[i*width + j + threadIdx.x] = (uchar)gray;
    }
}


__device__ uchar3 hsv(float3 *rgb)
{
    float h, s, v;
    float r = rgb->x, g = rgb->y, b = rgb->z;
    v = MAX_PIXEL(r, g, b);
    float min = MIN_PIXEL(r, g, b);
    min = v - min;
    s = v != 0.0f ? 255.0f*min / v : 0.0f;

    float tmp = 60.0f / min;
    if (v == r)
        h = tmp*(g - b);
    if (v == g)
        h = 120.0f + tmp*(b - r);
    if (v == b)
        h = 240.0f + tmp*(r - b);
    h = h < 0.0f ? 360.0f + h : h;
    //h = h > 180.0f ? h : 180.0f;

    return make_uchar3((uchar)h, (uchar)s, (uchar)v);
}


//__global__ void __launch_bounds__(MAX_BLOCK_SIZE, MIN_BLOCKS_PER_SM) rgb2hsv(uchar *d_input, int height, int width, uchar *d_output)
__global__ void rgb2hsv(uchar *d_input, int height, int width, uchar *d_output)
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x;

    __shared__ float smem[6][32 * 3];
    __shared__ uchar hsv_result[6][32*3];
    for (uint i = row; i < height; i += blockDim.y*gridDim.y)
        for (uint j = col; j + threadIdx.x < width; j += blockDim.x*gridDim.x)
        {
            uint index = 3 * (i*width + j) / 4 + threadIdx.x;
            uchar4 p0;
            if (threadIdx.x < 24) // 24 * 4 = 32 * 3
            {
                p0 = reinterpret_cast<uchar4*>(d_input)[index];
                reinterpret_cast<float4*>(smem)[24 * threadIdx.y + threadIdx.x] = make_float4((float)p0.x, (float)p0.y, (float)p0.z, (float)p0.w);
            }
            __syncthreads();

            float3 gray = make_float3(smem[threadIdx.y][3 * threadIdx.x], smem[threadIdx.y][3 * threadIdx.x + 1], smem[threadIdx.y][3 * threadIdx.x + 2]);
            reinterpret_cast<uchar3*>(hsv_result)[32 * threadIdx.y + threadIdx.x] = hsv(&gray);
            __syncthreads();

            if (threadIdx.x < 24) // 24 * 4 = 32 * 3
            {
                uchar4 p1 = reinterpret_cast<uchar4*>(hsv_result)[24 * threadIdx.y + threadIdx.x];
                reinterpret_cast<uchar4*>(d_output)[index] = p1;
            }
        }
}


__device__ uchar3 hsv_(uchar3 *rgb)
{
    float h, s, v;
    float r = rgb->x, g = rgb->y, b = rgb->z;
    v = MAX_PIXEL(r, g, b);
    float min = MIN_PIXEL(r, g, b);
    min = v - min;
    s = v != 0.0f ? 255.0f*min / v : 0.0f;

    float tmp = 60.0f / min;
    if (v == r)
        h = tmp*(g - b);
    if (v == g)
        h = 120.0f + tmp*(b - r);
    if (v == b)
        h = 240.0f + tmp*(r - b);
    h = h < 0.0f ? 360.0f + h : h;
    //h = h > 180.0f ? h : 180.0f;

    return make_uchar3((uchar)h, (uchar)s, (uchar)v);
}

__global__ void rgb2hsv_(uchar *d_input, int height, int width, uchar *d_output)
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    for (uint i = row; i < height; i += blockDim.y*gridDim.y)
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            uchar3 p0 = reinterpret_cast<uchar3*>(d_input)[i*width + j];
            reinterpret_cast<uchar3*>(d_output)[i*width + j] = hsv_(&p0);
        }
}

extern "C"
void cudaGray(const cv::Mat & input, cv::Mat & output) 
{
    if (input.channels() == 1) 
    {
        input.copyTo(output);
        return;
    }
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uchar *d_input; uchar *d_output;
    cudaMalloc(&d_input, sizeof(uchar3)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar3)*input.cols*input.rows, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar)*input.cols*input.rows);

    dim3 threadSize(32, 6);
    dim3 blockSize(input.cols / (4 * threadSize.x), input.rows / (4 * threadSize.y)); // enhance threads engagement
    grayscale<<<blockSize, threadSize, 0, stream>>>(d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(output.data, d_output, sizeof(uchar)*input.cols*input.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output);
}


extern "C"
void cudaHSV(const cv::Mat & input, cv::Mat & output) 
{
    if (input.channels() == 1) 
    {
        std::cout << "This image is a single channel image, please provide a mutil-channle image" << std::endl;
        return;
    }
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uchar *d_input; uchar *d_output;
    cudaMalloc(&d_input, sizeof(uchar3)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar3)*input.cols*input.rows, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar3)*input.cols*input.rows);

    dim3 threadSize(32, 6);
    dim3 blockSize(input.cols / (4 * threadSize.x), input.rows / (4 * threadSize.y)); // enhance threads engagement
    rgb2hsv <<<blockSize, threadSize, 0, stream>>>(d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(output.data, d_output, sizeof(uchar3)*input.cols*input.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output);
}