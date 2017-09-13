#include "..\cumath\cumath.cuh"

enum DATA_TYPE { INT32, FLOAT32 };

__global__ void convert(uchar *d_input, int height, int width, float *d_output)
{
    //int row = blockDim.y*blockIdx.y + 4 * threadIdx.y; // one thread loads 16 pixels
    int row = blockDim.y*blockIdx.y + 2 * threadIdx.y; // one thread loads 8 pixels
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    for (uint i = row; i < height / 4; i += blockDim.y*gridDim.y)
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)
        {
            uint index = i*width + j;
        
            uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[index];
            uchar4 p1 = reinterpret_cast<uchar4*>(d_input)[index + width];
            //uchar4 p2 = reinterpret_cast<uchar4*>(d_input)[index + 2 * width];
            //uchar4 p3 = reinterpret_cast<uchar4*>(d_input)[index + 3 * width];

            reinterpret_cast<float4*>(d_output)[index] = make_float4((float)p0.x, (float)p0.y, (float)p0.z, (float)p0.w);
            reinterpret_cast<float4*>(d_output)[index + width] = make_float4((float)p1.x, (float)p1.y, (float)p1.z, (float)p1.w);
            //reinterpret_cast<float4*>(d_output)[index + 2 * width] = make_float4((float)p2.x, (float)p2.y, (float)p2.z, (float)p2.w);
            //reinterpret_cast<float4*>(d_output)[index + 3 * width] = make_float4((float)p3.x, (float)p3.y, (float)p3.z, (float)p3.w);
        }
}


__global__ void convert_(uchar *d_input, int height, int width, float *d_output)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y; 
    int col = blockDim.x*blockIdx.x + threadIdx.x;

   if(row<height/4&&col<width)
   {

        uint index = row*width + col;

        uchar4 p0 = reinterpret_cast<uchar4*>(d_input)[index];
        //uchar4 p1 = reinterpret_cast<uchar4*>(d_input)[index + width];
        //uchar4 p2 = reinterpret_cast<uchar4*>(d_input)[index + 2 * width];
        //uchar4 p3 = reinterpret_cast<uchar4*>(d_input)[index + 3 * width];

        reinterpret_cast<float4*>(d_output)[index] = make_float4((float)p0.x, (float)p0.y, (float)p0.z, (float)p0.w);
        //reinterpret_cast<float4*>(d_output)[index + width] = make_float4((float)p1.x, (float)p1.y, (float)p1.z, (float)p1.w);
        //reinterpret_cast<float4*>(d_output)[index + 2 * width] = make_float4((float)p2.x, (float)p2.y, (float)p2.z, (float)p2.w);
        //reinterpret_cast<float4*>(d_output)[index + 3 * width] = make_float4((float)p3.x, (float)p3.y, (float)p3.z, (float)p3.w);
    }
}

extern "C"
void cudaConvert2F32(const cv::Mat & input, cv::Mat & output) 
{
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));
    uchar *d_input; float *d_output;

    cudaStream_t stream; CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaMalloc(&d_input, sizeof(uchar)*input.rows*input.cols));
    CUDA_CALL(cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMalloc(&d_output, sizeof(float)*input.rows*input.cols));

    // define block size and
    dim3 threadSize(32, 6);
    dim3 blockSize(input.cols / (1 * threadSize.x), input.rows / (1 * threadSize.y));

    convert_ <<<blockSize, threadSize, 0, stream>>> (d_input, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(output.data, d_output, sizeof(float)*output.cols*output.rows, cudaMemcpyDeviceToHost));

    // resource releasing
    CUDA_CALL(cudaStreamDestroy(stream));
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
}