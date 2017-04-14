/*
    just support 1 channel image, it can retate a image with angle 90 degree or 180, 270.
    use texture memory to improve performance, instead of global memory.
*/

#include "..\cumath\cumath.cuh"

texture<uchar, 2, cudaReadModeElementType> text2D;

__global__ void flip_90(uchar *d_input, uint in_height, uint in_width, size_t in_pitch, uchar *d_output, size_t out_pitch) 
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) 
    {
        uchar *out_pixel = (uchar*)((char*)d_output + col*out_pitch) + row;
        *out_pixel = tex2D(text2D, col, row);
    }
}

__global__ void flip_180(uchar *d_input, uint in_height, uint in_width, size_t in_pitch, uchar *d_output, size_t out_pitch) 
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) 
    {
        uchar *out_pixel = (uchar*)((char*)d_output + (in_height - row)*out_pitch) + (in_width - col);
        *out_pixel = tex2D(text2D, col, row);
    }
}

__global__ void flip_270(uchar *d_input, uint in_height, uint in_width, size_t in_pitch, uchar *d_output, size_t out_pitch) 
{
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) 
    {
        uchar *out_pixel = (uchar*)((char*)d_output + (in_width-col)*out_pitch) + row;
        *out_pixel = tex2D(text2D, col, row);
    }
}

extern "C"
void cudaFlip(cv::Mat & input, cv::Mat & output, Orientation ori) 
{
    int channel = input.channels();
    cv::Scalar scalar = channel == 3 ? cv::Scalar(0, 0, 0) : cv::Scalar(0);
    int pixel_size = channel == 3 ? sizeof(uchar3) : sizeof(uchar);

    uchar *d_input, *d_output;
    size_t in_pitch, out_pitch;
    CUDA_CALL(cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows));

    cudaStream_t in_stream, out_stream;
    CUDA_CALL(cudaStreamCreate(&in_stream)); CUDA_CALL(cudaStreamCreate(&out_stream));

    // setup texture
    text2D.filterMode = cudaFilterModePoint;
    text2D.addressMode[0] = cudaAddressModeWrap;
    text2D.addressMode[1] = cudaAddressModeWrap;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
    // bind texture
    CUDA_CALL(cudaBindTexture2D(NULL, &text2D, d_input, &desc, input.cols, input.rows, in_pitch));

    dim3 threadSize(32, 32);
    dim3 blockSize(input.cols / threadSize.x + 1, input.rows / threadSize.y + 1);
    
    switch (ori)
    {
    case angel_90:
        output = cv::Mat(cv::Size(input.rows, input.cols), input.type(), scalar);

        cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

        CUDA_CALL(cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream));
        CUDA_CALL(cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream));

        flip_90<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

        break;
    case angel_180:
        output = cv::Mat(cv::Size(input.rows, input.cols), input.type(), scalar);

        cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

        CUDA_CALL(cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream));
        CUDA_CALL(cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream));

        flip_180<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

        break;
    case angel_270:
        output = cv::Mat(cv::Size(input.rows, input.cols), input.type(), scalar);

        cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

        CUDA_CALL(cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream));
        CUDA_CALL(cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream));

        flip_270<<<blockSize, threadSize>>>(d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);

        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost));

        break;
    default:
        break;
    }

    CUDA_CALL(cudaUnbindTexture(&text2D)); // remenber to unbind texture memory
    CUDA_CALL(cudaFree(d_input)); CUDA_CALL(cudaFree(d_output));
    CUDA_CALL(cudaStreamDestroy(in_stream)); CUDA_CALL(cudaStreamDestroy(out_stream));
}