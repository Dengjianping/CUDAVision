#include "cumath.cuh"

template <typename type>
__global__ void flip_90(type *d_input, uint in_height, uint in_width, size_t in_pitch, type *d_output, size_t out_pitch) {
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) {
        type *in_pixel = (type*)((char*)d_input + row*in_pitch) + col;
        type *out_pixel = (type*)((char*)d_output + col*out_pitch) + row;

        *out_pixel = *in_pixel;
    }
}

template <typename type>
__global__ void flip_180(type *d_input, uint in_height, uint in_width, size_t in_pitch, type *d_output, size_t out_pitch) {
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) {
        type *in_pixel = (type*)((char*)d_input + row*in_pitch) + col;
        type *out_pixel = (type*)((char*)d_output + (in_height - row)*out_pitch) + (in_width - col);

        *out_pixel = *in_pixel;
    }
}

template <typename type>
__global__ void flip_270(type *d_input, uint in_height, uint in_width, size_t in_pitch, type *d_output, size_t out_pitch) {
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) {
        type *in_pixel = (type*)((char*)d_input + row*in_pitch) + col;
        type *out_pixel = (type*)((char*)d_output + (in_width-col)*out_pitch) + row;

        *out_pixel = *in_pixel;
    }
}

extern "C"
void cudaFlip(cv::Mat & input, cv::Mat & output, Orientation ori) {
    int channel = input.channels();
    cv::Scalar scalar = channel == 3 ? cv::Scalar(0, 0, 0) : cv::Scalar(0);
    int pixel_size = channel == 3 ? sizeof(uchar3) : sizeof(uchar);

    uchar *d_input, *d_output;
    size_t in_pitch, out_pitch;

    cudaStream_t in_stream, out_stream;
    cudaStreamCreate(&in_stream); cudaStreamCreate(&out_stream);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);
    
    switch (ori)
    {
    case angel_90:
        output = cv::Mat(cv::Size(input.rows, input.cols), input.type(), scalar);

        cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows);
        cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

        cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream);
        cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream);

        cudaStreamSynchronize(in_stream); cudaStreamSynchronize(out_stream);

        flip_90<uchar> <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);

        CUDA_CALL(cudaDeviceSynchronize());
        cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost);
        cudaFree(d_input); cudaFree(d_output);
        break;
    case angel_180:
        output = cv::Mat(input.size(), input.type(), scalar);

        cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows);
        cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

        cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream);
        cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream);

        cudaStreamSynchronize(in_stream); cudaStreamSynchronize(out_stream);

        flip_180<uchar> <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);
        CUDA_CALL(cudaDeviceSynchronize());

        cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost);
        cudaFree(d_input); cudaFree(d_output);

        break;
    case angel_270:
        output = cv::Mat(cv::Size(input.rows, input.cols), input.type(), scalar);

        cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows);
        cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

        cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream);
        cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream);

        cudaStreamSynchronize(in_stream); cudaStreamSynchronize(out_stream);

        flip_270<uchar> <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);
        CUDA_CALL(cudaDeviceSynchronize());

        cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost);
        cudaFree(d_input); cudaFree(d_output);

        break;
    default:
        break;
    }

    cudaStreamDestroy(in_stream); cudaStreamDestroy(out_stream);
}