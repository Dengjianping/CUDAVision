#include "cumath.cuh"

__global__ void integral(uchar *d_input, int height, int width, size_t in_pitch, float *d_output, size_t out_pitch, int *max, int *min) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < height  &&  col < width) {
        uchar *in_pixel = (uchar*)((char*)d_input + row*in_pitch) + col;
        float *out_pixel = (float*)((char*)d_output + row*out_pitch) + col;
        float *out_pixel1 = (float*)((char*)d_output + (row-1)*out_pitch) + col;
        float *out_pixel2 = (float*)((char*)d_output + row*out_pitch) + (col-1);
        float *out_pixel3 = (float*)((char*)d_output + (row - 1)*out_pitch) + (col - 1);

        *out_pixel = (float)*in_pixel + *out_pixel1 + *out_pixel2 - (*out_pixel3);
        atomicMin(min, *out_pixel);
        atomicMax(max, *out_pixel);
        //atomicAdd(sum, *out_pixel);
    }
    /*if (row < height  &&  col < width) {
        float *out_pixel = (float*)((char*)d_output + row*out_pitch) + col;
        *out_pixel = (*out_pixel - (float)(*min)) / ((float)(*max) - (float)(*min));
    }*/
}

extern "C"
void cudaIntegral(cv::Mat & input, cv::Mat & output) {
    //input.convertTo(input, CV_32F);
    output = cv::Mat(cv::Size(input.cols + 1, input.rows + 1), CV_32F, cv::Scalar(0));

    uchar *d_input; float *d_output;
    size_t in_pitch, out_pitch;
    cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &out_pitch, sizeof(float)*output.cols, output.rows);

    int *min, *max;
    cudaMalloc(&min, sizeof(int));
    cudaMalloc(&max, sizeof(int));
    cudaMemset(min, 0, sizeof(int));
    cudaMemset(max, 0, sizeof(int));

    cudaStream_t in_stream, out_stream;
    cudaStreamCreate(&in_stream); cudaStreamCreate(&out_stream);

    cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream);
    cudaMemcpy2DAsync(d_output, in_pitch, output.data, sizeof(float)*output.cols, sizeof(float)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream);

    cudaStreamSynchronize(in_stream); cudaStreamSynchronize(out_stream);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);

    integral <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, in_pitch, d_output,  out_pitch, max, min);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, out_pitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    int h_max, h_min;
    cudaMemcpy(&h_max, max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min, min, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << h_max << std::endl;
    std::cout << h_min << std::endl;

    cudaStreamDestroy(in_stream); cudaStreamDestroy(out_stream);
    cudaFree(d_input); cudaFree(d_output); cudaFree(max); cudaFree(min);

    //input.convertTo(input, CV_8U);
    output.convertTo(output, CV_32S);
}