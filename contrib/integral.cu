#include "..\cumath\cumath.cuh"

__global__ void integral(int *d_input, int height, int width, size_t pitch, int *d_output) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < height  &&  col < width) {
        int *in_pixel = (int*)((char*)d_input + row*pitch) + col;
        int *out_pixel = (int*)((char*)d_output + row*pitch) + col;
        //int *out_pixel1 = (int*)((char*)d_output + (row-1)*pitch) + col;
        //int *out_pixel2 = (int*)((char*)d_output + row*pitch) + (col-1);
        //int *out_pixel3 = (int*)((char*)d_output + (row - 1)*pitch) + (col - 1);

        for (uint i = 0; i <= row; i++)
            for (uint j = 0; j <= col; j++)
            {
                *out_pixel += *((int*)((char*)d_input + i*pitch) + j);
            }
    }
}

extern "C"
void cudaIntegral(cv::Mat & input, cv::Mat & output) {
    input.convertTo(input, CV_32S);
    //output = cv::Mat(cv::Size(input.cols + 1, input.rows + 1), CV_32S, cv::Scalar(0));
    output = cv::Mat(input.size(), CV_32S, cv::Scalar(0));

    int *d_input, *d_output;
    size_t pitch;
    cudaMallocPitch(&d_input, &pitch, sizeof(int)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &pitch, sizeof(int)*output.cols, output.rows);

    cudaStream_t in_stream, out_stream;
    cudaStreamCreate(&in_stream); cudaStreamCreate(&out_stream);

    cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(int)*input.cols, sizeof(int)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream);
    cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(int)*output.cols, sizeof(int)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream);

    cudaStreamSynchronize(in_stream); cudaStreamSynchronize(out_stream);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);

    integral <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, pitch, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy2D(output.data, sizeof(int)*output.cols, d_output, pitch, sizeof(int)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(in_stream); cudaStreamDestroy(out_stream);
    cudaFree(d_input); cudaFree(d_output);

    input.convertTo(input, CV_8U);
    output.convertTo(output, CV_8U);
}