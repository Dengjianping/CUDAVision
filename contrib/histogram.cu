#include "..\cumath\cumath.cuh"
/*

*/

#define SIZE 256

__global__ void histogram(uchar *d_input, int in_height, int in_width, size_t in_pitch, uchar *d_output, int out_height, size_t out_pitch, int *counter, int *d_pixelSum) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) {
        uchar *in_pixel = (uchar*)((char*)d_input + row*in_pitch) + col;
        int index = (int)(*in_pixel);
        if (*d_pixelSum < INT_MAX) {
            printf("%d\n", *d_pixelSum);
            atomicAdd(d_pixelSum, 1);
        }
        counter[index]++;
        //atomicAdd(d_pixelSum, *in_pixel);
        //atomicAdd(&counter[index], 1);
    }
    
    //if (row < SIZE&&col < SIZE) {
    if (21 < col < 279) {
        //uchar *out_pixel = (uchar*)((char*)d_input + row*out_pitch) + col;
        int hist = counter[col];
        for (size_t i = 0; i < hist; i++) {
            uchar *out_pixel = (uchar*)((char*)d_output + (out_height - 1 - i)*out_pitch) + col;
            *out_pixel = 0;
        }
    }
}

extern "C"
void cudaHistogram(cv::Mat & input, cv::Mat & output) {
    output = cv::Mat(cv::Size(300, 300), input.type(), cv::Scalar(255));
    
    size_t in_pitch, out_pitch;
    uchar *d_input, *d_output;
    cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar)*output.cols, output.rows);

    // store counters for each pixel
    int *counter, *d_pixelSum;
    cudaMalloc(&counter, sizeof(int) * 256);
    cudaMalloc(&d_pixelSum, sizeof(int));

    cudaMemset(counter, 0, sizeof(int) * 256);
    cudaMemset(d_pixelSum, 0, sizeof(int));

    const int N = 2;
    cudaStream_t streams[N];
    for (size_t i = 0; i < N; i++)cudaStreamCreate(&streams[i]);

    cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, streams[1]);

    for (size_t i = 0; i < N; i++)cudaStreamSynchronize(streams[i]);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);

    histogram <<<blockSize, threadSize>>> (d_input, input.rows, input.cols, in_pitch, d_output, output.rows, out_pitch, counter, d_pixelSum);
    CUDA_CALL(cudaDeviceSynchronize());
    /*cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cout << cudaGetErrorString(error) << std::endl;
    }*/
    cudaMemcpy2D(output.data, output.cols * sizeof(uchar), d_output, out_pitch, output.cols * sizeof(uchar), output.rows, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++)cudaStreamDestroy(streams[i]);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_pixelSum);
}