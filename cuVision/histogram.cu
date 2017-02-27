#include "cumath.cuh"
/*

*/

#define SIZE 256

__global__ void histogram(uchar *d_input, int in_height, int in_width, size_t in_pitch, uchar *d_output, size_t out_pitch, uchar *counter) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < in_height&&col < in_width) {
        uchar *in_pixel = (uchar*)((char*)d_input + row*in_pitch) + col;
        counter[*in_pixel]++;
    }
    if (row < SIZE&&col < SIZE) {
        //uchar *out_pixel = (uchar*)((char*)d_input + row*out_pitch) + col;
        int hist = counter[row];
        for (size_t i = 0; i < hist; i++) {
            uchar *out_pixel = (uchar*)((char*)d_input + row*out_pitch) + i;
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
    static uchar temp[256];
    uchar *counter;
    cudaMalloc(&counter, sizeof(uchar) * 256);
    cudaMemcpy(temp, counter, sizeof(uchar) * 256,cudaMemcpyHostToDevice);
    cudaMemset(counter, 0, sizeof(uchar) * 256);

    const int N = 2;
    cudaStream_t streams[N];
    for (size_t i = 0; i < N; i++)cudaStreamCreate(&streams[i]);

    cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, streams[1]);

    for (size_t i = 0; i < N; i++)cudaStreamSynchronize(streams[i]);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);
}