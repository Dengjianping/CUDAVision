#include "..\cumath\cumath.cuh"

#define RED_WEIGHT 0.2989
#define GREEN_WEIGHT 0.5870
#define BLUE_WEIGHT 0.1140

__global__ void grayscale(uchar3 *d_input, int height, int width, size_t in_pitch, uchar *d_output, size_t out_pitch) {
    uint row = blockDim.y*blockIdx.y + threadIdx.y;
    uint col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ uchar3 sh[];

    if (row < height&&col < width) {
        uchar3 *in_pixel = (uchar3*)((char*)d_input + row*in_pitch) + col;
        sh[threadIdx.y*blockDim.x + threadIdx.x] = *in_pixel;
        __syncthreads();

        uchar *out_pixel = (uchar*)((char*)d_output + row*out_pitch) + col;

        //*out_pixel = RED_WEIGHT * ((float)in_pixel->x) + GREEN_WEIGHT * ((float)in_pixel->x) + BLUE_WEIGHT*((float)in_pixel->x);
        *out_pixel = RED_WEIGHT * ((float)sh[threadIdx.y*blockDim.x + threadIdx.x].x) + GREEN_WEIGHT * ((float)sh[threadIdx.y*blockDim.x + threadIdx.x].y) + BLUE_WEIGHT*((float)sh[threadIdx.y*blockDim.x + threadIdx.x].z);
    }
}

__global__ void hsvscale() {

}

extern "C"
void cudaGray(cv::Mat & input, cv::Mat & output) {
    if (input.channels() == 1) {
        input.copyTo(output);
        return;
    }
    output = cv::Mat(input.size(), CV_8U, cv::Scalar(0));

    uchar3 *d_input; uchar *d_output;
    size_t in_pitch, out_pitch;

    cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar3)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &out_pitch, sizeof(uchar3)*output.cols, output.rows);

    cudaStream_t in_stream, out_stream;
    cudaStreamCreate(&in_stream); cudaStreamCreate(&out_stream);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);
    size_t shared = threadSize.x*threadSize.y * sizeof(uchar3);

    cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar3)*input.cols, sizeof(uchar3)*input.cols, input.rows, cudaMemcpyHostToDevice, in_stream);
    cudaMemcpy2DAsync(d_output, out_pitch, output.data, sizeof(uchar)*output.cols, sizeof(uchar)*output.cols, output.rows, cudaMemcpyHostToDevice, out_stream);

    cudaStreamSynchronize(in_stream); cudaStreamSynchronize(out_stream);

    grayscale<<<blockSize, threadSize, shared>>> (d_input, input.rows, input.cols, in_pitch, d_output, out_pitch);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaMemcpy2D(output.data, sizeof(uchar)*output.cols, d_output, out_pitch, sizeof(uchar)*output.cols, output.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(in_stream); cudaStreamDestroy(out_stream);
    cudaFree(d_input); cudaFree(d_output);
}