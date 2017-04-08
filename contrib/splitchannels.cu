#include "..\cumath\cumath.cuh"

__global__ void split(uchar3 *d_input, int height, int width, size_t in_pitch, uchar *r_ch, uchar *g_ch, uchar *b_ch, size_t out_pitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ uchar3 shared[];

    if (row < height && col < width) {
        uchar3 *in_pixel = (uchar3*)((char*)d_input + row*in_pitch) + col;
        shared[threadIdx.y*blockDim.x + threadIdx.x] = *in_pixel;
        __syncthreads();

        uchar *r_pixel = (uchar*)((char*)r_ch + row*out_pitch) + col;
        uchar *g_pixel = (uchar*)((char*)g_ch + row*out_pitch) + col;
        uchar *b_pixel = (uchar*)((char*)b_ch + row*out_pitch) + col;

        *r_pixel = shared[threadIdx.y*blockDim.x + threadIdx.x].x;
        *g_pixel = shared[threadIdx.y*blockDim.x + threadIdx.x].y;
        *b_pixel = shared[threadIdx.y*blockDim.x + threadIdx.x].z;
    }
}

extern "C"
void cudaSplit(cv::Mat & input, std::vector<cv::Mat> & channels) {
    if (input.channels() == 1) {
        channels.push_back(input);
        return;
    }
    //channels = std::vector<cv::Mat>(input.channels);
    for (size_t i = 0; i < input.channels(); i++) {
        cv::Mat ch = cv::Mat(input.size(), CV_8U, cv::Scalar(0));
        channels.push_back(ch);
    }

    uchar3 *d_input; uchar *r_ch, *g_ch, *b_ch;
    size_t in_pitch, out_pitch;
    cudaMallocPitch(&d_input, &in_pitch, sizeof(uchar3)*input.cols, input.rows);
    cudaMallocPitch(&r_ch, &out_pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&g_ch, &out_pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&b_ch, &out_pitch, sizeof(uchar)*input.cols, input.rows);

    const int N = 4;
    cudaStream_t streams[N];
    for (size_t i = 0; i < N; i++)cudaStreamCreate(&streams[i]);
    cudaMemcpy2DAsync(d_input, in_pitch, input.data, sizeof(uchar3)*input.cols, sizeof(uchar3)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy2DAsync(r_ch, out_pitch, channels[0].data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpy2DAsync(g_ch, out_pitch, channels[1].data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[2]);
    cudaMemcpy2DAsync(b_ch, out_pitch, channels[2].data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[3]);

    for (size_t i = 0; i < N; i++)cudaStreamSynchronize(streams[i]);

    //cudaMemset2D(r_ch, pitch, 0, sizeof(uchar)*input.cols, input.rows);
    //cudaMemset2D(g_ch, pitch, 0, sizeof(uchar)*input.cols, input.rows);
    //cudaMemset2D(b_ch, pitch, 0, sizeof(uchar)*input.cols, input.rows);

    dim3 blockSize(input.cols / (MAX_THREADS/2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS/2, MAX_THREADS);
    size_t size = threadSize.x*threadSize.y * sizeof(uchar3);

    split<<<blockSize, threadSize, size>>>(d_input, input.rows, input.cols, in_pitch, r_ch, g_ch, b_ch, out_pitch);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy2DAsync(channels[0].data, sizeof(uchar)*input.cols, r_ch, out_pitch, sizeof(uchar)*input.cols, input.rows, cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpy2DAsync(channels[1].data, sizeof(uchar)*input.cols, g_ch, out_pitch, sizeof(uchar)*input.cols, input.rows, cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpy2DAsync(channels[2].data, sizeof(uchar)*input.cols, b_ch, out_pitch, sizeof(uchar)*input.cols, input.rows, cudaMemcpyDeviceToHost, streams[2]);
    for (size_t i = 0; i < N-1; i++)cudaStreamSynchronize(streams[i]);

    for (size_t i = 0; i < N; i++)cudaStreamDestroy(streams[i]);
    cudaFree(r_ch); cudaFree(g_ch); cudaFree(b_ch); cudaFree(d_input);
}