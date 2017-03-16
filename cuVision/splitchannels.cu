#include "cumath.cuh"

__global__ void split(uchar3 *d_input, int height, int width, size_t pitch, uchar *r_ch, uchar *g_ch, uchar *b_ch) {

}

extern "C"
void cudaSplit(cv::Mat & input, std::vector<cv::Mat> & channels) {
    if (input.channels == 1) {
        channels.push_back(input);
        return;
    }
    channels = std::vector<cv::Mat>(input.channels);

    uchar3 *d_input; uchar *r_ch, *g_ch, *b_ch;
    size_t pitch;
    cudaMallocPitch(&d_input, &pitch, sizeof(uchar3)*input.cols, input.rows);
    cudaMallocPitch(&r_ch, &pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&g_ch, &pitch, sizeof(uchar)*input.cols, input.rows);
    cudaMallocPitch(&b_ch, &pitch, sizeof(uchar)*input.cols, input.rows);

    const int N = 4;
    cudaStream_t streams[N];
    for (size_t i = 0; i < N; i++)cudaStreamCreate(&streams[i]);
    cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(uchar3)*input.cols, sizeof(uchar3)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy2DAsync(r_ch, pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpy2DAsync(g_ch, pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[2]);
    cudaMemcpy2DAsync(b_ch, pitch, input.data, sizeof(uchar)*input.cols, sizeof(uchar)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[3]);

    for (size_t i = 0; i < N; i++)cudaStreamSynchronize(streams[i]);

    cudaMemset2D(r_ch, pitch, 0, sizeof(uchar)*input.cols, input.rows);
    cudaMemset2D(g_ch, pitch, 0, sizeof(uchar)*input.cols, input.rows);
    cudaMemset2D(b_ch, pitch, 0, sizeof(uchar)*input.cols, input.rows);
}