#include "..\cumath\cumath.cuh"


template<int N>
__global__ void reduction(uchar *d_input, int height, int width, int *aux)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 2 * blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int smem[17 * 33]; // add one more column to eliminate bank conflict

    int r = threadIdx.x / 32, c = threadIdx.x % 32;
    smem[r * 33 + c] = d_input[row*width + col];
    smem[(r + 8) * 33 + c] = d_input[row*width + col + 256];
    __syncthreads();

    int offset = 1;

    #pragma unroll
    for (int k = N >> 1; k > 0; k >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < k)
        {
            int ai = offset*(2 * threadIdx.x + 1) - 1;
            int bi = offset*(2 * threadIdx.x + 2) - 1;
            smem[bi / 32 * 33 + bi % 32] += smem[ai / 32 * 33 + ai % 32];
        }
        offset *= 2;
    }
    if (threadIdx.x == 0)aux[blockIdx.y*gridDim.x + blockIdx.x] = smem[526]; // store each block sum
}



__global__ void sumOfPixels(int N, int *aux, int *d_sum)
{
    /*__shared__ int shared[2][N];
    int offset = 1;
    shared[2 * threadIdx.x] = aux[2 * threadIdx.x];
    shared[2 * threadIdx.x + 1] = aux[2 * threadIdx.x + 1];

    for (int i = length >> 1; i > 0; i >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < i)
        {
            int ai = offset*(2 * threadIdx.x + 1) - 1;
            int bi = offset*(2 * threadIdx.x + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset *= 2;
    }
    if (threadIdx.x == 0)
    {
        *sum = shared[length - 1];
        printf("%d\n", *sum);
    }*/
    for (int i = 0; i < N; i += 256)
    {
        atomicAdd(d_sum, aux[i + threadIdx.x]);
    }
}

extern "C"
void cudaSum(const cv::Mat & input, int & sum)
{
    dim3 threadSize(256, 1);
    dim3 blockSize(input.cols / (2 * threadSize.x), input.rows / (1 * threadSize.y));

    cudaStream_t stream; cudaStreamCreate(&stream);
    uchar *d_input; 
    cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream);

    int *aux; // store each block of sum
    cudaMalloc(&aux, sizeof(int)*blockSize.x*blockSize.y);

    int *d_sum; // sum of all pixels
    cudaMalloc(&d_sum, sizeof(int)); cudaMemset(d_sum, 0, sizeof(int));

    reduction<512> <<<blockSize, threadSize, 0, stream >>> (d_input, input.rows, input.cols, aux);
    sumOfPixels <<<1, threadSize, 0, stream>>> (blockSize.x*blockSize.y, aux, d_sum);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(aux); cudaFree(d_sum);
}


extern "C"
void cudaMeanValue(const cv::Mat & input, float & mean)
{
    int sum = 0;
    cudaSum(input, sum);
    mean = (float)sum / (input.rows*input.cols);
}