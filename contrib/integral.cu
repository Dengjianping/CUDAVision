#include "..\cumath\cumath.cuh"
#define BLOCK_SIZE 32

template<int N, typename T, typename P>
//__global__ void __launch_bounds__(MAX_BLOCK_SIZE, MIN_BLOCKS_PER_SM) scan(uchar *d_input, int height, int width, uchar *d_output)
__global__ void scan(T *d_input, int height, int width, P *d_output, int *aux)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = 2 * blockDim.x*blockIdx.x + threadIdx.x;
    __shared__ int smem[17 * 33]; // add one more column to eliminate bank conflict

    int r = threadIdx.x / 32, c = threadIdx.x % 32;
    /*for (uint i = row; i < height; i += blockDim.y*gridDim.y)
        for (uint j = col; j < width; j += blockDim.x*gridDim.x)*/
    // grid stride uses too much registers in this kernel
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

    if (threadIdx.x == 0)
    {
        int t = smem[526];
        aux[blockIdx.y*gridDim.x + blockIdx.x] = t;
        smem[528] = t; // 15*33+32
        smem[526] = 0;
    }

    #pragma unroll
    for (int k = 1; k < N; k *= 2)
    {
        offset /= 2; __syncthreads();
        if (threadIdx.x < k)
        {
            int ai = offset*(2 * threadIdx.x + 1) - 1;
            int bi = offset*(2 * threadIdx.x + 2) - 1;
            int temp = smem[ai / 32 * 33 + ai % 32];
            smem[ai / 32 * 33 + ai % 32] = smem[bi / 32 * 33 + bi % 32];
            smem[bi / 32 * 33 + bi % 32] += temp;
        }
    }
    __syncthreads();
    d_output[row*width + col] = c == 31 ? smem[(r + 1) * 33] : smem[r * 33 + c + 1];
    d_output[row*width + col + 256] = c == 31 ? smem[(r + 9) * 33] : smem[(r + 8) * 33 + c + 1];
    if (blockIdx.x > 0)
    {
        d_output[row*width + col] = aux[blockIdx.y*gridDim.x + blockIdx.x - 1] + d_output[row*width + col];
        d_output[row*width + col + 256] = aux[blockIdx.y*gridDim.x + blockIdx.x - 1]+ d_output[row*width + col + 256];
    }
}


template<typename T, typename P>
__global__ void tranpose_right(T *d_input, int height, int width, P *d_output)
{
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ int smem[BLOCK_SIZE][BLOCK_SIZE + 1];
    if (row < height&&col < width)
    {
        //#pragma unroll
        for (size_t i = 0; i < BLOCK_SIZE; i += 8)
        {
            smem[threadIdx.y + i][threadIdx.x] = d_input[(row + i)*width + col];
        }
        __syncthreads();

        row = blockIdx.x*BLOCK_SIZE + threadIdx.y;
        col = blockIdx.y*BLOCK_SIZE + threadIdx.x;

        //#pragma unroll
        for (size_t i = 0; i < BLOCK_SIZE; i += 8)
        {
            d_output[(height - row - i)*width + width - col] = smem[threadIdx.x][threadIdx.y + i];
            //d_output[(width - col - i)*width + height - row] = smem[threadIdx.x][threadIdx.y + i];
        }
    }
}


template<typename T, typename P>
__global__ void tranpose_left(T *d_input, int height, int width, P *d_output)
{
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __shared__ int smem[BLOCK_SIZE][BLOCK_SIZE + 1];
    if (row < height&&col < width)
    {
        //#pragma unroll
        for (size_t i = 0; i < BLOCK_SIZE; i += 8)
        {
            smem[threadIdx.y + i][threadIdx.x] = d_input[(row + i)*width + col];
        }
        __syncthreads();

        row = blockIdx.x*BLOCK_SIZE + threadIdx.y;
        col = blockIdx.y*BLOCK_SIZE + threadIdx.x;

        //#pragma unroll
        for (size_t i = 0; i < BLOCK_SIZE; i += 8)
        {
            d_output[(row + i)*width + col] = smem[threadIdx.x][threadIdx.y + i];
        }
    }
}


extern "C"
void cudaIntegral(const cv::Mat & input, cv::Mat & output) 
{
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0));
    dim3 threadSize(256, 1);
    dim3 blockSize(input.cols / (2 * threadSize.x), input.rows / (1 * threadSize.y));

    cudaStream_t stream; cudaStreamCreate(&stream);
    uchar *d_input, *d_output; int *d_o,*d_i;
    cudaMalloc(&d_input, sizeof(uchar)*input.cols*input.rows);
    cudaMalloc(&d_o, sizeof(int)*input.cols*input.rows);
    cudaMalloc(&d_i, sizeof(int)*input.cols*input.rows);
    cudaMemcpyAsync(d_input, input.data, sizeof(uchar)*input.cols*input.rows, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_output, sizeof(uchar)*output.cols*output.rows);

    int *aux; // store each block of sum
    cudaMalloc(&aux, sizeof(int)*blockSize.x*blockSize.y);

    dim3 tran(32, 8);
    dim3 tranGrid(input.cols / (1 * tran.x), input.rows / (4 * tran.y));

    scan<512, uchar, int> <<<blockSize, threadSize, 0, stream >>> (d_input, input.rows, input.cols, d_i, aux);
    tranpose_left<int, int> <<<tranGrid, tran, 0, stream >>> (d_i, input.rows, input.cols, d_o);
    scan<512, int, int> <<<blockSize, threadSize, 0, stream>>> (d_o, input.rows, input.cols, d_o, aux);
    tranpose_right<int, uchar> <<<tranGrid, tran, 0, stream >>> (d_o, input.rows, input.cols, d_output);
    //tranpose<uchar, uchar> <<<tranGrid, tran, 0, stream>>> (d_output, input.rows, input.cols, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(output.data, d_output, sizeof(uchar)*output.cols* output.rows, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_output); cudaFree(aux); cudaFree(d_o); cudaFree(d_i);
}