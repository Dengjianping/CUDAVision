/*
follow this wiki page to know how to implement harris corner detecting: 
first derivative
    Ix convolutional kernel: [1,-1]
    Iy convolutioanl kernel: [1;
                             -1]
second derivative
Ixx convolutinal kernel: [1,-2,1]
Iyy convolutinal kernel: [1;
                         -2;
                          1]
Ixy=Iyx convolutional kenrel: [1,-1;
                              -1,1]
*/

#include "cumath.cuh"

#define ROW 2
#define COL 1

__constant__ float LX[COL][ROW] = { { 1,-1 } };
__constant__ float LY[ROW][COL] = { {1},{-1} };

struct CornerPoint{
    int x;
    int y;
    CornerPoint() {
        x = y = 0;
    }
};

__device__ float trace(float *xx, float *yy) {
    return *xx + *yy;
}

__device__ float det(float *xx, float *xy, float *yy) {
    return (*xx)*(*yy) - powf(*xy, 2);
}

__device__ float repressionValue(int row, int col, int window, float *xx, float *xy, float *yy, size_t pitch, float k) {
    float *Sxx = (float*)((char*)xx + row*pitch) + col;
    float *Sxy = (float*)((char*)xy + row*pitch) + col;
    float *Syy = (float*)((char*)yy + row*pitch) + col;
    return  det(Sxx, Sxy, Syy) - k*powf(trace(Sxx, Syy), 2);
}

__device__ float localMaxRepression(int row, int col, int window, float *xx, float *xy, float *yy, size_t pitch, float k) {
    float max = 0;
    for (int i = -window / 2; i <= window / 2; i++) {
        for (int j = -window / 2; j <= window / 2; j++) {
            float *Sxx = (float*)((char*)xx + (row + i)*pitch) + (col + j);
            float *Sxy = (float*)((char*)xy + (row + i)*pitch) + (col + j);
            float *Syy = (float*)((char*)yy + (row + i)*pitch) + (col + j);
            float r = det(Sxx, Sxy, Syy) - k*powf(trace(Sxx, Syy), 2);
            if (max < r)max = r;
        }
    }
    return max;
}

__global__ void harriscornel(float *d_input, size_t pitch, int heigth, int width, int radius, float theta, 
                             float k, float repression, int window, int *d_maxR, CornerPoint *d_points,
                             float *lx, float *ly, float *lxy, float *sxx, float *syy, float *sxy, float *d_output) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ float kernel[];
    if (row < heigth&&col < width) {
        // load gaussian kernel to shared memory
        if (row < 2 * radius + 1 && col < 2 * radius + 1) {
            kernel[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
            __syncthreads();
        }

        // get first differential matrix from x direction
        for (size_t i = 0; i < COL; i++)
            for (size_t j = 0; j < ROW; j++) {
                float *input = (float*)((char*)d_input + row *pitch) + col;
                float *x = (float*)((char*)lx + (row + i - COL / 2) *pitch) + (col + j - ROW / 2);
                *x += (*input)*LX[i][j];
                //float *y = (float*)((char*)ly + row *outputPitch) + col;
                //float *input = (float*)((char*)d_input + (row + i - COL / 2)*inputPitch) + (col + j - ROW / 2);
            }


        // get first differential matrix from y direction
        for (size_t i = 0; i < ROW; i++)
            for (size_t j = 0; j < COL; j++) {
                float *input = (float*)((char*)d_input + row *pitch) + col;
                float *y = (float*)((char*)ly + (row + i - ROW / 2) *pitch) + (col + j - COL / 2);
                *y += (*input)*LY[i][j];
                //float *y = (float*)((char*)ly + row *outputPitch) + col;
                //float *input = (float*)((char*)d_input + (row + i - COL / 2)*inputPitch) + (col + j - ROW / 2);
            }
       
        float *Ix = (float*)((char*)lx + row*pitch) + col;
        float *Iy = (float*)((char*)ly + row*pitch) + col;
        float *Ixy = (float*)((char*)lxy + row*pitch) + col;
        *Ixy = *Ix*(*Iy); // get Ixy
        *Ix = powf(*Ix, 2); // get Ix^2
        *Iy = powf(*Iy, 2); // get Iy^2  

        // convolve diffierential matrix with gaussian matrix
        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
                // blur Ix^2
                float *Sxx = (float*)((char*)sxx + (row + i - radius)*pitch) + (col + j - radius);
                *Sxx += (*Ix)*kernel[i*(2 * radius + 1) + j];

                // blur Iy^2
                float *Sxy = (float*)((char*)sxy + (row + i - radius)*pitch) + (col + j - radius);
                *Sxy += (*Ixy)*kernel[i*(2 * radius + 1) + j];

                // blur Ixy
                float *Syy = (float*)((char*)syy + (row + i - radius)*pitch) + (col + j - radius);
                *Syy += (*Iy)*kernel[i*(2 * radius + 1) + j];
            }

        // find max repression value in whole image
        float *Sxx = (float*)((char*)sxx + row*pitch) + col;
        float *Sxy = (float*)((char*)sxy + row*pitch) + col;
        float *Syy = (float*)((char*)syy + row*pitch) + col;
        float r = det(Sxx, Sxy, Syy) - k*powf(trace(Sxx, Syy), 2);
        
        atomicMax(d_maxR, r);
        printf("\d", *d_maxR);

        // core part
        if ((row > window - 1 && row < heigth - window / 2) && (col > window - 1 && col < width - window / 2)) {
            float r = repressionValue(row, col, window, sxx, sxy, syy, pitch, k);
            if (r > repression*(*d_maxR) && r > localMaxRepression(row, col, window, sxx, sxy, syy, pitch, k)) {
                d_points[row*width+col].x = row; d_points[row*width + col].y = col;
            }
            else {
                d_points[row*width + col].x = 0; d_points[row*width + col].y = 0;
            }
        }
    }
}

extern "C"
void cudaHarrisCorner(cv::Mat & input, cv::Mat & output, int radius, float theta, float k, float repression, int window) {
    input.convertTo(input, CV_32F);
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0, 0, 0));

    const int N = 8;
    float *d_input, *lx, *ly, *lxy, *sxx, *syy, *sxy, *d_output;
    size_t size = input.rows*input.cols*sizeof(CornerPoint);
    CornerPoint *h_points = new CornerPoint[input.rows*input.cols], *d_points; // record corner points

    cudaMalloc(&d_points, sizeof(CornerPoint)*input.rows*input.cols);

    cudaMallocPitch(&d_input, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&lx, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&ly, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&lxy, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&sxx, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&syy, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&sxy, &size, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &size, sizeof(float)*input.cols, input.rows);
    

    int h_maxR, *d_maxR;
    cudaMalloc(&d_maxR, sizeof(int));
    cudaMemset(d_maxR, 0, sizeof(int));

    cudaStream_t *streams = new cudaStream_t[N];
    for (size_t i = 0; i < N; i++) cudaStreamCreate(&streams[i]);

    size_t pitch = input.step;
    cudaMemcpy2DAsync(d_input, pitch, input.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy2DAsync(lx, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpy2DAsync(ly, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[2]);
    cudaMemcpy2DAsync(lxy, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[3]);
    cudaMemcpy2DAsync(sxx, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[4]);
    cudaMemcpy2DAsync(syy, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[5]);
    cudaMemcpy2DAsync(sxy, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[6]);
    cudaMemcpy2DAsync(d_output, pitch, output.data, pitch, pitch, input.rows, cudaMemcpyHostToDevice, streams[7]);

    for (size_t i = 0; i < N; i++) cudaStreamSynchronize(streams[i]);

    dim3 blockSize(input.cols / MAX_THREADS + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS, MAX_THREADS);
    
    harriscornel<<<blockSize, threadSize>>> (d_input, pitch, input.rows, input.cols, radius, theta, k, repression, window, d_maxR, d_points, lx, ly, lxy, sxx, syy, sxy, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    cudaMemcpy(&h_maxR, d_maxR, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_points, d_points, size, cudaMemcpyDeviceToHost);
    std::cout << h_maxR << std::endl;

    //cudaMemcpy2D(output.data, pitch, d_output, pitch, pitch, input.rows, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < input.rows; i++) 
        for (size_t j = 0; j < input.cols; j++){
            int t = i*input.cols + j;
            //std::cout << t  << ", " << i << ", " << j << std::endl;
            
            if (h_points[t].x != 0 && h_points[t].y != 0) {
                std::cout << h_points[t].x << ", " << h_points[t].y << std::endl;
                input.at<uchar>(h_points[t].x, h_points[t].y) = 255;
        }
    }
        
    cudaFree(d_input); cudaFree(lx); cudaFree(ly); cudaFree(sxx); cudaFree(syy); cudaFree(sxy); cudaFree(d_input); cudaFree(lxy); cudaFree(d_maxR);
    cudaFree(d_points);
    for (size_t i = 0; i < N; i++) cudaStreamDestroy(streams[i]);
}