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
__constant__ float P = 3.1415;


__global__ void harriscornel(float *d_input, size_t inputPitch, int rows, int cols, int radius, float theta, float *lx, float *ly, float *lxy, float *sxx, float *syy, float *sxy, float *d_output, size_t outputPitch) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ float kernel[];
    if (row < rows&&col < cols) {
        // load gaussian kernel to shared memory
        if (row < 2 * radius + 1 && col < 2 * radius + 1) {
            kernel[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
            __syncthreads();
        }

        // get first differential matrix from x direction
        for (size_t i = 0; i < COL; i++)
            for (size_t j = 0; j < ROW; j++) {
                float *input = (float*)((char*)d_input + row *inputPitch) + col;
                float *x = (float*)((char*)lx + (row + i - COL / 2) *outputPitch) + (col + j - ROW / 2);
                *x += (*input)*LX[i][j];
                //float *y = (float*)((char*)ly + row *outputPitch) + col;
                //float *input = (float*)((char*)d_input + (row + i - COL / 2)*inputPitch) + (col + j - ROW / 2);
            }


        // get first differential matrix from y direction
        for (size_t i = 0; i < ROW; i++)
            for (size_t j = 0; j < COL; j++) {
                float *input = (float*)((char*)d_input + row *inputPitch) + col;
                float *y = (float*)((char*)ly + (row + i - ROW / 2) *outputPitch) + (col + j - COL / 2);
                *y += (*input)*LY[i][j];
                //float *y = (float*)((char*)ly + row *outputPitch) + col;
                //float *input = (float*)((char*)d_input + (row + i - COL / 2)*inputPitch) + (col + j - ROW / 2);
            }
       
        float *Ix = (float*)((char*)lx + row*outputPitch) + col;
        float *Iy = (float*)((char*)ly + row*outputPitch) + col;
        float *Ixy = (float*)((char*)lxy + row*outputPitch) + col;
        *Ixy = *Ix*(*Iy); // get Ixy
        *Ix = powf(*Ix, 2); // get Ix^2
        *Iy = powf(*Iy, 2); // get Iy^2  

        // convolve diffierential matrix with gaussian matrix
        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
            }

        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
                // blur Ix^2
                float *Sxx = (float*)((char*)sxx + (row + i - radius)*outputPitch) + (col + j - radius);
                *Sxx += (*Ix)*kernel[i*(2 * radius + 1) + j];

                // blur Iy^2
                float *Sxy = (float*)((char*)sxy + (row + i - radius)*outputPitch) + (col + j - radius);
                *Sxy += (*Ixy)*kernel[i*(2 * radius + 1) + j];

                // blur Ixy
                float *Syy = (float*)((char*)syy + (row + i - radius)*outputPitch) + (col + j - radius);
                *Syy += (*Iy)*kernel[i*(2 * radius + 1) + j];
            }


    }
}

extern "C"
void cudaHarrisCorner(cv::Mat & input, cv::Mat & output) {
    input.convertTo(input, CV_32FC3);
    output = cv::Mat(input.size(), input.type(), cv::Scalar(0, 0, 0));

    float *d_input, *lx, *ly, *d_output;
}