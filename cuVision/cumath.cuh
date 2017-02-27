#ifndef CUMATH_H
#define CUMATH_H

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) { std::cout << std::endl << "CUDA Error: " << cudaGetErrorString(a) << ", error number: " << a << std::endl; cudaDeviceReset(); assert(0);}}
#define MAX_THREADS 32

#include "headers.h"

__constant__ float PI = 3.1415;

/*
    full: target_size + kernel_size - 1
    same: target_size
    valid: target_size - kernel_size + 1
*/
enum shape { full, same, valid };

// convolve 1-dim array
template <typename type>
inline __device__ void convolve1D(int threadRow, int threadCol, type *input, size_t inputPitch, type *output, size_t outputPitch);

// convolve matrix
template <typename type>
inline __device__ void convolve2D(type *input, int height, int width, size_t pitch, type *kernel, int kernelSize, type *output, shape mode) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if (row < height&&col < width) {
        for (size_t i = 0; i < kernelSize; i++)
            for (size_t j = 0; j < kernelSize; j++) {
                type *inputElement = (type*)((char*)input + row*pitch) + col;
                type *outputElement = (type*)((char*)output + row*pitch) + col;
                *outputElement += (*inputElement)*kernel[i*kernel + j];
            }
    }
}

// deconvolve 1-dim array
inline __device__ void deconvolve1D();

// deconvolve matrix
inline __device__ void deconvolve2D();

// matrix multiplication
inline __device__ void multiply2D();

// array addition
inline __device__ void add1D();

// matrix addition
inline __device__ void add2D();

inline __device__ void sub1D();

inline __device__ void sub2D();

inline __device__ float twoDimGaussian(int x, int y, float theta) {
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}

#endif // !CUMATH_H