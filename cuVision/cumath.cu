#include "cumath.cuh"

// convolve 1-dim array
template <typename type>
inline __device__ void convolve1D(int threadRow, int threadCol, type *input, size_t inputPitch, type *output, size_t outputPitch) {

}

// convolve matrix
template <typename type>
inline __device__ void convolve2D(int threadRow, int threadCol, type *input, int inputRow, int inputCol, size_t inputPitch, int kernelSize, type *kernel, type *output, size_t outputPitch, shape mode = full) {
    if (threadRow < inputRow&&threadCol < inputCol) {
        for (size_t i = 0; i < kernelSize; i++)
            for (size_t j = 0; j < kernelSize; j++) {
                type *inputValue = (type*)((char*)input + threadRow*inputPitch) + threadCol;
                type *outputValue = (type*)((char*)input + (threadRow + i)*inputPitch) + (threadCol + j);
                *outputValue += kernel[i*kernelSize + j] * (*inputValue);
            }
    }
}

// deconvolve 1-dim array
inline __device__ void deconvolve1D() {

}

// deconvolve matrix
inline __device__ void deconvolve2D() {

}

// matrix multiplication
inline __device__ void multiply2D() {

}

// array addition
inline __device__ void add1D() {

}

// matrix addition
inline __device__ void add2D() {

}

inline __device__ void sub1D() {

}

inline __device__ void sub2D() {

}

inline __device__ float twoDimGaussian(int x, int y, float theta) {
    float coeffient = 1 / (2 * PI*powf(theta, 2));
    float powerIndex = -(powf(x, 2) + powf(y, 2)) / (2 * powf(theta, 2));
    return coeffient*expf(powerIndex);
}