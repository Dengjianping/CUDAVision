#include "..\cumath\cumath.cuh"

#define SIZE 2

__constant__ char XAxis[2][2] = { {-1,1},{-1,1} };
__constant__ char YAxis[2][2] = { {-1,-1},{1,1} };

enum orientation { horizontal, vertical, possquint, negsquint };

//template <typename type>
//__device__ void binaryzation(int row, int col, type *input, type *output, size_t pitch, type thresholdL, type thresholdH) {
//    type *input_pixel = (type*)((char*)input + row*pitch) + col;
//    type *output_pixel = (type*)((char*)input + row*pitch) + col;
//    if (*input_pixel > thresholdL)*output_pixel = 255;
//    else {
//        *output_pixel = 0;
//    }
//}

__device__ float maxPixel(int row, int col, float *input, size_t pitch, int window, orientation angel) {
    if (angel == horizontal) {
        float *pixel1 = (float*)((char*)input + row*pitch) + (col - 1);
        float *pixel2 = (float*)((char*)input + row*pitch) + (col + 1);
        float *pixel3 = (float*)((char*)input + row*pitch) + col;
        float max1 = fmax(*pixel2, *pixel1), max2 = fmax(*pixel2, *pixel1);
        return max1 > max2 ? max1 : max2;
    }
    if (angel == vertical) {
        float *pixel1 = (float*)((char*)input + row*pitch) + (col - 1);
        float *pixel2 = (float*)((char*)input + (row + 1)*pitch) + (col - 1);
        float *pixel3 = (float*)((char*)input + (row - 1)*pitch) + (col + 1);
        float max1 = fmax(*pixel2, *pixel1), max2 = fmax(*pixel2, *pixel1);
        return max1 > max2 ? max1 : max2;
    }
    if (angel == possquint) {
        float *pixel1 = (float*)((char*)input + (row + 1)*pitch) + col;
        float *pixel2 = (float*)((char*)input + (row - 1)*pitch) + col;
        float *pixel3 = (float*)((char*)input + row*pitch) + col;
        float max1 = fmax(*pixel2, *pixel1), max2 = fmax(*pixel2, *pixel1);
        return max1 > max2 ? max1 : max2;
    }
    if (angel == negsquint) {
        float *pixel1 = (float*)((char*)input + (row + 1)*pitch) + (col + 1);
        float *pixel2 = (float*)((char*)input + (row - 1)*pitch) + (col - 1);
        float *pixel3 = (float*)((char*)input + row*pitch) + col;
        float max1 = fmax(*pixel2, *pixel1), max2 = fmax(*pixel2, *pixel1);
        return max1 > max2 ? max1 : max2;
    }
}

__global__ void canny(float *d_input, int height, int width, size_t pitch, 
                      float *blur, int radius, float theta, float *gradientX, float *gradientY, int window,
                      float *amplitude, float *angle, float *d_output) {
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    extern __shared__ float kernel[];

    if (row < height&&col < width) {
        // load gaussian kernel for convolving
        if (row < 2 * radius + 1 && col < 2 * radius + 1) {
            kernel[row*(2 * radius + 1) + col] = twoDimGaussian(col - radius, radius - row, theta);
        }

        // gaussian blur
        for (size_t i = 0; i < 2 * radius + 1; i++)
            for (size_t j = 0; j < 2 * radius + 1; j++) {
                // convolving, about how addressing matrix in device, 
                // see this link http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
                float *inputValue = (float *)((char *)d_input + row*pitch) + col;
                float *outputValue = (float *)((char *)blur + (row + i - radius)*pitch) + (col + j - radius);
                *outputValue += (float)(*inputValue) * kernel[i*(2 * radius + 1) + j];
            }

        // get gradient matrix for x direction
        for (size_t i = 0; i < SIZE; i++)
            for (size_t j = 0; j < SIZE; j++) {
                // convolving, about how addressing matrix in device, 
                // see this link http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c
                float *inputValue = (float *)((char *)blur + row*pitch) + col;
                float *x = (float *)((char *)gradientX + (row + i - radius)*pitch) + (col + j - radius);
                float *y = (float *)((char *)gradientY + (row + i - radius)*pitch) + (col + j - radius);
                *x += (float)(*inputValue) * XAxis[i][j];
                *y += (float)(*inputValue) * YAxis[i][j];
            }

        // get amplitude and angle
        float *x = (float *)((char *)gradientX + row*pitch) + col;
        float *y = (float *)((char *)gradientY + row*pitch) + col;

        float *ampl = (float*)((char*)amplitude + row*pitch) + col;
        *ampl = sqrtf(powf(*x, 2) + powf(*y, 2));

        float *angl = (float*)((char*)angle + row*pitch) + col;
        *angl = atan2f(*y, *x)*180.0f / PI;

        // non max signal repression
        if (0 <= *angl <= 22.5 && -22.5 <= *angl <= 0 && -180.0 <= *angl <= -157.5 && 157.5 <= *angl <= 180) {
            *x = 0.0; // use gradientX to store non max signal repression
        }
        if (22.5 <= *angl <= 67.5 && -157.5 <= *angl <= -112.5) {
            *x = -45.0;
        }
        if (67.5 <= *angl <= 112.5 && -112.5 <= *angl <= -67.5) {
            *x = 90.0;
        }
        if (112.5 <= *angl <= 157.5 && -67.5 <= *angl <= -22.5) {
            *x = 45.0;
        }
        if (window / 2 <= row <= height - window / 2 && window / 2 <= row <= width - window / 2) {
            float *x = (float *)((char *)gradientX + row*pitch) + col;
            float *out = (float *)((char *)d_output + row*pitch) + col;
            if (*x == 0)*out = maxPixel(row, col, d_input, pitch, window, horizontal);
            if (*x == -45.0)*out = maxPixel(row, col, d_input, pitch, window, negsquint);
            if (*x ==90.0)*out = maxPixel(row, col, d_input, pitch, window, vertical);
            if (*x == 45.0)*out = maxPixel(row, col, d_input, pitch, window, possquint);
        }
        
        // threshold
        float *out = (float *)((char *)d_output + row*pitch) + col;
        if (*out > 0.01) *out = 1.0;
    }
}

extern "C"
void cudaCanny(cv::Mat & input, cv::Mat & output, float radius, float theta, int window) {
    input.convertTo(input, CV_32F);
    output = cv::Mat(input.size(), CV_32F, cv::Scalar(0));
    float *d_input, *d_output, *blur, *gradientX, *gradientY, *amplitude, *angle;
    size_t pitch;

    cudaMallocPitch(&d_input, &pitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&d_output, &pitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&blur, &pitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&gradientX, &pitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&gradientY, &pitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&amplitude, &pitch, sizeof(float)*input.cols, input.rows);
    cudaMallocPitch(&angle, &pitch, sizeof(float)*input.cols, input.rows);

    const int N = 7;
    cudaStream_t streams[N];
    for (size_t i = 0; i < N; i++)cudaStreamCreate(&streams[i]);

    cudaMemcpy2DAsync(d_input, pitch, input.data, sizeof(float)*input.cols, sizeof(float)*input.cols, input.rows, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpy2DAsync(d_output, pitch, output.data, sizeof(float)*input.cols, sizeof(float)*input.cols, output.rows, cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpy2DAsync(blur, pitch, output.data, sizeof(float)*input.cols, sizeof(float)*input.cols, output.rows, cudaMemcpyHostToDevice, streams[2]);
    cudaMemcpy2DAsync(gradientX, pitch, output.data, sizeof(float)*input.cols, sizeof(float)*input.cols, output.rows, cudaMemcpyHostToDevice, streams[3]);
    cudaMemcpy2DAsync(gradientY, pitch, output.data, sizeof(float)*input.cols, sizeof(float)*input.cols, output.rows, cudaMemcpyHostToDevice, streams[4]);
    cudaMemcpy2DAsync(amplitude, pitch, output.data, sizeof(float)*input.cols, sizeof(float)*input.cols, output.rows, cudaMemcpyHostToDevice, streams[5]);
    cudaMemcpy2DAsync(angle, pitch, output.data, sizeof(float)*input.cols, sizeof(float)*input.cols, output.rows, cudaMemcpyHostToDevice, streams[6]);

    for (size_t i = 0; i < N; i++)cudaStreamSynchronize(streams[i]);

    dim3 blockSize(input.cols / (MAX_THREADS / 2) + 1, input.rows / MAX_THREADS + 1);
    dim3 threadSize(MAX_THREADS / 2, MAX_THREADS);

    int dynamicSize = (2 * radius + 1)*(2 * radius + 1) * sizeof(float);
    canny<<<blockSize, threadSize, dynamicSize>>>(d_input, input.rows, input.cols, pitch, blur, radius, theta, gradientX, gradientY, window, amplitude, angle, d_output);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy2D(output.data, sizeof(float)*output.cols, d_output, pitch, sizeof(float)*output.cols, output.rows, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < N; i++) cudaStreamDestroy(streams[i]);
    cudaFree(d_input); cudaFree(d_output); cudaFree(gradientX); cudaFree(gradientY); cudaFree(amplitude); cudaFree(angle); cudaFree(blur);
    input.convertTo(input, CV_8U);
    //output.convertTo(output, CV_8U);
}