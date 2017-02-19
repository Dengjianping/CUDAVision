#ifndef CORE_H
#define CORE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace cucv {
    extern "C"
        void cudaThreshold(cv::Mat & input, cv::Mat & output, uchar thresholdValue);

    extern "C"
        void cudaErode(cv::Mat & input, cv::Mat & output, int iteration = 1);

    extern "C"
        void cudaDilate(cv::Mat & input, cv::Mat & output, int iteration = 1);

    extern "C"
        cudaError_t cudaLaplace(cv::Mat & input, cv::Mat & output);

    extern "C"
        void cudaGaussianBlur(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0);

    extern "C"
        void cudaSobel(cv::Mat & input, cv::Mat & output);

    extern "C"
        void cudaUSM(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0, float weight = 0.6);

    extern "C"
        void cudaUSMColor(cv::Mat & input, cv::Mat & output, int radius, float theta = 1.0, float weight = 0.6);
}

#endif // !CORE_H