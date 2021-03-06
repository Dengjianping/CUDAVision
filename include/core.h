#ifndef CORE_H
#define CORE_H

#include "headers.h"

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

    extern "C"
        void cudaHarrisCorner(cv::Mat & input, cv::Mat & output, int radius, float theta, float k, float repression, int window);

    extern "C"
        void cudaCanny(cv::Mat & input, cv::Mat & output, float radius, float theta, int window);

    extern "C"
        void cudaHistogram(cv::Mat & input, cv::Mat & output);

    extern "C"
        void cudaIntegral(cv::Mat & input, cv::Mat & output);

    extern "C"
        void cudaFlip(cv::Mat & input, cv::Mat & output, Orientation ori);

    extern "C"
        void cudaGray(cv::Mat & input, cv::Mat & output);

    extern "C"
        void cudaSplit(cv::Mat & input, std::vector<cv::Mat> & channels);
}

#endif // !CORE_H