#include "headers.h"

namespace cucv {
    extern "C"
        cudaError_t cudaThreshold(cv::Mat & input, cv::Mat & output, uchar thresholdValue);

    extern "C"
        cudaError_t cudaDilate(cv::Mat & input, cv::Mat & output);
}