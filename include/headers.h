#ifndef HEADERS_H
#define HEADERS_H

// std include files
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdio.h>

// cuda include files
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_fp16.h"

// opencv include files
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

enum Orientation { angel_90, angel_180, angel_270 };


#endif // !HEADERS_H