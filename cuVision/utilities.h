#include "headers.h"

namespace utility {
    // show image
    void showImage(const std::string & title, const cv::Mat & img) {
        cv::namedWindow(title);
        cv::imshow(title, img);
    }

    void callPython() {
        Py_Initialize();

        PyObject *pyModule = NULL;
        PyObject *pyFunction = NULL;

        pyModule = PyImport_AddModule("3d");
        pyFunction = PyObject_GetAttrString(pyModule, "twoDimGaussian");

        PyEval_CallFunction(pyFunction, NULL);

        Py_Finalize();
        return;
    }

    int deviceCount() {
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }
}