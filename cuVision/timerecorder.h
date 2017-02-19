#include "headers.h"

class DeviceTimeRecorder {
private:
    cudaEvent_t start, end;
public:
    DeviceTimeRecorder();
    void startRecord();
    void stopRecord();
    float timeCost(); // ms
    ~DeviceTimeRecorder();
};

DeviceTimeRecorder::DeviceTimeRecorder() {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
}

void DeviceTimeRecorder::startRecord() {
    cudaEventRecord(start);
}

void DeviceTimeRecorder::stopRecord() {
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
}

float DeviceTimeRecorder::timeCost() {
    float time;
    cudaEventElapsedTime(&time, start, end);
    return time;
}

DeviceTimeRecorder::~DeviceTimeRecorder() {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

class HostTimeRecorder {
private:
    double start, end;
public:
    HostTimeRecorder();
    void startRecord();
    void stopRecord();
    double timeCost();
    ~HostTimeRecorder();
};

HostTimeRecorder::HostTimeRecorder() {
    start = end = 0.0;
}

void HostTimeRecorder::startRecord() {
    start = (double)cv::getTickCount();
}

void HostTimeRecorder::stopRecord() {
    end = (double)cv::getTickCount();
}

double HostTimeRecorder::timeCost() {
    return (end - start) / cv::getTickFrequency() * 1000; // ms
}

HostTimeRecorder::~HostTimeRecorder() {

}