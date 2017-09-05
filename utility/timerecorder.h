#include "..\include\headers.h"

class TimeRecorder
{
public:
    virtual void startRecord() = 0;
    virtual void stopRecord() = 0;
    virtual float timeCost() = 0;
};

class DeviceTimeRecorder: public TimeRecorder
{
private:
    cudaEvent_t start, end;
public:
    DeviceTimeRecorder();
    virtual void startRecord();
    virtual void stopRecord();
    virtual float timeCost(); // ms
    ~DeviceTimeRecorder();
};

DeviceTimeRecorder::DeviceTimeRecorder() 
{
    cudaEventCreate(&start);
    cudaEventCreate(&end);
}

void DeviceTimeRecorder::startRecord() 
{
    cudaEventRecord(start);
}

void DeviceTimeRecorder::stopRecord() 
{
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
}

float DeviceTimeRecorder::timeCost() 
{
    float time;
    cudaEventElapsedTime(&time, start, end);
    return time;
}

DeviceTimeRecorder::~DeviceTimeRecorder() 
{
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

class HostTimeRecorder: public TimeRecorder
{
private:
    double start, end;
public:
    HostTimeRecorder();
    virtual void startRecord();
    virtual void stopRecord();
    virtual float timeCost();
    ~HostTimeRecorder();
};

HostTimeRecorder::HostTimeRecorder() 
{
    start = end = 0.0;
}

void HostTimeRecorder::startRecord() 
{
    start = (float)cv::getTickCount();
}

void HostTimeRecorder::stopRecord() 
{
    end = (float)cv::getTickCount();
}

float HostTimeRecorder::timeCost() 
{
    return (end - start) / cv::getTickFrequency() * 1000; // ms
}

HostTimeRecorder::~HostTimeRecorder() 
{}