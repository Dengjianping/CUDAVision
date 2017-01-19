#include "headers.h"

class TimeRecorder {
private:
    cudaEvent_t start, end;
public:
    TimeRecorder();
    void startRecord();
    void stopRecord();
    float timeCost();
    ~TimeRecorder();
};

TimeRecorder::TimeRecorder() {
    cudaEventCreate(&start);
    cudaEventCreate(&end);
}

void TimeRecorder::startRecord() {
    cudaEventRecord(start);
}

void TimeRecorder::stopRecord() {
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
}

float TimeRecorder::timeCost() {
    float time;
    cudaEventElapsedTime(&time, start, end);
    return time;
}

TimeRecorder::~TimeRecorder() {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}