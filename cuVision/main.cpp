#include "headers.h"
#include "core.h"
#include "timerecorder.h"

using namespace std;
using namespace cv;

int main() {
    string path = "type-c.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);

    Mat result;

    TimeRecorder recorder;
    recorder.startRecord();
    cucv::cudaThreshold(img, result, 50);
    recorder.stopRecord();
    cout << "time cost on device: " << recorder.timeCost() << " ms" << endl;

    string title = "CUDA";
    namedWindow(title);
    imshow(title, result);
    waitKey(0);

    //system("pause");
    return 0;
}