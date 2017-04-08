//#include "headers.h"
#include "include\core.h"
#include "utility\timerecorder.h"
#include "utility\utilities.h"
//#include "trial.h"

using namespace std;
using namespace cv;

int main() {
    string path = "images/type-c.jpg";
    //string path = "calc.jpg";
    //string path = "images/room.jpg";
    //string path = "image1.jpg";
    Mat img = imread(path, IMREAD_GRAYSCALE);
    //Mat img = imread(path);

    Mat eq;
    img.copyTo(eq);
    Mat diff;
    absdiff(img, eq, diff);
    cout << countNonZero(diff) << endl;;

    //bool equal = (eq == img);
    //cout << "equal: " << (eq == img) << endl;

    Mat result;
    
    /*int degree = 30;
    Point point = Point(img.cols / 2, img.rows / 2);
    Mat warp = getRotationMatrix2D(point, degree, 1);*/
    //warpAffine(img, img, warp, img.size());
    //dft(img32, djp,DFT_COMPLEX_OUTPUT);
    //trial::fftImage(img32, djp, "DFT");

    // thresholdimg
    //threshold(img, img, 50, 255, THRESH_BINARY);
    double a[3][3] = { {1,2,3},{2,5,8},{4,6,2} };
    Mat inte(3,3, CV_32F, a);

    vector<Mat> ch;
    
    utility::showImage("CPU", img);

    DeviceTimeRecorder recorder;
    recorder.startRecord();
    //cucv::cudaThreshold(img, result, 50);
    //cucv::cudaDilate(img, result, 1);
    //cucv::cudaUSM(img, result, 2);
    //cucv::cudaGaussianBlur(img, result, 2);
    //cucv::cudaSobel(img, result);
    //cucv::cudaHistogram(img, result);
    //cucv::cudaIntegral(img, result);
    //cucv::cudaHarrisCorner(img, result, 2, 1.0, 0.06, 0.01, 3);
    //cucv::cudaUSMColor(img, result, 3);
    //cucv::cudaCanny(img, result, 1, 1, 3);
    cucv::cudaFlip(img, result, angel_90);
    //cucv::cudaGray(img, result);
    //cucv::cudaSplit(img, ch);
    recorder.stopRecord();
    cout << "time cost on device: " << recorder.timeCost() << " ms" << endl;

    vector<Mat> host;
    HostTimeRecorder record;
    record.startRecord();
    split(img, host);
    //cvtColor(img, img, CV_RGB2GRAY);
    Sobel(img, img, img.depth(), 1, 0);
    record.stopRecord();
    cout << "time cost on host: " << record.timeCost() << " ms" << endl;

    //utility::callPython();

    string title = "CUDA";
    utility::showImage(title, result);
    waitKey(0);

    //imwrite("flip90.jpg", result);

    //system("pause");
    return 0;
}