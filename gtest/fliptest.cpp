#include <gtest\gtest.h>

#include "..\include\core.h"

struct SameImage
{
    int depth, // image depth
        channel, // image channel
        diff; // whether the image has different pixels
};

class ImageFlipTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        std::string path = "flip90.jpg";
        cv::Mat original = cv::imread(path);
        cv::Mat result;
        cucv::cudaFlip(original, result, angel_90);
    }

    virtual void TearDown()
    {

    }
};