#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "drwnBase.h"
#include "drwnVision.h"

using namespace std;
using namespace cv;

void visualize(const Mat& img, const Mat& seg){
    IplImage cvimg = (IplImage)img;
    CvMat cvseg = (CvMat)seg;
    drwnAverageRegions(&cvimg, &cvseg);
    drwnDrawRegionBoundaries(&cvimg, &cvseg, CV_RGB(0, 0, 0));
    drwnShowDebuggingImage(&cvimg, "superpixels", true);
}

int main(int argc, char *argv[])
{
    unsigned gridSize = 10;
    const char *outFile = NULL;

    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-g", gridSize)
    DRWN_END_CMDLINE_PROCESSING();

    drwnCodeProfiler::enabled = true;
    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    cv::Mat img = cv::imread(imgFilename, CV_LOAD_IMAGE_COLOR);

    cv::Mat seg = drwnFastSuperpixels(img, gridSize);

    visualize(img,seg);

    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}
