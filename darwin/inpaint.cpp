#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>

// opencv library headers
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <opencv2/opencv.hpp>
// darwin library headers
#include "drwnBase.h"
#include "drwnVision.h"

int main(int argc, char *argv[])
{
  std::string imageFilename = argv[1];
  std::string outputFilename = "out.jpg";
  cv::Mat img = cv::imread(imageFilename, CV_LOAD_IMAGE_COLOR);
  DRWN_ASSERT_MSG(img.data, "could not read image from " << imageFilename);
    
  cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  cv::rectangle(mask, cv::Point(img.cols / 4, img.rows / 4),
		cv::Point(3 * img.cols / 4, 3 * img.rows / 4), cv::Scalar(0xff), -1);
  cv::Mat output;
  drwnInPaint::inPaint(img, output, mask);
  cv::imwrite(outputFilename, output);
  return 0;
}

