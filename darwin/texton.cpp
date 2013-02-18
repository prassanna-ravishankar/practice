#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>
#include <opencv2/opencv.hpp>

#include "drwnBase.h"
#include "drwnVision.h"

using namespace std;
using namespace cv;

void visualize(vector<IplImage *> response){
    for(int i = 0; i < response.size(); ++i){
        Mat img(response[i]);
        drwnShowDebuggingImage(response[i],"texton feature",true);
        //imshow("texton feature", img);
//        char key = (char)waitKey(5); //delay N millis, usually long enough to display and capture input
//        switch (key) {
//        case 'q':
//        case 'Q':
//        case 27: //escape key
//            return ;
//        default:
//            break;
//        }

    }
}

int main(int argc, char *argv[])
{

  Mat img = imread(argv[1]);
  drwnTextonFilterBank filter;
  drwnHOGFeatures hog;
  vector<IplImage *> response,features;
  IplImage tmp = img.operator IplImage();
  Ptr<IplImage> img_ptr = &tmp;
  filter.filter(img_ptr, response);
  hog.computeDenseFeatures(img_ptr,response);
  visualize(response);
  response.clear();
  features.clear();
  return 0;
}
