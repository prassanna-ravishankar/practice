#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

CImg<unsigned char> stereo(CImg<unsigned char> im1,CImg<unsigned char> im2){
  CImg<unsigned char> disp;
  return disp;
}

int main(int argc, char *argv[])
{
  const string left = argv[1];
  const string right = argv[1];
  CImg<unsigned char> im1(left.c_str());
  CImg<unsigned char> im2(right.c_str());
  CImg<unsigned char> disp = stereo(im1,im2);
  disp.display();
  return 0;
}

