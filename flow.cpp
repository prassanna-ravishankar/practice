#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

CImg<unsigned char> compute_flow(CImg<unsigned char> im1,CImg<unsigned char> im2){
  CImg<unsigned char> flow;
  return flow;
}

int main(int argc, char *argv[])
{
  const string left = argv[1];
  const string right = argv[1];
  CImg<unsigned char> im1(left.c_str());
  CImg<unsigned char> im2(right.c_str());
  CImg<unsigned char> flow = compute_flow(im1,im2);
  flow.display();
  return 0;
}
