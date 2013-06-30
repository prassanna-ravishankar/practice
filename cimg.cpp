#include <Eigen/Core>
#include "CImg.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace cimg_library;
using namespace Eigen;

int main(int argc, char **argv) {
  CImg<float> image(argv[1]), visu(500,400,1,3,0);
  image < 'c';
  const int w = image.width();
  const int h = image.height();
  Map<const MatrixXf> R(image.data(), h , w);
  Map<const MatrixXf> G(image.data() + h * w, h,w);
  Map<const MatrixXf> B(image.data() + 2 * h * w, h,w);
  //image.RGBtoHSI().get_channel(2).display();
  MatrixXf R2(R);
  MatrixXf G2(G);
  MatrixXf B2(B);
  R2 *= 0.5;
  G2 *= 0.5;
  B2 *= 0.5;

//  std::vector<float> raw(h * w * 3);
//  std::copy(raw.begin(), raw.begin() + h * w, R2.data());
//  std::copy(raw.begin() + h * w, raw.begin() + 2 * h * w, G2.data());
//  std::copy(raw.begin() + 2 * h * w, raw.end(), B2.data());

//  for(int i = 0; i < h * w * 3; ++i) std::cout << raw[i] << std::endl;
//  CImg<float> img(h,w,1,3,0);
//  std::cout << img.spectrum() << std::endl;
//  CImg<float> r(R2.data(),h,w);
//  CImg<float> g(G2.data(),h,w);
//  CImg<float> b(B2.data(),h,w);

//  image.channel(1) = g;
//  image.channel(2) = b;
//  image.channel(0) = r;
//  //image.display();
//  image.display();
//  r.display();
//  std::cout << image.spectrum() << std::endl;
  return 0;
}
