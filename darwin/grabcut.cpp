/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    grabCut.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
** DESCRIPTION:
**  Implementation of the grabCut algorithm by Rother et al., 2004.
**
*****************************************************************************/

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// opencv library headers
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnVision.h"
#include "drwnML.h"
using namespace std;
using namespace Eigen;

// main ---------------------------------------------------------------------

void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./grabCut [OPTIONS] <image> (<mask>)\n";
    cerr << "OPTIONS:\n"
         << "  -k <num>          :: number of mixture components (default: 5)\n"
         << "  -m <samples>      :: max samples to use when learning colour models\n"
         << "  -o <dir>          :: set output directory for segmentation masks and images\n"
         << "  -s <scale>        :: rescale input\n"
         << "  -w <weight>       :: use pairwise weight provided (otherwise tries many)\n"
         << "  -x                :: visualize\n"
         << "  -scm <file>       :: save final colour models to <file>\n"
         << "  -lcm <file>       :: load initial colour models from <file>\n"
         << DRWN_STANDARD_OPTIONS_USAGE
	 << endl;
}

int main(int argc, char *argv[])
{
    // default parameters
    const char *outDir = NULL;
    double scale = 1.0;
    double weight = -1.0;
    bool bVisualize = false;
    const char *finalColourModelFile = NULL;
    const char *initialColourModelFile = NULL;

    // process commandline arguments
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_INT_OPTION("-k", drwnGrabCutInstance::numMixtures)
        DRWN_CMDLINE_INT_OPTION("-m", drwnGrabCutInstance::maxSamples)
        DRWN_CMDLINE_STR_OPTION("-o", outDir)
        DRWN_CMDLINE_REAL_OPTION("-s", scale)
        DRWN_CMDLINE_REAL_OPTION("-w", weight)
        DRWN_CMDLINE_BOOL_OPTION("-x", bVisualize)
        DRWN_CMDLINE_STR_OPTION("-scm", finalColourModelFile)
        DRWN_CMDLINE_STR_OPTION("-lcm", initialColourModelFile)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if ((DRWN_CMDLINE_ARGC != 1) && (DRWN_CMDLINE_ARGC != 2)) {
        usage();
        return -1;
    }

    drwnCodeProfiler::tic(drwnCodeProfiler::getHandle("main"));

    // load image
    const char *imgFilename = DRWN_CMDLINE_ARGV[0];
    const char *maskFilename = DRWN_CMDLINE_ARGC == 1 ? NULL : DRWN_CMDLINE_ARGV[1];
    IplImage *img = cvLoadImage(imgFilename, CV_LOAD_IMAGE_COLOR);
    CvMat *mask = NULL;

    // load mask
    mask = cvCreateMat(img->height, img->width, CV_8UC1);
    if (maskFilename == NULL) {
        CvRect bb = drwnInputBoundingBox("annotate", img);
        DRWN_ASSERT((bb.width > 0) && (bb.height > 0));
        cvSet(mask, cvScalar(drwnGrabCutInstance::MASK_BG));
        cvRectangle(mask, cvPoint(bb.x, bb.y), cvPoint(bb.x + bb.width, bb.y + bb.height),
            cvScalar(drwnGrabCutInstance::MASK_C_FG), -1);
    } else {
        IplImage *m = cvLoadImage(maskFilename, CV_LOAD_IMAGE_GRAYSCALE);
        cvConvertScale(m, mask);
        cvReleaseImage(&m);
    }

    // rescale image and mask
    if (scale != 1.0) {
        drwnResizeInPlace(&img, (int)(scale * img->height), (int)(scale * img->width));
        drwnResizeInPlace(&mask, img->height, img->width, CV_INTER_NN);
    }

    // show image and mask
    if (bVisualize) {
        drwnShowDebuggingImage(img, "image", false);
        drwnShowDebuggingImage(mask, "mask", false);
    }
    drwnGrabCutInstance::bVisualize = bVisualize;

    // run grabCut with different weights
    const double minWeight = (weight < 0.0) ? 1.0 : weight;
    const double maxWeight = (weight < 0.0) ? 256.0 : weight;
    drwnGrabCutInstance model;
    model.name = strBaseName(imgFilename);
    for (double w = minWeight; w <= maxWeight; ) {
        // initialize model
        model.initialize(img, mask, initialColourModelFile);
        model.setBaseModelWeights(1.0, 0.0, w);
        CvMat *seg = model.inference();

        // save segmentation mask
        if (outDir != NULL) {
            string wStr = strReplaceSubstr(toString(0.01 * (int)(w * 100)), ".", "_");
            string filename = string(outDir) + strBaseName(string(imgFilename)) +
                string("_mask_") + wStr + string(".png");
            DRWN_LOG_VERBOSE("writing segmentation to " << filename << "...");
            IplImage *m = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
            cvConvertScale(seg, m);
            cvSaveImage(filename.c_str(), m);
            cvReleaseImage(&m);

            filename = string(outDir) + strBaseName(string(imgFilename)) +
                string("_img_") + wStr + string(".png");
            DRWN_LOG_VERBOSE("writing segmented image to " << filename << "...");
            m = cvCloneImage(img);
            cvCmpS(seg, 0.0, seg, CV_CMP_EQ);
            cvSet(m, CV_RGB(0, 0, 255), seg);
            cvSaveImage(filename.c_str(), m);
            cvReleaseImage(&m);
        }

        // free segmentation
        cvReleaseMat(&seg);

        // update weight
        if (w == 0.0) w = 1.0; else w *= 2.0;
    }

    // save final colour model file
    if (finalColourModelFile != NULL) {
        DRWN_LOG_VERBOSE("writing colour models to " << finalColourModelFile << "...");
        model.saveColourModels(finalColourModelFile);
    }

    if (bVisualize && (weight >= 0.0)) {
        cvWaitKey(-1);
    }

    // free memory
    cvReleaseMat(&mask);
    cvReleaseImage(&img);
    drwnCodeProfiler::toc(drwnCodeProfiler::getHandle("main"));
    drwnCodeProfiler::print();
    return 0;
}

