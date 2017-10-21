#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
  const char* vid_filename = NULL;
  const char* out_filename = NULL;
  const char* calib_file = "extrinsics.yml";

  static struct poptOption options[] = {
    { "in_filename",'i',POPT_ARG_STRING,&vid_filename,0,"input video file","STR" },
    { "out_filename",'o',POPT_ARG_STRING,&out_filename,0,"out image path","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Stereo calibration file","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  if (!vid_filename) {
      cerr << "Please supply a video file name" << endl;
      exit(EXIT_FAILURE);
  }

  VideoCapture capture(vid_filename);
  if(!capture.isOpened()){
     //error in opening the video input
      cerr << "Unable to open video file: " << vid_filename << endl;
      exit(EXIT_FAILURE);
  }

  Mat R1, R2, P1, P2, Q;
  Mat K1, K2, R;
  Vec3d T;
  Mat D1, D2;
  Mat frame;
  Size im_size;
  int cy;
  int cx = 1280;

  cv::FileStorage fs1(calib_file, cv::FileStorage::READ);
  fs1["K1"] >> K1;
  fs1["K2"] >> K2;
  fs1["D1"] >> D1;
  fs1["D2"] >> D2;
  fs1["R"] >> R;
  fs1["T"] >> T;

  fs1["R1"] >> R1;
  fs1["R2"] >> R2;
  fs1["P1"] >> P1;
  fs1["P2"] >> P2;
  fs1["Q"] >> Q;

  cv::Mat lmapx, lmapy, rmapx, rmapy;
  cv::Mat imgU1, imgU2;

  int window_size = 9;
  int min_disp = 0; // -20;
  int numberOfDisparities = 128; // ((cx/8) + 15) & -16;
  
#if 1
  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create (min_disp, numberOfDisparities, window_size,
        /* P1 */ 8*3*window_size * window_size,
        /* P2 */ 32*3*window_size * window_size,
        /* disp12MaxDiff = */ 2,
        /* prefilterCaps */ 5,
        /*uniquenessRatio = */ 2, // 10,
        /*speckleWindowSize = */ 75, // 100,
        /*speckleRange = */ 2, // 32
        StereoSGBM::MODE_HH
    );
#else
  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create (min_disp, numberOfDisparities,
        7, 100, 1000, 32, 0, 15, 50, 16, StereoSGBM::MODE_SGBM_3WAY);
#endif

  int i = 0;
  while (capture.read(frame)) {
    if (i++ % 4 != 0)
      continue;
    if ( im_size == Size() ) {
      im_size = frame.size();
      im_size.width /= 2;
      cy = im_size.height;
      cx = im_size.width;
      cv::initUndistortRectifyMap(K1, D1, R1, P1, im_size, CV_32F, lmapx, lmapy);
      cv::initUndistortRectifyMap(K2, D2, R2, P2, im_size, CV_32F, rmapx, rmapy);
    }
    //imwrite(string("left") + out_filename, imgU1);
    //imwrite(string("right") + out_filename, imgU2);

    Mat img1 = frame(Rect(0, 0, cx, cy));
    Mat img2 = frame(Rect(cx, 0, cx, cy));

    cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
    cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

    Mat disparity, disparity_eq;

    cout << "Computing disparity with " << numberOfDisparities << " disparities" << endl;

    stereo->compute (imgU1, imgU2, disparity);

    disparity.convertTo(disparity_eq, CV_8U, 255/(numberOfDisparities*16.));

    if (out_filename)
      imwrite(string("disparity") + out_filename, disparity_eq);

    imshow("left", imgU1);
    imshow("right", imgU2);
    imshow("disparity", disparity_eq);
    c = (char)waitKey(10);
    if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
      exit(-1);
  }
  return 0;
}
