#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

vector< vector< Point3f > > object_points;
vector< vector< Point2f > > imagePoints1, imagePoints2;
vector< Point2f > corners1, corners2;
vector< vector< Point2f > > left_img_points, right_img_points;

Mat imgL, imgR, grayL, grayR;

Size im_size;

void load_image_points(int board_width, int board_height, float square_size,
                      VideoCapture *capture)
{
  Size board_size = Size(board_width, board_height);
  int board_n = board_width * board_height;
  int img_n = 0;
  Mat frame;

  vector< Point3f > obj;
  for (int i = 0; i < board_height; i++)
    for (int j = 0; j < board_width; j++)
      obj.push_back(Point3f((float)j * square_size, (float)i * square_size, 0));

  while (capture->read(frame)) {
    if ( im_size == Size() ) {
      im_size = frame.size();
      im_size.width /= 2;
    }

    int cy = im_size.height;
    int cx = im_size.width;

    imgL = frame(Rect(0, 0, cx, cy));
    imgR = frame(Rect(cx, 0, cx, cy));

    cv::cvtColor(imgL, grayL, CV_BGR2GRAY);
    cv::cvtColor(imgR, grayR, CV_BGR2GRAY);

    bool found1 = false, found2 = false;

    found1 = cv::findChessboardCorners(imgL, board_size, corners1,
  CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    found2 = cv::findChessboardCorners(imgR, board_size, corners2,
  CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

    if (found1)
    {
      cv::cornerSubPix(grayL, corners1, cv::Size(5, 5), cv::Size(-1, -1),
  cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      //cv::drawChessboardCorners(gray1, board_size, corners1, found1);
    }
    if (found2)
    {
      cv::cornerSubPix(grayR, corners2, cv::Size(5, 5), cv::Size(-1, -1),
  cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      //cv::drawChessboardCorners(grayR, board_size, corners2, found2);
    }

    if (found1 && found2) {
      cout << img_n << ". Found both checkerboards" << endl;
      imagePoints1.push_back(corners1);
      imagePoints2.push_back(corners2);
      object_points.push_back(obj);
    }
    img_n++;
  }
  for (int i = 0; i < imagePoints1.size(); i++) {
    vector< Point2f > v1, v2;
    for (int j = 0; j < imagePoints1[i].size(); j++) {
      v1.push_back(Point2f((double)imagePoints1[i][j].x, (double)imagePoints1[i][j].y));
      v2.push_back(Point2f((double)imagePoints2[i][j].x, (double)imagePoints2[i][j].y));
    }
    left_img_points.push_back(v1);
    right_img_points.push_back(v2);
  }
}

int main(int argc, char const *argv[])
{
  const char* incalib_file = "intrinsics.yml";
  char* videoFilename = NULL;
  const char* out_file = "extrinsics.yml";

  static struct poptOption options[] = {
    { "video_filename",'v',POPT_ARG_STRING,&videoFilename,0,"Video file to read", "STR" },
    { "cameras_calibration_file",'u',POPT_ARG_STRING,&incalib_file,0,"cameras calibration","STR" },
    { "out_file",'o',POPT_ARG_STRING,&out_file,0,"Output calibration filename (YML)","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  FileStorage fsl(incalib_file, FileStorage::READ);

  if (!videoFilename) {
      cerr << "Please supply a video file name" << endl;
      exit(EXIT_FAILURE);
  }

  VideoCapture capture(videoFilename);
  if(!capture.isOpened()){
     //error in opening the video input
      cerr << "Unable to open video file: " << videoFilename << endl;
      exit(EXIT_FAILURE);
  }

  load_image_points(fsl["board_width"], fsl["board_height"], fsl["square_size"], &capture);

  printf("Starting Calibration\n");
  Mat K1, K2, R, F, E;
  Vec3d T;
  Mat D1, D2;
  fsl["K1"] >> K1;
  fsl["K2"] >> K2;
  fsl["D1"] >> D1;
  fsl["D2"] >> D2;
  int flag = CV_CALIB_FIX_INTRINSIC | CALIB_SAME_FOCAL_LENGTH;
  
  cout << "Read intrinsics" << endl;
  
  stereoCalibrate(object_points, left_img_points, right_img_points, K1, D1, K2, D2, im_size, R, T, E, F, flag);

  cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
  fs1 << "K1" << K1;
  fs1 << "K2" << K2;
  fs1 << "D1" << D1;
  fs1 << "D2" << D2;
  fs1 << "R" << R;
  fs1 << "T" << T;
  fs1 << "E" << E;
  fs1 << "F" << F;
  
  printf("Done Calibration\n");

  printf("Starting Rectification\n");

  cv::Mat R1, R2, P1, P2, Q;
  flag = CALIB_ZERO_DISPARITY;
  stereoRectify(K1, D1, K2, D2, im_size, R, T, R1, R2, P1, P2, Q, flag);

  fs1 << "R1" << R1;
  fs1 << "R2" << R2;
  fs1 << "P1" << P1;
  fs1 << "P2" << P2;
  fs1 << "Q" << Q;

  printf("Done Rectification\n");

  return 0;
}
