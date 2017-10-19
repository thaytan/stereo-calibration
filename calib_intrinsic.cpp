#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

vector< vector< Point3f > > l_object_points;
vector< vector< Point2f > > l_img_points;
vector< vector< Point3f > > r_object_points;
vector< vector< Point2f > > r_img_points;

Mat imgL, grayL;
Mat imgR, grayR;
Size im_size;

void setup_calibration(int board_width, int board_height,
                       float square_size, VideoCapture *capture, bool show_output = false) {
  Size board_size = Size(board_width, board_height);
  int board_n = board_width * board_height;
  Mat frame;
  int k = 0;
  vector< Point2f > corners;

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

    bool found = false;

    found = cv::findChessboardCorners(grayR, board_size, corners,
                                      CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    if (found)
    {
      cornerSubPix(grayR, corners, cv::Size(5, 5), cv::Size(-1, -1),
                   TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      if (show_output) {
        drawChessboardCorners(imgR, board_size, corners, found);
        imshow("cornersR", imgR);
        char c = (char)waitKey(500);
        if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
          exit(-1);
      }
      cout << k << ". Found " << corners.size() << " right corners" << endl;
      r_img_points.push_back(corners);
      r_object_points.push_back(obj);
    }

    found = cv::findChessboardCorners(grayL, board_size, corners,
                                      CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
    if (found)
    {
      cornerSubPix(grayL, corners, cv::Size(5, 5), cv::Size(-1, -1),
                   TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
      if (show_output) {
        drawChessboardCorners(imgL, board_size, corners, found);
        imshow("cornersL", imgL);
        char c = (char)waitKey(500);
        if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
          exit(-1);
      }
      cout << k << ". Found " << corners.size() << " left corners" << endl;
      l_img_points.push_back(corners);
      l_object_points.push_back(obj);
    }
    k++;
  }
}

double computeReprojectionErrors(const vector< vector< Point3f > >& objectPoints,
                                 const vector< vector< Point2f > >& imagePoints,
                                 const vector< Mat >& rvecs, const vector< Mat >& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs) {
  vector< Point2f > imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  vector< float > perViewErrors;
  perViewErrors.resize(objectPoints.size());

  for (i = 0; i < (int)objectPoints.size(); ++i) {
    projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                  distCoeffs, imagePoints2);
    err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
    int n = (int)objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr += err*err;
    totalPoints += n;
  }
  return std::sqrt(totalErr/totalPoints);
}

int main(int argc, char const **argv)
{
  int board_width = 8, board_height = 6;
  int show_images = 0;
  float square_size = 1.0;
  char* videoFilename = NULL;
  const char* out_file = "intrinsics.yml";

  static struct poptOption options[] = {
    { "show_images",'i',POPT_ARG_NONE,&show_images,0,"Display found checkerboard corners", NULL },
    { "board_width",'w',POPT_ARG_INT,&board_width,0,"Checkerboard width","NUM" },
    { "board_height",'h',POPT_ARG_INT,&board_height,0,"Checkerboard height","NUM" },
    { "square_size",'s',POPT_ARG_FLOAT,&square_size,0,"Size of checkerboard square","NUM" },
    { "video_filename",'v',POPT_ARG_STRING,&videoFilename,0,"Video file to read", "STR" },
    { "out_file",'o',POPT_ARG_STRING,&out_file,0,"Output calibration filename (YML)","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

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

  setup_calibration(board_width, board_height, square_size, &capture, show_images);

  printf("Starting Calibration with %d left and %d right images\n", l_img_points.size(), r_img_points.size());
  Mat K_l, K_r;
  Mat D_l, D_r;

  vector< Mat > rvecs, tvecs;
  int flag = CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5;

  calibrateCamera(r_object_points, r_img_points, im_size, K_r, D_r, rvecs, tvecs, flag);
  cout << "Right Calibration error: " << computeReprojectionErrors(r_object_points, r_img_points, rvecs, tvecs, K_r, D_r) << endl;

  calibrateCamera(l_object_points, l_img_points, im_size, K_l, D_l, rvecs, tvecs, flag);

  cout << "Left Calibration error: " << computeReprojectionErrors(l_object_points, l_img_points, rvecs, tvecs, K_l, D_l) << endl;


  FileStorage fs(out_file, FileStorage::WRITE);
  fs << "K1" << K_l;
  fs << "D1" << D_l;
  fs << "K2" << K_r;
  fs << "D2" << D_r;
  fs << "board_width" << board_width;
  fs << "board_height" << board_height;
  fs << "square_size" << square_size;
  printf("Done Calibration\n");

  return 0;
}
