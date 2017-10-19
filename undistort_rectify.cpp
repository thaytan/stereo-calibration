#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include "popt_pp.h"

using namespace std;
using namespace cv;

static void saveXYZ(const char* filename, const Mat& mat, const Mat& img, const Mat& mask)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            Vec3b col = img.at<Vec3b>(y, x);
            uint8_t alpha = mask.at<uint8_t>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            if (alpha == 0) continue; /* Ignore alpha=0.0 color */

            fprintf(fp, "%f;%f;%f;%u;%u;%u\n", point[0], point[1], point[2], col[2], col[1], col[0]);
        }
    }
    fclose(fp);
}

static void
reproject_and_save (cv::Mat &disparity, cv::Mat &in_img, cv::Mat &mask, cv::Mat Q, const char *filename)
{
  Mat disparityF;
  Mat disparityT;
  Mat QF;
  Mat img;

  //cv::normalize(disparity, disparityF, 0, 256, cv::NORM_MINMAX, CV_32F);
  disparity.convertTo( disparityF, CV_32F, 1./16);
  in_img.convertTo(img, CV_8U);

  cv::Mat_<cv::Vec3f> XYZ(disparityF.rows,disparityF.cols);   // Output point cloud

  Q.convertTo( QF, CV_32F, 1.);
  QF.at<float>(3,3)=-QF.at<float>(3,3);

#if 1
  cv::Mat_<float> vec_tmp(4,1);
  for(int y=0; y<disparityF.rows; ++y) {
      for(int x=0; x<disparityF.cols; ++x) {
          vec_tmp(0)=x; vec_tmp(1)=y; vec_tmp(2)=disparityF.at<float>(y,x); vec_tmp(3)=1;
          vec_tmp = QF*vec_tmp;
          vec_tmp /= vec_tmp(3);
          cv::Vec3f &point = XYZ.at<cv::Vec3f>(y,x);
          point[0] = vec_tmp(0);
          point[1] = vec_tmp(1);
          point[2] = vec_tmp(2);
      }
  }
#else
  reprojectImageTo3D(disparityF, XYZ, QF, true);
#endif

  saveXYZ(filename, XYZ, img, mask);
}

int main(int argc, char const *argv[])
{
  const char* img_filename = NULL;
  const char* out_filename = NULL;
  const char* calib_file = "extrinsics.yml";
  const char* point_cloud_filename = NULL;

  static struct poptOption options[] = {
    { "in_filename",'i',POPT_ARG_STRING,&img_filename,0,"input image path","STR" },
    { "out_filename",'o',POPT_ARG_STRING,&out_filename,0,"out image path","STR" },
    { "calib_file",'c',POPT_ARG_STRING,&calib_file,0,"Stereo calibration file","STR" },
    { "point_cloud",'p',POPT_ARG_STRING,&point_cloud_filename,0,"Write point cloud","STR" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  POpt popt(NULL, argc, argv, options, 0);
  int c;
  while((c = popt.getNextOpt()) >= 0) {}

  if (img_filename == NULL || out_filename == NULL) {
    cerr << "Please supply input and output file names" << endl;
    exit (1);
  }
  Mat R1, R2, P1, P2, Q;
  Mat K1, K2, R;
  Vec3d T;
  Mat D1, D2;
  Mat img = imread(img_filename, CV_LOAD_IMAGE_COLOR);
  Size im_size = img.size();
  int cy = im_size.height;
  int cx = im_size.width / 2;

  Mat img1 = img(Rect(0, 0, cx, cy));
  Mat img2 = img(Rect(cx, 0, cx, cy));

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

  /* Alpha mask used to ignore useless pixels in output */
  cv::Mat mask(cx, cy, CV_8U);
  cv::Mat maskU;
  mask = cv::Scalar(255);

  cv::initUndistortRectifyMap(K1, D1, R1, P1, img1.size(), CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, img2.size(), CV_32F, rmapx, rmapy);

  cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR, BORDER_CONSTANT);
  cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR, BORDER_CONSTANT);
  cv::remap(mask, maskU, lmapx, lmapy, cv::INTER_LINEAR, BORDER_CONSTANT);

  imwrite(string("left") + out_filename, imgU1);
  imwrite(string("right") + out_filename, imgU2);

  int window_size = 7;
  int min_disp = 0;
  int numberOfDisparities = ((cx/8) + 15) & -16;

  cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create (min_disp, numberOfDisparities, window_size,
        /* P1 */ 8*3*window_size * window_size,
        /* P2 */ 32*3*window_size * window_size,
        /* disp12MaxDiff = */ 1,
        /* prefilterCaps */ 63,
        /*uniquenessRatio = */ 2, // 10,
        /*speckleWindowSize = */ 50, // 100,
        /*speckleRange = */ 2, // 32
        StereoSGBM::MODE_HH
    );

  Mat disparity, disparity_eq;

  cout << "Computing disparity with " << numberOfDisparities << " disparities" << endl;

  stereo->compute (imgU1, imgU2, disparity);

  /* Scale from signed 16-bit fixed point to 0..255 for display and storage */
  //disparity.convertTo(disparity_eq, CV_8U, 255/(numberOfDisparities*16.));
  cv::normalize(disparity, disparity_eq, 0, 256, cv::NORM_MINMAX, CV_8U);

  imwrite(string("disparity") + out_filename, disparity_eq);

  if(point_cloud_filename != NULL)
  {
    printf("storing the point cloud...");
    fflush(stdout);
    reproject_and_save (disparity, imgU1, maskU, Q, point_cloud_filename);
    printf("\n");
  }

  imshow("left", imgU1);
  imshow("right", imgU2);
  imshow("disparity", disparity_eq);
  c = (char)waitKey(50000);
  if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
    exit(-1);

  return 0;
}
