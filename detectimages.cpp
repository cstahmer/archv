#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

using namespace cv;
using namespace std;

int usage (char **argv);
void read_flags (int argc, char **argv, string *imgfile1, string *imgfile2, string *output, string *param, double *scale, double *ratio);
void read_surfparams (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin);
void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin);
void showkeypts (vector <KeyPoint> &keypoints, Mat &drawImg);
Mat DrawMatch(Mat& image1, vector<KeyPoint>& keypoints1, Mat& image2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches1to2);
int ratioTest(vector<vector<DMatch> > &matches, double ratio);
void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches);
Mat ransacTest(const vector<DMatch>& matches,const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches);

int main(int argc, char** argv)
{
/* ===============================================================================================
   Show usage if needed
   =============================================================================================== */
   if (argc < 2)
    return usage(argv);
   string argument = argv[1];
   if (argument == "-h" || argument == "-help")
    return usage(argv);

/* ===============================================================================================
   Initialize parameters for SURF
   =============================================================================================== */
   int minhessian, octaves, layers, SizeMin;
   double RespMin;
   double scale, ratio;
   string param, output;
   string imgfile1, imgfile2;

/* ===============================================================================================
   Read in parameters for run
   =============================================================================================== */
   read_flags (argc, argv, &imgfile1, &imgfile2, &output, &param, &scale, &ratio);

/* ===============================================================================================
   Read in parameters for SURF
   =============================================================================================== */
   read_surfparams (param, &minhessian, &octaves, &layers, &SizeMin, &RespMin);

/* ===============================================================================================
   Create all structures that are needed to process the images:
        - use SURF for key points detection and feature extraction
   =============================================================================================== */
   vector <KeyPoint> keypoints1;
   vector <KeyPoint> keypoints2;
   SurfFeatureDetector detector (minhessian, octaves, layers);          // SURF detector
   Ptr <DescriptorExtractor> extractor = new SurfDescriptorExtractor(); // Feature Extractor from Keypoints

   BFMatcher matcher;
   vector <vector<DMatch>> matches1;
   vector <vector<DMatch>> matches2;
   vector <DMatch> sym_matches;
   vector <DMatch> matches;

/*  ===============================================================================================
    Process images:
	- detect keypoints
	- filter keypoints
    =============================================================================================== */
   Mat img1, img2;
   img1 = imread (imgfile1);
   img2 = imread (imgfile2);

   detector.detect (img1, keypoints1);
   detector.detect (img2, keypoints2);

   filter_keypoints (keypoints1, SizeMin, RespMin);
   filter_keypoints (keypoints2, SizeMin, RespMin);

   cout << "Number of keypoints 1 : " << keypoints1.size() << endl;
   cout << "Number of keypoints 2 : " << keypoints2.size() << endl;

/*  ===============================================================================================
    Generate descriptors from key points
    =============================================================================================== */
   Mat descriptors1, descriptors2;
   extractor->compute(img1, keypoints1, descriptors1);
   extractor->compute(img2, keypoints1, descriptors2);

/*  ===============================================================================================
    Find matches based on descriptors: 
	- first from img1 to img2 (with 2 NN), then from img2 to img1
	- filter based on ratio test
	- filter for symmetry
	- filter by RANSAC
    =============================================================================================== */
   matcher.knnMatch (descriptors1, descriptors2, matches1, 2);
   matcher.knnMatch (descriptors2, descriptors1, matches2, 2);

   int removed = ratioTest (matches1, ratio);
   removed = ratioTest (matches2, ratio);

   symmetryTest (matches1, matches2, sym_matches);

   Mat fundemental = ransacTest (sym_matches, keypoints1, keypoints2, matches);

/*  ===============================================================================================
    Only keep "good" keypoints (i.e. those that correspond to good matches
    =============================================================================================== */
   vector <KeyPoint> keypt1;
   vector <KeyPoint> keypt2;

   int count = 0;
   for (int m = 0; m < matches.size(); m++)
   {
     int i1 = matches[m].queryIdx;
     int i2 = matches[m].trainIdx;

     keypt1.push_back(keypoints1[i1]);
     keypt2.push_back(keypoints2[i2]);

     count++;
   }
   cout << "matches size: " << count << endl;

/*  ===============================================================================================
    Generate image with keypoints and good matches
    =============================================================================================== */
   Mat imgkpts1;
   Mat imgkpts2;

   img1.copyTo(imgkpts1);
   img2.copyTo(imgkpts2);

   Mat img_matches;
   img_matches = DrawMatch (imgkpts1, keypoints1, imgkpts2, keypoints2, matches);

/*  ===============================================================================================
    Localize image1 into image2
    =============================================================================================== */
   vector <Point2f> obj1;
   vector <Point2f> obj2;

   for (int i = 0; i < matches.size(); i++)
   {
     obj1.push_back (keypoints1[ matches[i].queryIdx ].pt );
     obj2.push_back (keypoints2[ matches[i].trainIdx ].pt );
   }

   Mat H = findHomography (obj1, obj2, CV_RANSAC);

   vector <Point2f> obj1_corners(4);
   obj1_corners[0] = cvPoint (0,0); 
   obj1_corners[1] = cvPoint (img1.cols, 0);
   obj1_corners[2] = cvPoint (img1.cols, img1.rows);
   obj1_corners[3] = cvPoint (0, img1.rows);

   vector<Point2f> obj2_corners(4);

   perspectiveTransform (obj1_corners, obj2_corners, H);

   double spacer = 20;
   double offset = img1.cols + spacer;
   line (img_matches, obj2_corners[0] + Point2f (offset, 0), obj2_corners[1] + Point2f (offset, 0), Scalar (0, 255, 0), 4);
   line (img_matches, obj2_corners[1] + Point2f (offset, 0), obj2_corners[2] + Point2f (offset, 0), Scalar (0, 255, 0), 4);
   line (img_matches, obj2_corners[2] + Point2f (offset, 0), obj2_corners[3] + Point2f (offset, 0), Scalar (0, 255, 0), 4);
   line (img_matches, obj2_corners[3] + Point2f (offset, 0), obj2_corners[0] + Point2f (offset, 0), Scalar (0, 255, 0), 4);

/*  ===============================================================================================
    Rescale matched image
    =============================================================================================== */
   Mat img_scaled;
   int interpolation = INTER_LINEAR;
   resize (img_matches, img_scaled, Size(), scale, scale, interpolation);

/*  ===============================================================================================
    save results
    =============================================================================================== */
   imwrite (output, img_scaled);

    return 0;
}




int usage (char **argv)
{
  cout << "\n this program finds if one image can be found within another, and then displays them side by side" << endl;
  cout << "./detectimages.exe -i1 <img file 1> -i2 <img file 2> -o <output img file> -p <param file> -s <scale> -r <ratio>" << endl;
  return 1;
}

void read_flags (int argc, char **argv, string *imgfile1, string *imgfile2, string *output, string *param, double *scale, double *ratio)
{
  string input;
  for (int i = 1; i < argc; i++)  
  {
    input = argv[i];
    if (input == "-i1")
      *imgfile1 = argv[i+1];
    if (input == "-i2")
      *imgfile2 = argv[i+1];
    if (input == "-o")
      *output = argv[i+1];
    if (input == "-p")
      *param = argv[i+1];
    if (input == "-r")
      *ratio = atof(argv[i+1]);
    if (input == "-s")
      *scale = atof(argv[i+1]);
  }
}

void read_surfparams (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin)
{
  ifstream inFile;
  inFile.open(param);
  string record;
  stringstream ss;

  while (!inFile.eof())
  {
    getline (inFile, record);
    if (record.find("minhessian") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *minhessian;
      ss.str("");
      ss.clear();
    }
    if (record.find("octaves") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *octaves;
      ss.str("");
      ss.clear();
    }
    if (record.find("layers") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *layers;
      ss.str("");
      ss.clear();
    }
    if (record.find("min Size") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *SizeMin;
      ss.str("");
      ss.clear();
    }
    if (record.find("min Resp") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *RespMin;
      ss.str("");
      ss.clear();
    }
  }
}

void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin)
{
  vector <KeyPoint> keypoint2;
  int npoints = keypoints.size();
  int size;
  double response;

/*    ==============================================================================================
      filter keypoints based on size
      ============================================================================================== */
      for (int i = 0; i < npoints; i++)
      {
        size = keypoints[i].size;
        if (size > SizeMin)
          keypoint2.push_back(keypoints[i]);
      }

      keypoints.clear();
      keypoints = keypoint2;
      npoints = keypoints.size();
      keypoint2.clear();

/*    ==============================================================================================
      filter keypoints based on size
      ============================================================================================== */
      for (int i = 0; i < npoints; i++)
      {
        response = keypoints[i].response;
        if (response > RespMin)
          keypoint2.push_back(keypoints[i]);
      }

      keypoints.clear();
      keypoints = keypoint2;
}

void showkeypts (vector <KeyPoint> &keypoints, Mat &drawImg)
{
/* =====================================================================================
	all variables needed for openCV circle function, n is number of keypoints 
  ===================================================================================== */
  int radius, thickness = 1, lineType = 8, shift = 0;
  int n = keypoints.size();

/* =====================================================================================
	for each keypoint: draw a circle with radius scaled to keypoints size around
  ===================================================================================== */
  for (int i = 0; i < n; i++)
  {
    radius = keypoints[i].size / 4;
    circle (drawImg, keypoints[i].pt, radius, Scalar (255,0,0), thickness, lineType, shift); 
  }
  return;
}

Mat DrawMatch(Mat& image1, vector<KeyPoint>& keypoints1, Mat& image2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches1to2)
{
  Mat newimage;
  int offset_x;
  int width, height;
  int spacer_x;

  // get information from the images
  spacer_x = 20;
  width = image1.cols + image2.cols + spacer_x;
  height = image1.rows;
  if (image2.rows > height)
    height = image2.rows;

  // create newimage with white background and size it such that it can contain both
  newimage = Mat (cvSize (width, height), CV_8UC3, Scalar (255,255,255));


  // place all the images onto the newimage
  Mat TargetROI;
  offset_x = 0;

    /* first put image on the left: 
       - Set a rectangle where we want to put the image
       - Define a "Region of Interest" corresponding to this rectangle
       - Copy the image in that rectangle */
       Rect ROI (offset_x, 0, image1.cols, image1.rows);
       TargetROI = newimage (ROI);
       image1.copyTo (TargetROI);

    // repeat for image representing the histogram on the right
      offset_x = offset_x + image1.cols + spacer_x;
      Rect ROI2 (offset_x, 0, image2.cols, image2.rows);
      TargetROI = newimage (ROI2);
      image2.copyTo (TargetROI);


  // now add lines between the matched keypoints
  int thickness = 4;
  int lineType = 8;
  int shift = 0;

  for (int m = 0; m < matches1to2.size(); m++)
  {
    int i1 = matches1to2[m].queryIdx;
    int i2 = matches1to2[m].trainIdx;
    CV_Assert (i1 >= 0 && i1 < static_cast<int> (keypoints1.size()));
    CV_Assert (i2 >= 0 && i2 < static_cast<int> (keypoints2.size()));

    Point2f pt1 = keypoints1[i1].pt;
    Point2f pt2 = keypoints2[i2].pt;
    pt2.x = pt2.x + offset_x;

    line (newimage, pt1, pt2, Scalar(255,0,0), thickness, lineType, shift);
  }

  return newimage;
}

/* This function performs a ratio test:
  - clear matches for which NN ratio is > than threshold
  - return the number of removed points
*/
int ratioTest(vector<vector<cv::DMatch> > &matches, double ratio)
{
  int removed = 0;
  for (vector <vector<cv::DMatch>>::iterator matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
  {
    // if 2 NN have been identified
    if (matchIterator->size() > 1)
    {
      // check distance ratio
      if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio)
      {
        matchIterator->clear(); // remove match
        removed++;
      }
    }

    // if does not have 2 neighbours, then remove match
    else 
    {
      matchIterator->clear(); // remove match
      removed++;
    }
  }

  return removed;
}

void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches)
{
/*	=========================================================================================
    for all matches image 1 -> image 2
   	========================================================================================= */
	for (vector<vector<cv::DMatch> >:: const_iterator matchIterator1= matches1.begin(); matchIterator1!= matches1.end(); ++matchIterator1) 
	{
   	// ignore deleted matches
  	if (matchIterator1->size() < 2) continue;

/* 		=================================================================================
     	for all matches image 2 -> image 1
   		================================================================================= */

		for (std::vector<std::vector<cv::DMatch> >:: const_iterator matchIterator2= matches2.begin();
		matchIterator2!= matches2.end(); ++matchIterator2) 
		{
			// ignore deleted matches
			if (matchIterator2->size() < 2) continue;
/* 			=========================================================================
     		Symmetry test
   			========================================================================= */
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) 
			{
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

Mat ransacTest(const vector<DMatch>& matches,const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches)
{

/*     	=========================================================================================
        Convert keypoints1 and keypoints2 into Point2f
       	========================================================================================= */

	vector<cv::Point2f> points1, points2;

	Mat fundemental;

	for (vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it) 
	{

		double x= keypoints1[it->queryIdx].pt.x;
		double y= keypoints1[it->queryIdx].pt.y;
		points1.push_back(Point2f(x,y));

		x= keypoints2[it->trainIdx].pt.x;
		y= keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x,y));
	}

/*     	=========================================================================================
        Compute fundamental matrix using RANSAC
       	========================================================================================= */

	vector<uchar> inliers(points1.size(),0);

	double confidence = 0.99;
	double distance = 3.0;
	int refineF = 1;

	if (points1.size()>0&&points2.size()>0)
	{

		Mat fundemental= cv::findFundamentalMat(
                        cv::Mat(points1),cv::Mat(points2), // matching points
                        inliers,       // match status (inlier or outlier)
                        CV_FM_RANSAC, // RANSAC method
                        distance,      // distance to epipolar line
                        confidence); // confidence probability

		// extract the surviving (inliers) matches

		vector<uchar>::const_iterator itIn= inliers.begin();
		vector<cv::DMatch>::const_iterator itM= matches.begin();

		for ( ;itIn!= inliers.end(); ++itIn, ++itM) 
		{
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}

		if (refineF) {

			// The F matrix will be recomputed with all accepted matches
			// Convert keypoints into Point2f for final F computation

			points1.clear();
			points2.clear();

			for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();it!= outMatches.end(); ++it) 
			{
				double x= keypoints1[it->queryIdx].pt.x;
				double y= keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x,y));

				x= keypoints2[it->trainIdx].pt.x;
				y= keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x,y));
			}

			// Compute 8-point F from all accepted matches

			if (points1.size()>0&&points2.size()>0){

				fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), // matches
						CV_FM_8POINT); // 8-point method

			}
		}
	}

	return fundemental;

}

