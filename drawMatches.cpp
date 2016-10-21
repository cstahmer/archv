/* ============================================================================================
  drawMatches.cpp                Version 2             10/15/2016                  Arthur Koehl

  This program takes two images (generally ones that are known to be similar) processes them
  by finding their keypoints and descriptors for the keypoints. Then it finds the matching
  keypoints and filters them using homography (ratio test, symmetry test, ransac test).
  Lastly it draws the keypoints and a line connecting its matching one in the other image.

 ============================================================================================ */

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

int usage();
void read_flags (int argc, char** argv, string *imgfile1, string *imgfile2, string *output, string *param);
void read_surfparams(string param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin);

int ratioTest(vector<vector<cv::DMatch> > &matches, double ratio);
void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches);
Mat ransacTest(const vector<cv::DMatch>& matches,const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& outMatches);

void filter_keypoints (vector <KeyPoint> &keypoints, int sizemin, double responsemin);
void show_keypoints (vector<KeyPoint>& keypoints, Mat& drawImg);
Mat DrawMatch(Mat& image1, vector<KeyPoint>& keypoints1, Mat& image2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches1to2);

int main(int argc, char** argv)
{
/* ===============================================================================================
   Show usage if needed
   =============================================================================================== */
  if ( argc < 2 )
    return usage();

  string input = argv[1];
  if ( input == "-h" || input == "-help" )
    return usage();

/* ===============================================================================================
    (1) Initialize all variables and surf parameters (2) parse command line (3) read in parameters 
   =============================================================================================== */
  string imgfile1, imgfile2, output, param;

  int minh= 2000 ;
  int octaves = 8;
  int layers= 8;
  int sizemin = 50;
  double responsemin = 100;
  double scale = 1;
  double ratio = 0.8;

  read_flags (argc, argv, &imgfile1, &imgfile2, &output, &param);

  read_surfparams (param, &minh, &octaves, &layers, &sizemin, &responsemin);

/* ===============================================================================================
   Create all structures that are needed to process the images:
        - use SURF for key points detection and feature extraction
	- use Brute Force Matching to match descriptors between two images
   =============================================================================================== */
  SurfFeatureDetector detector (minh, octaves, layers);  // Define SURF detector

  Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor(); // Feature extractor from keypoints


  vector<KeyPoint> keypoints1;                                        // Vector of keypoints for image 1
  vector<KeyPoint> keypoints2;                                        // Vector of keypoints for image 2

  BFMatcher matcher;							 // Brute Force Matcher

  vector <vector <DMatch> > matches1;	 // Matches when comparing image 1 -> image 2
  vector <vector <DMatch> > matches2;	 // Matches when comparing image 2 -> image 1
  vector <DMatch> sym_matches;				 // Matches after symmetry filter
  vector <DMatch> matches;						 // Matches after RANSAC filter

/* ===============================================================================================
   Process images:
	- detect keypoints
	- filter keypoints
   =============================================================================================== */
  Mat img1;
  Mat img2;

  img1 = imread(imgfile1);
  img2 = imread(imgfile2);

  detector.detect( img1, keypoints1 );
  detector.detect( img2, keypoints2 );

  filter_keypoints (keypoints1, sizemin, responsemin);
  filter_keypoints (keypoints2, sizemin, responsemin);

  cout << "Number of keypoints 1 : " << keypoints1.size() << endl;
  cout << "Number of keypoints 2 : " << keypoints2.size() <<endl;

/* ===============================================================================================
   Generate descriptors from key points
   =============================================================================================== */
  Mat descriptors1, descriptors2;
  extractor->compute(img1, keypoints1, descriptors1);
  extractor->compute(img2, keypoints2, descriptors2);

/* ===============================================================================================
   Find matches based on descriptors: 
	- first from img1 to img2 (with 2 NN), then from img2 to img1
	- filter based on ratio test
	- filter for symmetry
	- filter by RANSAC
   =============================================================================================== */
  matcher.knnMatch(descriptors1,descriptors2,matches1,2);
  matcher.knnMatch(descriptors2,descriptors1,matches2,2);

  int removed= ratioTest(matches1,ratio);
  removed= ratioTest(matches2,ratio);

  symmetryTest(matches1,matches2,sym_matches);

  Mat fundamental= ransacTest(sym_matches,keypoints1,keypoints2, matches);

/* ===============================================================================================
   Only keep "good" keypoints (i.e. those that correspond to good matches
   =============================================================================================== */
  vector<KeyPoint> keypt1;
  vector<KeyPoint> keypt2;

  int count = 0;
  for (int m = 0; m < matches.size(); m++)
  {
	  int i1 = matches[m].queryIdx;
	  int i2 = matches[m].trainIdx;

	  keypt1.push_back(keypoints1[i1]);
	  keypt2.push_back(keypoints2[i2]);

	  count++;
  }

  cout << "number of remaining matches after homography: " << count << endl;

/* ===============================================================================================
   Generate image with keypoints and good matches
   =============================================================================================== */
  Mat imgkpts1;
  Mat imgkpts2;

  img1.copyTo(imgkpts1);
  img2.copyTo(imgkpts2);

  //call drawkeypoints from opencv
  //show_keypoints (keypt1, imgkpts1); 
  drawKeypoints (img1, keypt1, imgkpts1, Scalar (0, 0, 155));
  //show_keypoints (keypt2, imgkpts2); 
  drawKeypoints (img2, keypt2, imgkpts2, Scalar (0, 0, 155));

  Mat img_matches;
  img_matches = DrawMatch (imgkpts1, keypoints1, imgkpts2, keypoints2, matches);

/* ===============================================================================================
   Rescale matched image
   =============================================================================================== */
  Mat img_scaled;
  int interpolation=INTER_LINEAR;
  resize (img_matches, img_scaled, Size(), scale, scale, interpolation);

/* ===============================================================================================
   Save results
   =============================================================================================== */
  imwrite( output, img_scaled);

  return 0;
}



/* ===============================================================================================
   Usage
   =============================================================================================== */
int usage()
{
    cout << "\n\n" <<endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "=                                      DrawMatches                                             ="  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "=     This program reads in two iamges, detects their keypoints, finds the matches,            ="  <<  endl;
    cout << "     " << "=     uses homography (ratio test, symmetry test, RANSAC) to filter the matches,               ="  <<  endl;
    cout << "     " << "=     and then displays the remaining matches                                                  ="  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "=     Usage is:                                                                                ="  <<  endl;
    cout << "     " << "=                 ./drawMatches.exe                                                            ="  <<  endl;
    cout << "     " << "=                                 -i1  <path to image file 1>                                  ="  <<  endl;
    cout << "     " << "=                                 -i2  <path to image file 2>                                  ="  <<  endl;
    cout << "     " << "=                                 -o  <path to output image file>                              ="  <<  endl;
    cout << "     " << "=                                 -p  <path to param file for SURF>                            ="  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "\n\n" <<endl;

  return -1;
}



/* ===============================================================================================
   Procedure to read in flag values
   =============================================================================================== */
void read_flags (int argc, char** argv, string *imgfile1, string *imgfile2, string *output, string *param)
{
  string input;
  for(int i = 1; i < argc; i++)
  {
    input = argv[i];
    if (input == "-i1") 
      *imgfile1 = argv[i + 1];
    if (input == "-i2") 
      *imgfile2 = argv[i + 1];
    if (input == "-o") 
      *output = argv[i + 1];
    if (input == "-p") 
      *param = argv[i + 1];
  }
}



/* ===============================================================================================
   Procedure to read parameters for SURF
   =============================================================================================== */
void read_surfparams(string param, int *minHessian, int *octaves, int *octaveLayers, int *SizeMin, double *RespMin)
{
  ifstream inFile;
  inFile.open(param.c_str());
	string record;
	stringstream ss;

	while ( !inFile.eof () ) 
  {    
		getline(inFile,record);
		if (record.find("minHessian") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *minHessian;
			ss.str("");
			ss.clear();
		}
		if (record.find("octaves") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *octaves;
			ss.str("");
			ss.clear();
		}
		if (record.find("octaveLayers") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *octaveLayers;
			ss.str("");
			ss.clear();
		}
		if (record.find("min Size") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *SizeMin;
			ss.str("");
			ss.clear();
		}
		if (record.find("min Resp") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *RespMin;
			ss.str("");
			ss.clear();
		}
	}
}

/* ===============================================================================================
   Procedure to filter the keypoints from the keypoint vector by minimum size and response
   =============================================================================================== */
void filter_keypoints (vector <KeyPoint> &keypoints, int sizemin, double responsemin)
{
  vector <KeyPoint> temp;
  int npoints = keypoints.size();
  int size;
  double response;

  //filter based on size and response size
  for (int i = 0; i < npoints; i++)
  {
    size = keypoints[i].size;
    response = keypoints[i].response;
    if (size > sizemin && response > responsemin)
      temp.push_back(keypoints[i]);
  }

  keypoints.clear();
  keypoints = temp;

  return;
}



/* ===============================================================================================
   Procedure to draw positions of key points on image using circles
   =============================================================================================== */
void show_keypoints (vector<KeyPoint>& keypoints, Mat& drawImg)
{
	int thickness = 2;
	int lineType = 8;
	int shift = 0;
	int radius;

	int npoints=keypoints.size();

 	for(int i = 0; i < npoints; i++)
	{
		radius=keypoints[i].size/5;
		circle(drawImg,keypoints[i].pt,radius,Scalar(0,0,255),thickness,lineType,shift);
	}
}



/* ===============================================================================================
   This function combines multiple images into a single image
   =============================================================================================== */
Mat DrawMatch(Mat& image1, vector<KeyPoint>& keypoints1, Mat& image2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches1to2)
{
	Mat NewImage;

	int offset_x;
	int width,height;
	int spacer_x;

/* ===============================================================================================
	Get information from images: size (width and height)
   =============================================================================================== */
	spacer_x = 20;
	width = image1.cols + image2.cols + spacer_x;
	height = image1.rows;
	if(image2.rows > height) height = image2.rows;

/* ===============================================================================================
	Create new image and size it so that it can contain all the images
	Generate image with white background
   =============================================================================================== */
	NewImage = Mat( cvSize(width,height), CV_8UC3, Scalar(255,255,255));

/* ===============================================================================================
	Now place all images into the New Image
   =============================================================================================== */
	Mat TargetROI;
	offset_x = 0;

/* ==================================================================================
   	First put image on the left
   ================================================================================== */

/* ==================================================================================
	- Set a rectangle where we want to put the image;
	- Define a "Region Of Interest" corresponding to this rectangle
	- Copy the image in that rectangle
  ================================================================================== */
	Rect ROI(offset_x, 0, image1.cols, image1.rows);
	TargetROI = NewImage(ROI);
	image1.copyTo(TargetROI);

/* ==================================================================================
	Repeat for image representing the histogram, on the right
   ================================================================================== */
	offset_x += image1.cols + spacer_x;

	Rect ROI2(offset_x,0, image2.cols,image2.rows);
	TargetROI = NewImage(ROI2);
	image2.copyTo(TargetROI);

/* ==================================================================================
	Now add lines between keypoints
   ================================================================================== */
	int thickness = 2; //from 4
	int lineType = 8;
	int shift = 0;

	for (int m = 0; m < matches1to2.size(); m++)
	{
		int i1 = matches1to2[m].queryIdx;
		int i2 = matches1to2[m].trainIdx;
		CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
		CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));

		Point2f pt1=keypoints1[i1].pt;
		Point2f pt2=keypoints2[i2].pt;
		pt2.x += offset_x;

		line(NewImage,pt1,pt2,Scalar( 255, 0, 0),thickness,lineType,shift);
	}

/* ===============================================================================================
	Now return New Image to main program
   =============================================================================================== */
	return NewImage;
}

/* ===============================================================================================
   This function perform a ratio test: 
   - Clear matches for which NN ratio is > than threshold
   - return the number of removed points
   =============================================================================================== */
int ratioTest(vector<vector<cv::DMatch> > &matches, double ratio) 
{
	int removed=0;

	for (vector<vector<cv::DMatch> >::iterator
		matchIterator= matches.begin();
		matchIterator!= matches.end(); ++matchIterator) 
	{
    //if 2 NN have been identified
	  if (matchIterator->size() > 1)
    {
    // check distance ratio
   	 if ((*matchIterator)[0].distance/ (*matchIterator)[1].distance > ratio) 
     {
				matchIterator->clear(); // remove match
				removed++;
     }
    } 

    //does not have 2 neighbours, then remove match
    else 
    {
  	  matchIterator->clear(); // remove match
  	  removed++;
	  }
	}

	return removed;
}



/* ===============================================================================================
   This function perform a symmetry test:
   matches from 1 to 2 should match with matches from 2 to 1
   =============================================================================================== */
void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches) 
{
/*    	=========================================================================================
        for all matches image 1 -> image 2
       	========================================================================================= */
	for (vector<vector<cv::DMatch> >:: const_iterator matchIterator1= matches1.begin();
           matchIterator1!= matches1.end(); ++matchIterator1) 
	{
    // ignore deleted matches
  	if (matchIterator1->size() < 2) continue;
/*     		=================================================================================
        	for all matches image 2 -> image 1
       		================================================================================= */
		for (std::vector<std::vector<cv::DMatch> >:: const_iterator matchIterator2= matches2.begin();
		matchIterator2!= matches2.end(); ++matchIterator2) 
		{
			// ignore deleted matches
			if (matchIterator2->size() < 2) continue;

/*     			=========================================================================
        		Symmetry test
       			========================================================================= */
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
			(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) 
			{
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
				(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));

				break; // next match in image 1 -> image 2
			}
		}
	}
}



/* ===============================================================================================
   Identify good matches using RANSAC; return fundamental matrix
   =============================================================================================== */
Mat ransacTest(const vector<cv::DMatch>& matches,const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& outMatches) 
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
