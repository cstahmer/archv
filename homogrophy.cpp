/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 *
 * file: homogrophy.cpp
 *
 * Contains a class/function that takes two images and SURF
 * feature point extraction parameters and returns an image
 * that draws feature point matches between image one and
 * image two.
 *
 * Exmample: ./homography ./images/31642-10.jpg ./images/31710-30.jpg -minhessian 1000 -octaves 15 -octavelayers 10
 *
 *
 * Copyright (C) 2012 Carl Stahmer (cstahmer@gmail.com)
 * Early Modern Center, University of California, Santa Barbara
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Creative Commons licence, version 3.
 *
 * See http://creativecommons.org/licenses/by/3.0/legalcode for the
 * complete licence.
 *
 * This program is distributed WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "helper.cpp"

using namespace cv;
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;

//Helper helper;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{

	Helper helper;
	string event;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;




  Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	//---Initialize various objects and parameters with base values
	int dictionarySize = 8000; // originally set to 1500
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;;


	int intSurfMinHession = SURF_MIN_HESSIAN;
	int intSurfOctaves = SURF_OCTAVES;
	int intSurfOctaveLayers = SURF_OCTAVE_LAYERS;
	bool blnSaveFeaturePointImages = SAVE_FEATURE_POINT_IMAGES;
	string blnFeaturePointImagesOutDir = FEATURE_POINT_IMAGES_OUT_DIR;
	int intTermCritMaxCount = TERM_CRIT_MAX_COUNT;
	double dblTermCritEpsilon = TERM_CRIT_EPSILON;
	string filename = "./homography.jpg";

  for (int i = 3; i < argc; i++) {
  	string arument = argv[i];
      if (arument == "-back") {
      	runInBackground = true;
      }
      if (arument == "-log") {
      	writelog = true;
      }

      if (arument == "-minhessian") {
      	intSurfMinHession = atoi(argv[i + 1]);
      }
      if (arument == "-octaves") {
      	intSurfOctaves = atoi(argv[i + 1]);
      }
      if (arument == "-octavelayers") {
      	intSurfOctaveLayers = atoi(argv[i + 1]);
      }
      if (arument == "-sfile") {
      	filename = atoi(argv[i + 1]);
      }
      if (arument == "-help") {
          cout << "Usage is <image 1> <image2>" <<endl << "    -sfile <output filename>" <<endl <<"   -back <flag to run in background>" <<endl << "    -log <flag to run in log mode>" <<endl << "    -minhessian<[minimum pixil blur>" <<endl << "    -octave <the amount to reduce at each octave step>" <<endl << "    -octavelayers <how many layers to the octave pyramid>";
          exit(0);
      }
  }

  event = "Starting homogrophy execuatable.";
  helper.logEvent(event, 2, runInBackground, writelog);
  string strMinHessian = static_cast<ostringstream*>( &(ostringstream() << (intSurfMinHession)) )->str();
  event = "SURF Min Hessian: " + strMinHessian;
  helper.logEvent(event, 2, runInBackground, writelog);
  string strOctaves = static_cast<ostringstream*>( &(ostringstream() << (intSurfOctaves)) )->str();
  event = "SURF Octaves: " + strOctaves;
  helper.logEvent(event, 2, runInBackground, writelog);
  string strOctaveLayers = static_cast<ostringstream*>( &(ostringstream() << (intSurfOctaveLayers)) )->str();
  event = "SURF Octave Layers: " + strOctaveLayers;
  helper.logEvent(event, 2, runInBackground, writelog);


  //-- Step 1: Detect the keypoints using SURF Detector
  //int minHessian = 400;


  //SurfFeatureDetector detector( minHessian );
  SurfFeatureDetector detector(intSurfMinHession, intSurfOctaves, intSurfOctaveLayers);

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;
  //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
  //Ptr<SurfDescriptorExtractor extractor = new SurfDescriptorExtractor();

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  int gmfound = 0;
  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
    { 
	  good_matches.push_back( matches[i]); 
	  gmfound++;
    }
  }
  
  //printf("-- Good Matches Found : %f \n", gmfound );

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


  //-- Localize the object from img_1 in img_2
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);


  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  Point2f offset( (float)img_object.cols, 0);
  line( img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar( 0, 255, 0), 4 );
  


  //--save detectyed images;
  if( !imwrite( filename, img_matches ) ) {
  	event = "Error Saving " + filename + ".";
  	//helper.logEvent(event, 0, runInBackground, writelog);
  } else {
  	event = filename + " Successfully saved.";
  	//helper.logEvent(event, 4, runInBackground, writelog);
  }

  cout << event << endl;
  


  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << "./homography -help" << std::endl; }
