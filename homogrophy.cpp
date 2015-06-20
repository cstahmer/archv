/**
 * @file SURF_Homography
 * @brief SURF detector + descriptor + FLANN Matcher + FindHomography
 * @author A. Huaman
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

  if( argc != 3 )
  { readme(); return -1; }

  Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	//---Initialize various objects and parameters with base values
	int dictionarySize = 8000; // originally set to 1500
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	string trainingDirectory = TRAINING_DIR;
	string dictionaryFileName = "dictionary";


	int intSurfMinHession = SURF_MIN_HESSIAN;
	int intSurfOctaves = SURF_OCTAVES;
	int intSurfOctaveLayers = SURF_OCTAVE_LAYERS;
	bool blnSaveFeaturePointImages = SAVE_FEATURE_POINT_IMAGES;
	string blnFeaturePointImagesOutDir = FEATURE_POINT_IMAGES_OUT_DIR;
	int intTermCritMaxCount = TERM_CRIT_MAX_COUNT;
	double dblTermCritEpsilon = TERM_CRIT_EPSILON;

	double dblKeypointSizeFilter = KEYPOINT_SIZE_FILTER;
	double dblKeypointResponseFilter = KEYPOINT_RESPONSE_FILTER;


  for (int i = 3; i < argc; i++) {
  	string arument = argv[i];
      if (arument == "-d") {
      	trainingDirectory = argv[i + 1];
      }
      if (arument == "-n") {
      	dictionaryFileName = argv[i + 1];
      }
      if (arument == "-s") {
      	dictionarySize = atoi(argv[i + 1]);
      }
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
      if (arument == "-images") {
      	blnSaveFeaturePointImages = true;
      }
      if (arument == "-imageoutput") {
      	blnFeaturePointImagesOutDir = argv[i + 1];
      }

      if (arument == "-tcmax") {
      	intTermCritMaxCount = atoi(argv[i + 1]);
      }

      if (arument == "-tcepsilon") {
      	dblTermCritEpsilon = atof(argv[i + 1]);
      }

      if (arument == "-sizefilter") {
      	dblKeypointSizeFilter = atof(argv[i + 1]);
      }

      if (arument == "-responsefilter") {
      	dblKeypointResponseFilter = atof(argv[i + 1]);
      }

      if (arument == "-help") {
          cout << "Usage is -d <dirctory of training files> -n <name of dictionary output file and structure name> -s <size of dictionary> -back [flag to run in backbround mode] -log [flag to run in log mode]"<<endl;
          exit(0);
      }
  }

  event = "Starting makeDictionary execuatable.";
  helper.logEvent(event, 2, runInBackground, writelog);
  event = "Training Directory: " + trainingDirectory;
  helper.logEvent(event, 2, runInBackground, writelog);
  event = "Filename to use when saving dictionary: " + fullDictionaryFileName;
  helper.logEvent(event, 2, runInBackground, writelog);
  string strDictSize = static_cast<ostringstream*>( &(ostringstream() << (dictionarySize)) )->str();
  event = "Size of dictionary: " + strDictSize;
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

  TermCriteria tc(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, intTermCritMaxCount, dblTermCritEpsilon);

  //SurfFeatureDetector detector( minHessian );
  SurfFeatureDetector detector(intSurfMinHession, intSurfOctaves, intSurfOctaveLayers);

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  //SurfDescriptorExtractor extractor;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
  Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  //FlannBasedMatcher matcher;
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
  
  printf("-- Good Matches Found : %f \n", gmfound );

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


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
  
  string filename = "/web/sites/beeb/finished-woodcut-images-res/homography.jpg";

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
{ std::cout << " Usage: ./homography <img1> <img2>" << std::endl; }
