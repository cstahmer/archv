/* ============================================================================================
  showKeypoints.cpp                Version 2           10/15/2016                  Arthur Koehl

	this program:
		[1] reads in parameters for surf detection, input image and output image from command line
          if no options for surf specified will use those set in code in variable declaration
		[2] uses opencv's Feature Detection class to detect all the keypoints
		[3] filters the keypoints based on a sizemin and responsemin specified 
		[4] uses opencv's Circle function to draw all the keypoints over the image
		[5] saves the image with drawn keypoints into file specified in step 1

	mainly use this program to test parameters and to see the location and intensity of keypoints

 ============================================================================================ */

#include <iostream>
#include <cstdlib>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int usage ();
void read_instructions (int argc, char **argv, string *input, string *output, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin);
void filter_keypoints (vector <KeyPoint> &keypoints, int sizemin, double responsemin);

int main(int argc, char **argv)
{
/* =====================================================================================
	if program executed without commands or '-h' or '-help' flag then run usage
   ===================================================================================== */
  if (argc < 2) 
    return usage();
  string checker = argv[1];
  if (checker == "-h" || checker == "-help") 
    return usage();
  
/* =====================================================================================
	variables for surf parameters, file names, image, and vector of keypoints
   ===================================================================================== */
  int minh = 2000;
  int octaves = 8;
  int layers = 8;
  int sizemin = 50;
  double responsemin = 100;
  string input, output;
  vector <KeyPoint> keypoints;
  Mat image;
  Mat outimage; 

/* =====================================================================================
	parse command line into variables above and read in the image into Mat image
   ===================================================================================== */
  read_instructions(argc, argv, &input, &output, &minh, &octaves, &layers, &sizemin, &responsemin);
  image = imread (input);

/* =====================================================================================
	create openCV's SurfFreatureDetector and then run detect() function
   ===================================================================================== */
  SurfFeatureDetector detector (minh, octaves, layers);
  detector.detect (image, keypoints);
  int original = keypoints.size();
  filter_keypoints (keypoints, sizemin, responsemin);
  cout << "Keypoints: " << original << " After filter: " << keypoints.size() << endl;

/* =====================================================================================
	call drawkeypoints from opencv and write to the output image
  ===================================================================================== */
  drawKeypoints (image, keypoints, outimage, Scalar (155,0,0), 4);
  imwrite (output, outimage);

  return 0;
}  


int usage ()
{
  cout << "./a.out -i input -o output -minh # -octaves # - layers # -sizemin # -responsemin #" << endl;
  return -1;
}

void read_instructions(int argc, char **argv, string *input, string *output, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin) 
{
  string parser;
  for (int i = 0; i < argc; i++)
  {
    parser = argv[i];
    if (parser == "-i")
      *input = argv[i + 1];
    if (parser == "-o")
      *output = argv[i+1];
    if (parser == "-minh")
      *minh = atoi(argv[i+1]);
    if (parser == "-octaves")
      *octaves = atoi(argv[i+1]);
    if (parser == "-layers")
      *layers = atoi(argv[i+1]);
    if (parser == "-sizemin")
      *sizemin = atoi(argv[i+1]);
    if (parser == "-responsemin")
      *responsemin = atoi(argv[i+1]);
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



  
  
