/*

	Arthur Koehl			showkeypoints.cpp		February 2016

	this program:
		[1] reads in parameters for surf feature detection from the command line
		[2] uses opencv's Feature Detection class to detect all the keypoints
		[3] uses opencv's Circle function to draw all the keypoints over the image
		[4] saves the image with drawn keypoints into file specified in step 1

	mainly use this program to test parameters and to see the location and intensity of keypoints

*/

#include <iostream>
#include <cstdlib>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

void read_instructions (int argc, char **argv, string *input, string *output, int *minh, int *octaves, int *layers); 
void draw_keypoints (Mat &outimage, vector<KeyPoint> &keypoints);

int main(int argc, char **argv)
{
/* =====================================================================================
	if program executed without commands or '-h' or '-help' flag then run usage
   ===================================================================================== */
  if (argc < 2) 
  {
    cout << "./a.out -i input -o output -minh # -octaves # - layers #" << endl;
    return -1;
  }
  string checker = argv[1];
  if (checker == "-h" || checker == "-help")
  {
    cout << "./a.out -i input -o output -minh # -octaves # - layers #" << endl;
    return -1;
  }
  
/* =====================================================================================
	variables for surf parameters, file names, image, and vector of keypoints
   ===================================================================================== */
  int minh, octaves, layers;
  string input, output;
  vector <KeyPoint> keypoints;
  Mat image; 

/* =====================================================================================
	parse command line into variables above and read in the image into Mat image
   ===================================================================================== */
  read_instructions(argc, argv, &input, &output, &minh, &octaves, &layers);
  image = imread (input);

/* =====================================================================================
	create openCV's SurfFreatureDetector and then run detect() function
   ===================================================================================== */
  SurfFeatureDetector detector (minh, octaves, layers);
  detector.detect (image, keypoints);
  cout << "found " << keypoints.size() << " keypoints!" << endl;

/* =====================================================================================
	call drawkeypoints and write to the output image
  ===================================================================================== */
  draw_keypoints (image, keypoints);
  imwrite (output, image);

  return 0;
}  



void read_instructions(int argc, char **argv, string *input, string *output,  int *minh, int *octaves, int *layers) 
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
  }
  return;
}

void draw_keypoints (Mat &image, vector<KeyPoint> &keypoints)
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
    circle (image, keypoints[i].pt, radius, Scalar (255,0,0), thickness, lineType, shift); 
  }
  return;
}

  

  
