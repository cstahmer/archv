/*
	Arthur Koehl			calibrate.cpp		February 2016

	this program:
		[1] reads in parameters from a file for multiple runs 
		[2] for each set of parameters:
			(a) uses opencv Feature Detection class to detect keypoints
			(b) filters the keypoints based on size and response time 
			(c) draws the keypoints over the image using opencv circle function
		[3] combines all the images into one image and saves in file

	mainly use this program to see what happens when you change one parameter (ie hessian) and leave others as control

*/
	
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>
#include <fstream>
#include <stdarg.h>

using namespace cv;
using namespace std;

void read_flags (int argc, char **argv, string *input, string *output, string *param, double *scale);
void read_parameters (string param, int minhessian[], int octaves[], int layers[], int size[], double response[],  int *n);
void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin);
void draw_keypoints (Mat &output, vector <KeyPoint> &keypoints);
Mat combine_images (int n, Mat images[], string titles[]);

int main(int argc, char** argv)
{
/* ===============================================================================================
   Show usage if needed
   =============================================================================================== */
  if (argc < 2)
  {
    cout << "./calibrate.exe -i input -o output -p paramfile -s scale" << endl;
    return -1;
  }
  string checker = argv[1];
  if (checker == "-h" || checker == "-help")
  {
    cout << "./calibrate.exe -i input -o output -p paramfile -s scale" << endl;
    return -1;
  }
   
/* ===============================================================================================
   read in the file names as well as the scale.
   Names of the three files for the run: input image, output image, param file, and scale (double)
   =============================================================================================== */
  string input, output, param;
  double scale;
  read_flags (argc, argv, &input, &output, &param, &scale);

/* ===============================================================================================
   Read in parameters for detecting key points in the image: multiple conditions are considered
   (maximum 10 conditions)
   =============================================================================================== */
  int minhessian[10], octaves[10], layers[10], size[10];
  double response[10];
  int n;
  read_parameters (param, minhessian, octaves, layers, size, response, &n);
   
/* ===============================================================================================
   Read in image and store in structure img, make keypoints vector
   =============================================================================================== */
  Mat image = imread (input);
  vector <KeyPoint> keypoints[n];
  string titles[n];
  stringstream str[n];
  int interpolation = INTER_LINEAR;
  Mat drawImg[n];
  Mat scaledImg[n];
  for (int i = 0; i < n; i++)
    image.copyTo (drawImg[i]);

/* ===============================================================================================
   Loop over all conditions (minHessian, octaves, and octaveLayers
   =============================================================================================== */
  for (int i = 0; i < n; i++)
  {
/* 	==========================================================================================
	Create a SURF detector based on current parameters
   	========================================================================================== */
    FeatureDetector *detector = new SurfFeatureDetector (minhessian[i], octaves[i], layers[i]);

/* 	==========================================================================================
	Detect Keypoints on the image based on this detector and then filter them
   	========================================================================================== */
    detector->detect (image, keypoints[i]);
    cout << "keypoints: " << keypoints[i].size() << endl;
    filter_keypoints (keypoints[i], size[i], response[i]);
    cout << "after filter: " << keypoints[i].size() << endl;

/* 	==========================================================================================
	Overlay keypoints on the original image (put into a new image)
   	========================================================================================== */
    draw_keypoints (drawImg[i], keypoints[i]);

/* 	==========================================================================================
	Scale image for display
   	========================================================================================== */
    resize (drawImg[i], scaledImg[i], Size(), scale, scale, interpolation);

/* 	==========================================================================================
	Define a title that stores the parameter values
   	========================================================================================== */
    str[i] << "Hessian = " << minhessian[i] << "Octaves = " << octaves[i] << "Layers = " << layers[i];
    getline (str[i], titles[i]);

/* 	==========================================================================================
	Destroy detector
   	========================================================================================== */
    delete detector;
  }

/* ===============================================================================================
   Now combine all images into a single, large one
   =============================================================================================== */
    Mat combined = combine_images (n, scaledImg, titles);  

/* ===============================================================================================
   save image into a file
   =============================================================================================== */
    imwrite (output, combined);

    return 0;
}


void read_flags (int argc, char **argv, string *input, string *output, string *param, double *scale)
{
  string parser;
  for (int i = 0; i < argc; i++)
  {
    parser = argv[i];
    if (parser == "-i")
      *input = argv[i+1];
    if (parser == "-o")
      *output = argv[i+1];
    if (parser == "-p")
      *param = argv[i+1];
    if (parser == "-s")
      *scale = atof(argv[i+1]);
  }
  return;
}

void read_parameters (string param, int minhessian[], int octaves[], int layers[], int size[], double response[], int *n)
{
  ifstream infile (param);
  int i = 0;

  while (infile >> minhessian[i] >> octaves[i] >> layers[i] >> size[i] >> response[i])
  {
    i = i + 1;
  }

  *n = i;
  return;
} 

void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin)
{
  vector<KeyPoint> keypoints2;
  int npoints = keypoints.size();
  int size;
  double response;

/* ===================================================================================
   filter based upoon keypoint size
====================================================================================== */
  for (int i = 0; i < npoints; i++)
  {
    size = keypoints[i].size;
    if (size > SizeMin)
      keypoints2.push_back(keypoints[i]);
  }
  keypoints.clear();
  keypoints = keypoints2;
  npoints = keypoints.size();
  keypoints2.clear();

/* ===================================================================================
   filter based upoon keypoint response time
====================================================================================== */
  for (int i = 0; i < npoints; i++)
  {
    response = keypoints[i].response;
    if (response > RespMin)
      keypoints2.push_back(keypoints[i]);
  }
  keypoints.clear();
  keypoints = keypoints2;
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
  

Mat combine_images (int n, Mat images[], string titles[])
{
	Mat NewImage;
	Mat myimg;
	string title;

	int i;
	int offset_x, offset_y;
	int width,height;
	int spacer_x,spacer_y;
	int pos_x,pos_y;

/* ===============================================================================================
	row_max : (maximum) number of images per row
	col_max : (maximum) number of images per column
   =============================================================================================== */

	int img_row, img_col;

/* ===============================================================================================
	Define the geometry of the new image: number of rows and columns,
	as well as size of each individual image
   =============================================================================================== */

	if (n == 1) {
		img_row = 1;
		img_col = 1;
	}
	else if (n == 2) {
		img_row = 2;
		img_col = 1;
	}
	else if (n == 3 || n == 4) {
		img_row = 2;
		img_col = 2;
	}
	else if (n == 5 || n == 6) {
		img_row = 3;
		img_col = 2;
	}
	else if (n == 7|| n == 8) {
		img_row = 4;
		img_col = 2;
	}
	else {
		img_row = 4;
		img_col = 3;
	}

/* ===============================================================================================
	spacer_x and spacer_y are the spaces (# of pixels) left empty between images
	pos_x and pos_y is the position of the text for the first title
   =============================================================================================== */

	spacer_x = 20;
	spacer_y = 20;

	pos_x   = 2*spacer_x;
	pos_y   = spacer_y/2;

/* ===============================================================================================
	Get information from first image: size (width and height), and title
   =============================================================================================== */

	myimg = images[0];
	title = titles[0];

	width = myimg.cols;
	height = myimg.rows;

/* ===============================================================================================
	Create new image and size it so that it can contain all the images
	Generate image with white background
   =============================================================================================== */

	int size_x = spacer_x*(img_row+1) + width*img_row;
	int size_y = spacer_y*(img_col+1) + height*img_col;

       NewImage = Mat( cvSize(size_x,size_y), CV_8UC3, Scalar(255,255,255));

/* ===============================================================================================
	Now place all images into the New Image
   =============================================================================================== */

	Mat TargetROI;


	offset_x = spacer_x;
	offset_y = spacer_y;

	for (i = 0; i < n; i++)
	{
		myimg = images[i];
		title = titles[i];

/* 		==================================================================================
		- Set a rectangle where we want to put the image;
		- Define a "Region Of Interest" corresponding to this rectangle
		- Copy the image in that rectangle
   		================================================================================== */
 
		Rect ROI(offset_x, offset_y, width, height);
		TargetROI = NewImage(ROI);
		myimg.copyTo(TargetROI);

/* 		==================================================================================
		Write a title above the image: indicate the minHessian, octaves, and octaveLayers
		that were used for that image
		Title is written in black
   		================================================================================== */

		putText(NewImage, title, Point(pos_x,pos_y), CV_FONT_HERSHEY_PLAIN, 0.7, Scalar(0,0,0));

/* 		==================================================================================
		Reset the offsets in X and Y: move along rows. If the current image was the last
		in the row, switch to new row
   		================================================================================== */

		offset_x += width + spacer_x;
		pos_x    += width + spacer_x;

		if((i+1) % img_row == 0) {
			offset_x = spacer_x;
			offset_y += spacer_y + height;
			pos_x = 2*spacer_x;
			pos_y += height +spacer_y;
		}
	}

/* ===============================================================================================
	Now return New Image to main program
   =============================================================================================== */

	return NewImage;

}
