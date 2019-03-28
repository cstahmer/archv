/* ============================================================================================
  showKeypoints.cpp                Version 3           last update: 03/28/2019                  

	this program:
		[1] reads in parameters for surf detection, input image and output image from command line
          if no options for surf specified will use those set in code in variable declaration
		[2] uses opencv's Feature Detection class to detect all the keypoints
		[3] filters the keypoints based on a sizemin and responsemin specified 
		[4] uses opencv's Circle function to draw all the keypoints over the image
		[5] saves the image with drawn keypoints into file specified in step 1

	mainly use this program to test parameters and to see the location and intensity of keypoints
	
	This file is part of the Arch-V Platform -- https://github.com/cstahmer/archv

	Copyright 2012 by Carl G. Stahmer -- http://www.carlstahmer.com
	
	Arch-V was originally created by Carl G. Stahmer through the generous support of 
	the National Endowment for the Humanities.  Subsequent development was performed 
	by Carl G. Stahmer (http://www.carlstahmer.com) and Arthur Koehl (avkoehl@ucdavis.edu) 
	at the Digital Scholars Lab at the the University of California Davis, Univeristy 
	Library (http://ds.lib.ucdavis.edu/). Documentation authored by Henry Le 
	(hutle@ucdavis.edu).

	Arch-V is licensed under a Creative Commons Attribution 4.0 International
	License (https://creativecommons.org/licenses/by/4.0/legalcode).

	You are FREE to SHARE (copy and redistribute the material in any medium or format) 
	and ADAPT (remix, transform, and build upon the material for any purpose, even 
	commercially) WITH THE FOLLOWING RESTRICTIONS:

	1. 	You must credit Carl G. Stahmer (http://www.carlstahmer.com) and Arthur Koehl 
		(avkoehl@ucdavis.edu) as the original developers of this software.
		
	2. 	You must credit the National Endowment for the Humanities and Univeristy of 
		California, Davis Univeristy Library as having supported the original development 
		of the software.
		
	3. 	You must provide a copyright notice.
	
	4. 	You must provide a link to the license 
		(https://creativecommons.org/licenses/by/4.0/legalcode).
		
	5. 	You must indicate if and what changes you made to the software.
	
	6. 	You must provide a link to the original software at
		https://github.com/cstahmer/archv](https://github.com/cstahmer/archv

 ============================================================================================ */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int usage ();
void read_flags (int argc, char **argv, string *input, string *output, string *param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin);
void read_surfparams (string param, int *min, int *octaves, int *layers, int *sizemin, double *responsemin);
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
  string param = "";
  vector <KeyPoint> keypoints;
  Mat image;
  Mat outimage; 

/* =====================================================================================
	parse command line into variables above and read in the image into Mat image
   ===================================================================================== */
  read_flags(argc, argv, &input, &output, &param, &minh, &octaves, &layers, &sizemin, &responsemin);
  if (param != "")
    read_surfparams (param, &minh, &octaves, &layers, &sizemin, &responsemin);
  image = imread (input);

/* =====================================================================================
	create openCV's SurfFreatureDetector and then run detect() function
   ===================================================================================== */
  SurfFeatureDetector detector (minh, octaves, layers);
  detector.detect (image, keypoints);
  int original = keypoints.size();
  filter_keypoints (keypoints, sizemin, responsemin);

/* =====================================================================================
	call drawkeypoints from opencv and write to the output image
  ===================================================================================== */
  drawKeypoints (image, keypoints, outimage, Scalar (155,0,0), 4);
  imwrite (output, outimage);

  return 0;
}  


int usage ()
{
  cout << "./a.out -i input -o output -p paramfilepath " << endl;
  cout << "otherwise, without param file:" << endl;
  cout << "./a.out -i input -o output -h # -oct # -l # -s # -r #" << endl;
  return -1;
}

void read_flags (int argc, char **argv, string *input, string *output, string *param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin)
{
  string parser;
  for (int i = 0; i < argc; i++)
  {
    parser = argv[i];
    if (parser == "-i")
      *input = argv[i + 1];
    if (parser == "-o")
      *output = argv[i+1];
    if (parser == "-p")
      *param = argv[i+1];

    if (parser == "-h")
      *minh = atoi(argv[i+1]);
    if (parser == "-oct")
      *octaves = atoi(argv[i+1]);
    if (parser == "-l")
      *layers = atoi(argv[i+1]);
    if (parser == "-s")
      *sizemin = atoi(argv[i+1]);
    if (parser == "-r")
      *responsemin = atoi(argv[i+1]);
  }
}
 

/* ===============================================================================================
   Procedure to read parameters for SURF from the parameter file
   =============================================================================================== */
void read_surfparams(string param, int *minHessian, int *octaves, int *octaveLayers, int *SizeMin, double *RespMin)
{
  ifstream inFile;
  inFile.open(param.c_str());
	string record;
	stringstream ss;

	while ( !inFile.eof () ) {    
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



  
  
