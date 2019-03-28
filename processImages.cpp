/* ============================================================================================
  processImages.cpp                Version 3           Last Update: 03/28/2019             

  This program reads in an input directory contianing a set of images, processes them, 
  computes features and descriptors and outputs them to YAML format files in output directory
  
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
#include <dirent.h>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

int usage ();
void read_flags(int argc, char** argv, string *path2dir, string *path2outdir, string *param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin);
void read_surfparams (string param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin);
int get_filelist (string path, vector <string> &allfiles);
void filter_keypoints (vector <KeyPoint> &keypoints, int sizemin, double responsemin);

int main(int argc, char **argv)
{
/* ===============================================================================================
   Display usage if necessary
   =============================================================================================== */
  if (argc < 2)
    return usage();
  string argument = argv[1];
  if (argument == "-h" || argument == "-help")
    return usage();

/* ===============================================================================================
   (1) Initialize all variables and surf parameters (2) parse command line (3) read in parameters
   =============================================================================================== */
  string path2dir, path2outdir;
  string param = "";
  int minh = 2000, octaves = 5, layers = 5;
  int sizemin = 50;
  double responsemin = 100;
  string name, nameful;
  string extension = ".yml";

  read_flags (argc, argv, &path2dir, &path2outdir, &param, &minh, &octaves, &layers, &sizemin, &responsemin);

  if (param != "")
    read_surfparams (param, &minh, &octaves, &layers, &sizemin, &responsemin);

/* ===============================================================================================
   Create all structures needed for SURF key point detection and feature extraction
   =============================================================================================== */
  SurfFeatureDetector detector (minh, octaves, layers);
  Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
  vector <KeyPoint> keypoints;
  Mat descriptors;


/* ===============================================================================================
   Got to input directory, (1) get the file list (2) keep only image files (.jpg extension)
   =============================================================================================== */
  vector <string> allfiles;
  vector <string> files; //remaining images

  int error =  get_filelist (path2dir, allfiles);
  if ( error != 0)
  {
    cout << "no files in directory" << endl;
    return -1;
  }

  //filter for image files
  for (int i = 0; i < allfiles.size(); i++)
  {
    string filename = allfiles[i];
    if (filename.substr (filename.find_last_of(".") + 1) == "jpg")
      files.push_back (filename);
  }


/* ===============================================================================================
   For each image file: 
      (1) generate output file name
      (2) detect keypoints
      (3) filter them
      (4) compute descriptors
      (5) write in YAML format the keypoints and descriptors to output file
   =============================================================================================== */
  for (int i = 0; i < files.size(); i++)
  {
    if (i % 100 == 0  || i == files.size())
      cout << "Processing image # " << setw(4) << i << " out of " << files.size() << " files" << endl;

    //read in image file and generate output file name
    name = path2dir + files[i];
    nameful = files[i];
    nameful.erase(nameful.find_last_of("."), string::npos);
    nameful = path2outdir + nameful + extension;
    Mat image = imread (name);

    //SURF detection and then filter
    detector.detect (image, keypoints);
    //cout << "keypoints: " << keypoints.size();
    filter_keypoints (keypoints, sizemin, responsemin);
    //cout << "  after filter: " << keypoints.size() << endl;

    //write into output file
    FileStorage fs (nameful, FileStorage::WRITE);
    fs << "keypoints" << keypoints;
    if (keypoints.size() > 0)
    {
      extractor->compute (image, keypoints, descriptors);
      fs << "descriptors" << descriptors;
    }
    fs.release();
  }

  cout << "Processed all " << files.size() << " images, and placed the .yml files in " << path2outdir << endl;
  return 0;
}  




/* ===============================================================================================
   Procedure that generates the usage for the code, only called if incorrectly run
   =============================================================================================== */
int usage()
{
    cout << "\n\n" <<endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "=                                      ProcessImages                                           ="  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "=     This program reads in a collection of images, finds keypoints, computers features        ="  <<  endl;
    cout << "     " << "=     of those keypoints, and store them                                                       ="  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "=     Usage is:                                                                                ="  <<  endl;
    cout << "     " << "=                 ./processImages.exe                                                          ="  <<  endl;
    cout << "     " << "=                                 -i  <path to directory with images>                          ="  <<  endl;
    cout << "     " << "=                                 -o  <path to output directory for keypoints>                 ="  <<  endl;
    cout << "     " << "=                                 -p  <path to param file for SURF>                            ="  <<  endl;
    cout << "     " << "=                                                                                              ="  <<  endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "     " << "================================================================================================"  <<  endl;
    cout << "\n\n" <<endl;

    cout << "otherwise if not using a parameter file:" << endl;
    cout << "./a.out -i -o -h -oct -l -s -r" << endl;
}



/* ===============================================================================================
   Procedure to parse the command line options for the program
   =============================================================================================== */
void read_flags(int argc, char** argv, string *path2dir, string *path2outdir, string *param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin)
{
  string input;
  for(int i = 1; i < argc; i++)
  {
    input = argv[i];
    if (input == "-i") 
      *path2dir = argv[i + 1];
    if (input == "-o") 
      *path2outdir = argv[i + 1];
    if (input == "-p")
      *param = argv[i + 1];

    if (input == "-h")
      *minh = atoi(argv[i+1]);
    if (input == "-oct")
      *octaves = atoi(argv[i+1]);
    if (input == "-l")
      *layers = atoi(argv[i+1]);
    if (input == "-s")
      *sizemin = atoi(argv[i+1]);
    if (input == "-r")
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
   Procedure to extract list of files from a directory
   =============================================================================================== */
int get_filelist (string path, vector <string> &allfiles)
{
 //uses dirent.h to get the name of all the files within a directory
  DIR *dp; //directory stream
  struct dirent *entry; //directoryentry

  if ((dp = opendir (path.c_str())) == NULL)
    return -1;

  while ((entry = readdir(dp)) != NULL)
    allfiles.push_back (string (entry->d_name));

  closedir (dp);
  return 0;
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



  
