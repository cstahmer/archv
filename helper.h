/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: helper.h
 *
 * Contains classes and functions for various file system
 * operations
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

#ifndef HELPER_CONFIG_H
#define HELPER_CONFIG_H

#include <opencv2/opencv.hpp>
#include <cv.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <string>
#include <locale>

using namespace cv;
using namespace std;

//const string TRAINING_DIR = "/usr/local/share/archive-vision/build/train/";
const string TRAINING_DIR = "/web/sites/beeb/training_images/";
const string EVAL_DIR = "/web/sites/beeb/impression_images/";
const string EVAL_DESCRIPTOR_DIR = "/usr/local/share/archive-vision/build/eval_descriptors/";
const string COLLECTION_VW_FILES_DIR = "/usr/local/share/archive-vision/build/index_files/";
const string EVAL_DICTIONARY_PATH = "/usr/local/share/archive-vision/build/dictionary.yml";
const string CONTOUR_IMAGE_DIR = "/usr/local/share/archive-vision/build/countour_images/";
const string CONTOUR_YAML_DIR = "/usr/local/share/archive-vision/build/countour_yaml/";
const bool RUN_IN_BACKGROUND = false;
const string LOG_PATH = "/usr/local/share/archive-vision/build/log/";
const bool WRITE_LOG = false;
const int TERM_CRIT_MAX_COUNT = 10; // 10 is sampel code value
const double TERM_CRIT_EPSILON = 0.001; // .001 is sample code value
const int SURF_MIN_HESSIAN = 5000; // 500 is sample code value
const int SURF_OCTAVES = 4;
const int SURF_OCTAVE_LAYERS = 2;
const string VISUAL_WORD_FILE_DIRECTORY = "/usr/local/share/archive-vision/build/contour_vw_files/";
const double CONTOUR_COMPARISSON_EPSILON = 0.5;
const int CONTOUR_COMPARISON_MATCH_TYPE = 1;
const double CONTOUR_COMPARISSON_MIN_CONTOUR = 90;  // The minimum area of contours to consider when doing a contour comparision
const bool SAVE_FEATURE_POINT_IMAGES = false;
const string FEATURE_POINT_IMAGES_OUT_DIR = "/usr/local/share/archive-vision/build/feature_point_images";
const double KEYPOINT_SIZE_FILTER = 100;
const double KEYPOINT_RESPONSE_FILTER = 2000;



/*
 * SVM CLASSIFIER flag set true to turn on the classifier
 */
//const bool SVM_CLASSIFIER          = true;
/*
 * Naive Bayes CLASSIFIER flag set true to turn on the classifier
 */
//const bool NB_CLASSIFIER           = true;
//const int  NUM_TEST_IMAGES         = 5;
//const int  NUM_TRAINING_IMAGES     = 50;
/*
 * Total number of clusters
 */
//const int  NUMBER_OF_CLUSTERS      = 1000;
/*
 * Total number of training categories.
 */
//const int  NUM_TRAINING_CATEGORIES = 7;
/*
 * Load from file flag, set true to read already saved files from disk.
 */
//const bool LOAD_FROM_FILE          = true;
/*
 * path on disk where keypoints for training images are stored.
 */
//const string KEYPOINTS_BASE_PATH = "data//keypoints//";
/*
 * Base path for the size of keypoints.
 */
//const string KEYPOINTS_SIZE_BASE_PATH = KEYPOINTS_BASE_PATH + "size//";
//const string DOUBLE_SLASH = "//";

/*
 * Training classes categories
 */
//const string categories[NUM_TRAINING_CATEGORIES] = {"airplane","chandelier","face","ketch", "leopards", "motorbikes", "watch"};


/*
 * Custom structure which encapsulates values like imagename, image, keypoints,
 * histogram and image class.
 * These values are used by various stages in the VCat project.
 */
//struct Image 
//{
//	string imageName; //Name of the training image without extension.
//	Mat image;        //The matrix representation
//	vector<KeyPoint> keyPoints; // key points identified for this particular image
//	Mat histogram; // histogram generated using image and keypoints.
//	int imageClass; //Can be one of the indexes belonging to 'categories' array
//};

#endif

#pragma once
class Helper
{

public:

	Helper();
	~Helper();

	/*
	 * The method writes Mat object to disk.
	 * 
	 *  fileName - Name of the file to be written to disk.
	 *                                                             
	 *  structure - Mat object to be written to disk.
	 *  structureName - structure name to be used to write the file.
	 *
	 *  PreCondition - The Mat object to be written to disk is and the filename are passed in as arguments.
	 *                 
	 *  PostCondition - The Mat object is written to file and saved as the file name passed in as argument.
	 *                  
	 */
	void WriteToFile(string fileName, Mat structure, string structureName);

	/*
	 * The method reads Mat object from disk.
	 * 
	 *  fileName - Name of the file to be read from disk.
	 *                                                             
	 *  structure - Mat object to read data in.
	 *
	 *  structureName - structure name to be used to read the file.
	 *
	 *  return Mat read from the disk.
	 *
	 *  PreCondition - The Mat object to be read from disk and the filename are passed in as arguments.
	 *                 
	 *  PostCondition - The Mat object is read from file and returned.
	 *                  
	 */
	Mat ReadFromFile(string fileName, Mat structure, string structureName);
	
	Mat ReadMatFromFile(string fileName, string structureName);
	
   /*
	* The method prints the elapsed time.
	* 
	*  PreCondition - The time t1 is already set by StartClock method.
	*                 
	*  PostCondition - The method prints the difference between start time and end time.
	*                  
	*/
	void StopAndPrintClock();
	
	/*
	 * The method prints the running elapsed time.
	 * 
	 *  PreCondition - The time t1 is already set by StartClock method.
	 *                 
	 *  PostCondition - The method prints the difference between start time and time of invocation.
	 *                  
	 */	
	void PrintElapsedClock();
	
	/*
	 * The method captures the start time as t1 to calculate total elapsed time.
	 * 
	 *  PreCondition - The time t1 object is created.
	 *                 
	 *  PostCondition - The time t1 is set to current time.
	 *                  
	 */
	void StartClock();
	
	/*
	 * The method returns a vector of all files in a directory.
	 * 
	 *  PreCondition - The directory to be searched is passed into string directory.
	 *                 
	 *  PostCondition - A vector containing all non-directory filenames in the directory is returned.
	 *                  
	 */
	int GetFileList(string directory, vector<string> &files);
	
	/*
	 * Method that performs a java instr type function returning the 
	 * start position of the occurance of a substring withing a string
	 * 
	 * str - 		the string to be searched
	 * toFind - 	the string to find
	 * start -		the position in string to start searching forward from
	 * ignoreCae -	true ignores case on search
	 */
	int instr(string str, string toFind, int start, bool ignoreCase);
	
	/*
	 * Function that receives string input and writes content to the filesystem
	 * 
	 * strdata -	string data to write to the file
	 * filename - 	the filename to write the datat to
	 */
	int writeTextFile(string strdata, string filename);
	
	void logEvent(string strEvent, int eventType, bool background, bool writelog);
	
	vector<string> Split(string delim, string baseString);
	
	void writeStingToFile(string strData, string strFilename);
	
	

};