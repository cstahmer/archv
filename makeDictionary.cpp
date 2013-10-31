/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeDictionary.cpp
 *
 * Contains classes and functions for building a Visual Word Dictionary
 * using a collecitons files in a command line argument designated directory.
 * Size of dictionary and output dictionary name also delivered as command line arguments
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

#include "makeDictionary.h"

using namespace cv; 
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;


char ch[30];


int main(int argc, char* argv[]) {
	
	Helper helper;
	string event;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	
	//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
	
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
	
	

	Mat dictionary;
	
	//"Usage is -d <dirctory of training files> -n <name of dictionary output file> -s <size of dictionary> \n"
	
    for (int i = 1; i < argc; i++) { 
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
    
    //TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
    TermCriteria tc(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, intTermCritMaxCount, dblTermCritEpsilon);
	
	// SurfFeatureDetector( double hessianThreshold = 400., int octaves = 3, int octaveLayers = 4 );
	// double hessianThreshold - Threshold for the keypoint detector. Only features, whose hessian is larger than hessianThreshold are retained by the detector. Therefore, the larger the value, the less keypoints you will get. A good default value could be from 300 to 500, depending from the image contrast.
	// int nOctaves - The number of a gaussian pyramid octaves that the detector uses. It is set to 4 by default. If you want to get very large features, use the larger value. If you want just small features, decrease it.
	// int nOctaveLayers - The number of images within each octave of a gaussian pyramid. It is set to 2 by default.
	
	
	// SurfFeatureDetector detector(SURF_MIN_HESSIAN); // original value 500 documentation says somewhere between 300-500 is good depending on sharpness and contrast
	SurfFeatureDetector detector(intSurfMinHession, intSurfOctaves, intSurfOctaveLayers); 
	
    
	string fullDictionaryFileName = dictionaryFileName + ".yml";
	
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
    
    
    

	
	// configure BOW trainer and extractor according to set paramaters
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	
	//make sure that the training directory exists
	bool isDir = false;
	struct stat sb;
	const char * fc = trainingDirectory.c_str();
	if (stat(fc, &sb) == 0 && S_ISDIR(sb.st_mode)) {
		isDir = true;
	}
	if (!isDir) {
		event = trainingDirectory + " is not a valid directory or you do not have the correct permissions!";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
	}
	
	event = "Calculating Centroids...";
	helper.logEvent(event, 2, runInBackground, writelog);
	// Call the collectclasscentroids function from imgextract.cpp
	collectclasscentroids(detector, extractor, bowTrainer, trainingDirectory, runInBackground, writelog, blnSaveFeaturePointImages, blnFeaturePointImagesOutDir, dblKeypointSizeFilter, dblKeypointResponseFilter);
	
	string strDescriptorCount = static_cast<ostringstream*>( &(ostringstream() << (bowTrainer.descripotorsCount())) )->str();
	event = "Clustering " + strDescriptorCount + " features.";
	helper.logEvent(event, 2, runInBackground, writelog);
	//cluster the descriptors int a dictionary
	dictionary = bowTrainer.cluster();
	
	
	event = "Saving Dictionary File";
	helper.logEvent(event, 2, runInBackground, writelog);
	helper.WriteToFile(fullDictionaryFileName, dictionary, dictionaryFileName);
	
	event = "Dictionary saved as " + fullDictionaryFileName + ".";
	helper.logEvent(event, 4, runInBackground, writelog);

	event = "makeDictionary Process Completed.";
	helper.logEvent(event, 2, runInBackground, writelog);
	event = "Bye, Bye!";
	helper.logEvent(event, 2, runInBackground, writelog);
	
	return 0;

}