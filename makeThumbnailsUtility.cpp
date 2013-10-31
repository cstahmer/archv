/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeWordHistograms.cpp
 *
 * Reads a library of images and outputs visual word documents
 * for the library.
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

#include "cv.h" 
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

#include "helper.cpp"
#include "imgextract.cpp"
#include "imageDocument.cpp"

using namespace cv; 
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;

char ch[30];

Mat dictionary;


int main(int argc, char* argv[]) {
	
	Helper helper;
	ImgDoc imagedoc;
	string event;
	
	// set default directories
	string evalDirectory = "/web/sites/beeb/finished-woodcut-images/";
	string writeToDirectory = "/web/sites/beeb/finished-woodcut-images_tn/";
	
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	int maxheight = 200;
	int maxwidth = 200;
	
	bool isDir = false; // set to designate if this is an evaluation directory of a single file to be evaluated

	
	//"Usage is -eval <dirctory of evaluation files> -write <directory to which histogram files should be saved> -dict <dictionary file to load>"
	
    for (int i = 1; i < argc; i++) { 
    	string arument = argv[i];
        if (arument == "-eval") {
        	evalDirectory = argv[i + 1];
        }
        if (arument == "-write") {
        	writeToDirectory = argv[i + 1];
        }
        if (arument == "-height") {
        	string strMaxHeight = argv[i + 1];
        	maxheight = atoi(strMaxHeight.c_str());
        }
        if (arument == "-width") {
        	string strMaxWidth = argv[i + 1];
        	maxwidth = atoi(strMaxWidth.c_str());
        }
        if (arument == "-log") {
        	writelog = true;
        }
        if (arument == "-back") {
        	runInBackground = true;
        }
        if (arument == "-help") {
            cout << "Usage: ./makeThumbnailsUtility -eval <dirctory of base files> -write <directory to save thumbnails> -height <max height of thumbnails> -width <max width of the thumbnails> -back [flag to run in background mode] -log [flag to run in log mode]"<<endl;
            exit(0);
        } 
    }
    
	event = "Starting makeThumbnailsUntility execuatable.";
	helper.logEvent(event, 2, runInBackground, writelog);
    event = "Evaluation Directory: " + evalDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Thumbnails will be saved to: " + writeToDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    string strMaxHeight = static_cast<ostringstream*>( &(ostringstream() << (maxheight)) )->str();
    event = "Max Height: " + strMaxHeight;
    helper.logEvent(event, 2, runInBackground, writelog);
    string strMaxWidth = static_cast<ostringstream*>( &(ostringstream() << (maxwidth)) )->str();
    event = "Max Width: " + strMaxWidth;
    helper.logEvent(event, 2, runInBackground, writelog);   
    
	
	//make sure that the eval directory exists
	struct stat sb;
	const char * fc = evalDirectory.c_str();
	if (stat(fc, &sb) == 0 && S_ISDIR(sb.st_mode)) {
		isDir = true;
	}
	if (!isDir) {
		event = evalDirectory + " is not a valid directory or you do not have the correct permissions!";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
	}
	

	helper.StartClock();
	//helper.PrintElapsedClock();

	event = "Reading Eval Directory...";
	helper.logEvent(event, 2, runInBackground, writelog);
	vector<string> files = vector<string>();	
	int totalEvalFiles = 0;
	string numFilesToProcess = "0";
	helper.GetFileList(evalDirectory, files);
	totalEvalFiles = files.size()-2;
	numFilesToProcess = static_cast<ostringstream*>( &(ostringstream() << (totalEvalFiles)) )->str();
	
	event = "Processing " + numFilesToProcess + " files in evaluation directory.";
	helper.logEvent(event, 2, runInBackground, writelog);		
	int numSuccessFiles = 0;
	int wfcount = 0;
	string evalFileBase;
	string fileToConvert;
	string fileToWrite;
	for (unsigned int iz = 0;iz < files.size();iz++) {
		fileToConvert = evalDirectory + files[iz];
		fileToWrite = writeToDirectory + files[iz];
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	event = "Processing " + fileToConvert + ".";
        	helper.logEvent(event, 2, runInBackground, writelog);
        	Mat originalMat = imread( fileToConvert, CV_LOAD_IMAGE_COLOR);
        	IplImage* original = cvLoadImage(fileToConvert.c_str());
        	double origHeight = originalMat.rows;
        	double origWidth = originalMat.cols;
        	
        	string strOH = static_cast<ostringstream*>( &(ostringstream() << (origHeight)) )->str();
        	string strOW = static_cast<ostringstream*>( &(ostringstream() << (origWidth)) )->str();
        	event = "Original Height: " + strOH + ".";
        	helper.logEvent(event, 2, runInBackground, writelog);
        	event = "Original Width: " + strOW + ".";
        	helper.logEvent(event, 2, runInBackground, writelog);
        	
        	
        	int newHeight;
        	int newWidth;
        	if (origHeight > origWidth) {
        		newHeight = maxheight;
        		double aspectRatio = newHeight / origHeight;
        		newWidth = origWidth * aspectRatio;
        	} else {
        		newWidth = maxwidth;
        		double aspectRatio = (newWidth / origWidth);
        		newHeight = origHeight * aspectRatio;
        	}
        

        	Mat newThumbnail = cvCreateMat(newHeight, newWidth, CV_8UC3);	       	
        	
        	//Mat thumbnail = cvCreateMat(newHeight, newWidth, CV_8UC3);
        	//cvResize(original, thumbnail);
        	
        	IplImage* thumbnail;
        	thumbnail = cvCreateImage(cvSize(newWidth,newHeight),original->depth,original->nChannels);
        	cvResize(original,thumbnail,CV_INTER_LINEAR);
        	
        	
			if(!cvSaveImage(fileToWrite.c_str(), thumbnail) ) {
				event = "Error Saving thumbnail " + fileToWrite + ".";
				helper.logEvent(event, 0, runInBackground, writelog);
			} else {
				event = "Successfully Saved Thumbnail " + fileToWrite + ".";
				helper.logEvent(event, 4, runInBackground, writelog);
				numSuccessFiles++;
			}
        } else {
        	event = "Skipping " + fileToConvert + ".";
        	helper.logEvent(event, 2, runInBackground, writelog);
        }
	}
	
	string strNumProcessedFiles = static_cast<ostringstream*>( &(ostringstream() << (numSuccessFiles)) )->str();
	event = "Created Thumbnails for " + strNumProcessedFiles + " of " + numFilesToProcess + " Files.";
	if (numSuccessFiles < totalEvalFiles) {
		helper.logEvent(event, 0, runInBackground, writelog);
	} else {
		helper.logEvent(event, 4, runInBackground, writelog);		
	}


	event = "./makeThumbnailsUtility Process Completed.";
	helper.logEvent(event, 2, runInBackground, writelog);
	event = "Bye, Bye!";
	helper.logEvent(event, 2, runInBackground, writelog);

	return 0;

}