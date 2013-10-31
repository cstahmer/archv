#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
const int MAX_KERNEL_LENGTH = 31;
const float MAX_CONTOUR_RATIO = 1;


/** @function main */
int main( int argc, char** argv ) {

	Helper helper;
	string evalDirectory = EVAL_DIR;
	string contourImageDirectory = "/usr/local/share/archive-vision/build/countour_images/";
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	bool isDir = false; // set to designate if this is an evaluation directory of a single file to be evaluated
	bool makeContours = false;
	bool makeThreshold = false;
	bool makeHull = false;
	bool makeOutline = false;

	
	//"Usage is -eval <dirctory of evaluation files> -write <directory to which histogram files should be saved> -dict <dictionary file to load>"
	
    for (int i = 1; i < argc; i++) { 
    	string arument = argv[i];
        if (arument == "-eval") {
        	evalDirectory = argv[i + 1];
        }
        if (arument == "-write") {
        	contourImageDirectory = argv[i + 1];
        }
        if (arument == "-back") {
        	runInBackground = argv[i + 1];
        }
        if (arument == "-contours") {
        	makeContours = true;
        }
        if (arument == "-threshold") {
        	makeThreshold = true;
        } 
        if (arument == "-hull") {
        	makeHull = true;
        }
        if (arument == "-outline") {
        	makeOutline = true;
        }     
        if (arument == "-help") {
            cout << "Usage: ./makeWordHistograms -eval <dirctory of evaluation files or name of evaluation file> -write <directory to which contour files should be saved> -hull <to make hull contour> -back <true/false to run in background mode>"<<endl;
            exit(0);
        } 
    }
	
	event = "Starting Contours execuatable.";
	helper.logEvent(event, 2, runInBackground, writelog);
    event = "Evaluation Directory/File: " + evalDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "New Contour images will be saved to: " + contourImageDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    if (makeHull) {
    	event = "Saving all contours.";
    } else {
    	event = "Saving convex hull contour";
    }
    helper.logEvent(event, 2, runInBackground, writelog);

	
	//make sure that the eval directory exists
	struct stat sb;
	const char * fc = evalDirectory.c_str();
	if (stat(fc, &sb) == 0 && S_ISDIR(sb.st_mode)) {
		isDir = true;
	}
	if (!isDir) {
		// ok, if this isn't a directory, see if it is a single file that can be loaded/
		// ultimately should check name and see if it is a .jpg file and not just a file
		if (access (evalDirectory.c_str (), F_OK) != 0) {
			event = evalDirectory + " is not a valid directory or image file or you do not have the correct permissions!";
		    helper.logEvent(event, 0, runInBackground, writelog);
		    event = "Operation Aborted!";
		    helper.logEvent(event, 0, runInBackground, writelog);
			exit(0);
		}
	}

	vector<string> files = vector<string>();	
	int totalEvalFiles = 0;
	string numFilesToProcess = "0";
	// if this is a whole directory process according
	if (isDir) {
		helper.GetFileList(evalDirectory, files);
		totalEvalFiles = files.size()-2;
		numFilesToProcess = static_cast<ostringstream*>( &(ostringstream() << (totalEvalFiles)) )->str();
		event = "Processing " + numFilesToProcess + " files in evaluation directory.";
		helper.logEvent(event, 2, runInBackground, writelog);
	} else {
		totalEvalFiles = 1;
		files.push_back(evalDirectory);	
		numFilesToProcess = "1";
		event = "Processing 1 file.";
		helper.logEvent(event, 2, runInBackground, writelog);
	}
  
	
	string evalFileBase;
	string evalFileLessExtension;
	string evalFileFullPath;
	int totalContourFiles = 0;
	int totalThresholdFiles = 0;
	int totalHullFiles = 0;
	int totalOutlineFiles = 0;
	int totalNumFiles = 0;
	for (unsigned int iz = 0;iz < files.size();iz++) {
		totalNumFiles++;
		
		event = "Processing file " + files[iz] + ".";
		helper.logEvent(event, 2, runInBackground, writelog); 
		
		int isImage = helper.instr(files[iz], "jpg", 0, true);
		if (isImage > 0) {
		
			evalFileBase = files[iz];
			string secondDelim = ".";
			vector<string> filenamePart = helper.Split(secondDelim, evalFileBase);
			string finalPart;
			if (filenamePart.size() > 1) {
				evalFileLessExtension = filenamePart[(filenamePart.size() - 2)];
			} else {
				evalFileLessExtension = evalFileBase;
			}	
			evalFileFullPath = evalDirectory + evalFileBase;		
			
			Mat threshold_output;
			vector<vector<Point> > contours;
			vector<vector<Point> > contoursThresh;
			vector<Vec4i> hierarchy;
			
			/// Load source image and convert it to gray
			src = imread( evalFileFullPath, 1 );
			cvtColor( src, src, CV_BGR2GRAY );
			blur( src, src, Size(7,7) );
			Mat src_contours;
			Mat src_Threshold;
			Mat src_outline;
			  
			// first setup image clones to use for each type of process
			if (makeContours) {
				src_contours = src.clone();
			}
			if (makeThreshold || makeHull || makeOutline) {
				src_Threshold = src.clone();
			}
			
			  
			// CANNY EDGE DETECTION
			if (makeContours) {
				event = "Calculating Canny Contours for " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog); 
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_canny_contours.jpg";
				Mat canny_output;
				  
				/// Detect edges using canny
				Canny( src_contours, canny_output, thresh, thresh*2, 3 );
				  
				/// Find contours canny
				findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			
				/// Draw Canny Contours
				Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
				  
				int iterations = contours.size() * MAX_CONTOUR_RATIO;
				for( int i = 0; i< iterations; i++ ) {
					Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
					drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
				}
				   
				event = "Saving canny contour image " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog); 
				if(!imwrite(writeFileFullPath, drawing) ) {
					event = "Error Saving " + writeFileFullPath + ".";
					helper.logEvent(event, 0, runInBackground, writelog);
				} else {
					event = writeFileFullPath + "Successfully Saved.";
					helper.logEvent(event, 4, runInBackground, writelog);
					totalContourFiles++;
				}
			  
			}
			
			// Threshold, Covex, or Outline
			if (makeThreshold || makeHull || makeOutline) {
				event = "Calculating Threshold Contours for " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				/// Dectect edges using Threshold
				threshold(src_Threshold, threshold_output, thresh, 255, THRESH_BINARY );
				
				/// Find countours threshold
				findContours( threshold_output, contoursThresh, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
				
			}
			
			// if Threshold then draw the threshold image.
			if (makeThreshold) {
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_threshold_contours.jpg";
				event = "Saving Threshold Contours Image " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog);
				Mat drawingThresh = Mat::zeros( threshold_output.size(), CV_8UC3 ); 
				for( int i = 0; i< contoursThresh.size(); i++ ) {
				     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				     drawContours( drawingThresh, contoursThresh, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
				}
				if(!imwrite(writeFileFullPath, drawingThresh) ) {
					event = "Error Saving " + writeFileFullPath + ".";
					helper.logEvent(event, 0, runInBackground, writelog);
				} else {
					event = writeFileFullPath + "Successfully Saved.";
					helper.logEvent(event, 4, runInBackground, writelog);
					totalThresholdFiles++;
				}
				
			}

			if (makeHull) {
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_cvhull_contours.jpg";
				event = "Saving Threshold Contours Image " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog);
				
				//find Convex Hull object for each contour
				vector<vector<Point> >hull( contours.size() ); 
				for( int i = 0; i < contoursThresh.size(); i++ ) { 
					convexHull( Mat(contoursThresh[i]), hull[i], false ); 
				}
				
				Mat drawingHull = Mat::zeros( threshold_output.size(), CV_8UC3 ); 
				
				for( int i = 0; i< contoursThresh.size(); i++ ) {
				     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				// 	   drawContours( drawingThresh, contoursThresh, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
				     drawContours( drawingHull, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
				}
				
				if(!imwrite(writeFileFullPath, drawingThresh) ) {
					event = "Error Saving " + writeFileFullPath + ".";
					helper.logEvent(event, 0, runInBackground, writelog);
				} else {
					event = writeFileFullPath + "Successfully Saved.";
					helper.logEvent(event, 4, runInBackground, writelog);
					totalHullFiles++;
				}
				
			}
			
			  
			// OUTLINE
			if (makeOutline) {
			
				event = "This functionality not yet implemented.";
				helper.logEvent(event, 1, runInBackground, writelog);
			}
			  
		} else {
			event = "Skipping file " + files[iz] + ".";
			helper.logEvent(event, 1, runInBackground, writelog); 
		}
	}
	
	string strSavedFilesCount;
	string strTotalFiles = static_cast<ostringstream*>( &(ostringstream() << (totalNumFiles)) )->str();
	if (makeContours) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalContourFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  "Canny counter images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}		
	}
	if (makeThreshold) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalThresholdFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  "Threshold contour images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}
	}
	if (makeHull) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalHullFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  "Convex Hull counter images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}
	}
	if (makeOutline) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalOutlineFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  "Outline counter images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}
	}
	
	event = "Contours Process Completed.";
	helper.logEvent(event, 2, runInBackground, writelog);
	event = "Bye, Bye!";
	helper.logEvent(event, 2, runInBackground, writelog);

	return(0);
}
