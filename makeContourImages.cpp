/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeContourImages.cpp
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

#include "makeContourImages.h"

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 150;
int max_thresh = 255;
RNG rng(12345);
const int MAX_KERNEL_LENGTH = 31;
const float MAX_CONTOUR_RATIO = 1;
string event;


/** @function main */
int main( int argc, char** argv ) {

	Helper helper;
	string evalDirectory = EVAL_DIR;
	string contourImageDirectory = CONTOUR_IMAGE_DIR;
	string contourYamlDirectory = CONTOUR_YAML_DIR;
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
        if (arument == "-writeyaml") {
        	contourYamlDirectory = argv[i + 1];
        }
        if (arument == "-back") {
        	runInBackground = true;
        }
        if (arument == "-log") {
        	writelog = true;
        }
        if (arument == "-canny") {
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
            cout << "Usage: ./makeWordHistograms -eval <dirctory of evaluation files or name of evaluation file> -write <directory to which contour files should be saved> -canny <make canny edge contours> -threshold <make threshold contours -hull <make convex hull contours> -outline <make a largest outline hull contour> -back [flag to run in background mode] -log [flag to run in log mode]"<<endl;
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
		
		event = "Processing file " + files[iz] + ".";
		helper.logEvent(event, 2, runInBackground, writelog); 
		
		int isImage = helper.instr(files[iz], "jpg", 0, true);
		if (isImage > 0) {
			totalNumFiles++;
			
			string writeFileFullPath;
			string writeContoursYAML;
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
			Mat shapesYMLCanny;
			
			/// Load source image and convert it to gray
			src = imread( evalFileFullPath, 1 );
			cvtColor( src, src, CV_BGR2GRAY );
			//blur( src, src, Size(5,5) );
			blur( src, src, Size(1,1) );
			Mat src_contours;
			Mat src_Threshold;
			Mat src_Outline;
			Mat canny_output;
			  
			// first setup image clones to use for each type of process
			if (makeContours) {
				src_contours = src.clone();
			}
			if (makeThreshold || makeHull || makeOutline) {
				src_Threshold = src.clone();
			}

			if (makeOutline) {
				src_Outline = src.clone();
			}
			  
			// CANNY EDGE DETECTION
			if (makeContours) {
				event = "Calculating Canny Contours for " + evalFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog); 
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_canny_contours.jpg";
				writeContoursYAML =  contourYamlDirectory + evalFileLessExtension + "_canny_contours.yml";
				  
				/// Detect edges using canny
				Canny( src_contours, canny_output, thresh, thresh*2, 3 );
				  
				/// Find contours canny
				findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			
				/// Draw Canny Contours
				Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
				 
				if (contours.size() > 0) {
					
					// save the contours yaml
					event = "Getting Canny Shapes for " + evalFileFullPath + ".";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					event = "Creating Canny Image for " + evalFileFullPath + ".";
					helper.logEvent(event, 2, runInBackground, writelog); 
				
					int iterations = contours.size() * MAX_CONTOUR_RATIO;
					for( int i = 0; i< iterations; i++ ) {
						Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
						drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
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
				
				} else {
					event = evalFileFullPath + "Contains no Contours.";
					helper.logEvent(event, 3, runInBackground, writelog);
					event = "No contour image saved.";
					helper.logEvent(event, 3, runInBackground, writelog);
				}
			  
			}
			
			// Threshold, Covex, or Outline
			if (makeThreshold || makeHull || makeOutline) {
				event = "Calculating Threshold Contours for " + evalFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				/// Dectect edges using Threshold
				threshold(src_Threshold, threshold_output, thresh, 255, THRESH_BINARY );
				
				/// Find countours threshold
				findContours( threshold_output, contoursThresh, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
				
			}
			
			// if Threshold then draw the threshold image.
			if (makeThreshold) {
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_threshold_contours.jpg";
				writeContoursYAML = contourYamlDirectory + evalFileLessExtension + "_threshold_contours.yml";
				event = "Saving Threshold Contours Image " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog);
				Mat drawingThresh = Mat::zeros( threshold_output.size(), CV_8UC3 ); 
				
				if (contoursThresh.size() > 0) {
					
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
				} else {
					event = evalFileFullPath + "Contains no Contours.";
					helper.logEvent(event, 3, runInBackground, writelog);
					event = "No contour image saved.";
					helper.logEvent(event, 3, runInBackground, writelog);
				}
				
				
			}

			if (makeHull) {
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_cvhull_contours.jpg";
				writeContoursYAML =  contourYamlDirectory + evalFileLessExtension + "_cvhull_contours.yml";
				event = "Saving Convex Hull Contours Image " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog);
				
				//find Convex Hull object for each contour
				vector<vector<Point> >hull( contoursThresh.size() ); 
				for( int i = 0; i < contoursThresh.size(); i++ ) { 
					convexHull( Mat(contoursThresh[i]), hull[i], false ); 
				}
				
				Mat drawingHull = Mat::zeros( threshold_output.size(), CV_8UC3 ); 
				
				if (contoursThresh.size() > 0) {
				
					for( int i = 0; i< contoursThresh.size(); i++ ) {
					     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
					     drawContours( drawingHull, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
					}
					
					if(!imwrite(writeFileFullPath, drawingHull) ) {
						event = "Error Saving " + writeFileFullPath + ".";
						helper.logEvent(event, 0, runInBackground, writelog);
					} else {
						event = writeFileFullPath + "Successfully Saved.";
						helper.logEvent(event, 4, runInBackground, writelog);
						totalHullFiles++;
					}
				}  else {
					event = evalFileFullPath + "Contains no Contours.";
					helper.logEvent(event, 3, runInBackground, writelog);
					event = "No contour image saved.";
					helper.logEvent(event, 3, runInBackground, writelog);
				}
				
			}
			
			  
			// OUTLINE
			if (makeOutline) {

				/// Detect edges using canny
				Canny( src_Outline, canny_output, thresh, thresh*2, 3 );
				  
				/// Find contours canny
				findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
				
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_outline_contour.jpg";
				writeContoursYAML = contourYamlDirectory + evalFileLessExtension + "_outline_contours.yml";
				
				event = "Calculating Outline Contours Image for " + evalFileLessExtension + ".jpg.";
				helper.logEvent(event, 2, runInBackground, writelog);
				
				vector<Point> pointList;
				vector<Point> rightSidePointList;
				vector<Point> leftSidePointList;
				vector<Point> clockwiseWalkPointList;
				vector<Point> outputPointList;
				
				//pushd all points from all contours into a single point vector
				
				//if (contoursThresh.size() > 0) {
					/* 
					//from threshold contours
					vector<vector<Point> >hull( contoursThresh.size() ); 
					for( int i = 0; i < contoursThresh.size(); i++ ) { 
						vector<Point> thisPointList = contoursThresh[i];
						for( int itp = 0; itp < thisPointList.size(); itp++ ) { 
							pointList.push_back(thisPointList[itp]);
						}
					}
					*/
					
				if (contours.size() > 0) {
					// from canny edges
					vector<vector<Point> >hull( contoursThresh.size() ); 
					for( int i = 0; i < contours.size(); i++ ) { 
						vector<Point> thisPointList;
						thisPointList = contours[i];
						for( int itp = 0; itp < thisPointList.size(); itp++ ) { 
							pointList.push_back(thisPointList[itp]);
						}
					}
					
					int pointListSize = pointList.size();
					string numPointsInList = static_cast<ostringstream*>( &(ostringstream() << (pointListSize)) )->str();
					event = "Found " + numPointsInList + " total points.";
					helper.logEvent(event, 2, runInBackground, writelog);
					
					Mat drawingOutline = Mat::zeros( threshold_output.size(), CV_8UC3 ); 
					int intXAxisPixils = drawingOutline.cols;
					int intYAxisPixils = drawingOutline.rows;
					string drawingRows = static_cast<ostringstream*>( &(ostringstream() << (intYAxisPixils)) )->str();;
					string drawingCols = static_cast<ostringstream*>( &(ostringstream() << (intXAxisPixils)) )->str();;
					
					event = "Image is " + drawingCols + " pixils wide and " + drawingRows  + " pixils high.";
					helper.logEvent(event, 2, runInBackground, writelog);
					
					// for each column value from 0 to hight of image
					for( int ifpl = 0; ifpl < intYAxisPixils; ifpl++ ) { 
						vector<int> rowXPoints;
						vector<int> rowPointVec;
						int arrSize = 0;
						// iterate through the entire list of points that sit on this row
						for( int izt = 0; izt < pointList.size(); izt++ ) { 
							Point thisPoint = pointList[izt];
							if (thisPoint.y == ifpl) {
								//cout << thisPoint.x << ", " << thisPoint.y << endl;
								arrSize ++;
								rowPointVec.push_back(thisPoint.x);
							}
						}
						if (rowPointVec.size() > 0) {
							const int highest = *max_element(rowPointVec.begin(), rowPointVec.end());
							const int lowest = *min_element(rowPointVec.begin(), rowPointVec.end());
							sort(rowPointVec.begin(), rowPointVec.end());
							/*
							 * At this point I have an ordered list of x points on this row (y).
							 * Check to see if there is only one point on this row and if so push it to the end of the points vector.
							 * If more than one, push the lowest and then the highest to the vector.
							 */
							
							if (lowest == highest) {
								Point singlePoint;
								singlePoint.x = lowest;
								singlePoint.y = ifpl;
								outputPointList.push_back(singlePoint);
							} else {
								Point lowPoint;
								lowPoint.x = lowest;
								lowPoint.y = ifpl;
								Point highPoint;
								highPoint.x = highest;
								highPoint.y = ifpl;
								outputPointList.push_back(lowPoint);
								outputPointList.push_back(highPoint);
								
							}
							
							Point rightPoint;
							rightPoint.x = highest;
							rightPoint.y = ifpl;
							rightSidePointList.push_back(rightPoint);
							Point leftPoint;
							leftPoint.x = lowest;
							leftPoint.y = ifpl;
							leftSidePointList.push_back(leftPoint);
							
						}
					}
					
					// now reverse the leftSidePointList so that it is a walk back up from bottom to top
					reverse(leftSidePointList.begin(), leftSidePointList.end());
					// now combine the rightSidePointList and leftSidePointList and the resulting vector should be a clockwise walk around the poly.
					clockwiseWalkPointList.reserve( rightSidePointList.size() + leftSidePointList.size() ); // preallocate memory
					clockwiseWalkPointList.insert( clockwiseWalkPointList.end(), rightSidePointList.begin(), rightSidePointList.end() );
					clockwiseWalkPointList.insert( clockwiseWalkPointList.end(), leftSidePointList.begin(), leftSidePointList.end() );
					
					
					
					/*
					 * ok.  Here outputPointList contains an ordered list of lowest and highest x points on each row.
					 * Now I need to traverse this list so that I'm walking from point to point clockwise around the polygon.
					 * This will create an ordered list of points that can be drawn by just connectng the dots.
					 */
					
					//vector<Point> rightSidePointList;
					//vector<Point> leftSidePointList;
					//vector<Point> clockwiseWalkPointList;
					
					//outputPointList
					//Point thisPoint
					
					//int numFinishedPoints = outputPointList.size();
					//fillConvexPoly( cv::Mat&,      const cv::Point*, int,                 const cv::Scalar&,    int, int)
					//fillConvexPoly( drawingOutline, outputPointList, numFinishedPoints, Scalar( 255, 255, 255 ), 8, 0);
					
					
					//helper.WriteToFile(writeContoursYAML, clockwiseWalkPointList, "outline_contour");
					
					
					for( int iztax = 0; iztax < (clockwiseWalkPointList.size()-1); iztax++ ) {
						Point thisPointFinal = clockwiseWalkPointList[iztax];
						Point thisNextPoint = clockwiseWalkPointList[iztax+1];
						line( drawingOutline, thisPointFinal,  thisNextPoint, Scalar( 255 ), 1, 8, 0 );
						//cout << thisPointFinal.x << ", " << thisPointFinal.y << endl;
					}
					
					
					/*
					
					for( int i = 0; i< contoursThresh.size(); i++ ) {
					     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
					     drawContours( drawingHull, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
					}
					
					*/
					
					if(!imwrite(writeFileFullPath, drawingOutline) ) {
						event = "Error Saving " + writeFileFullPath + ".";
						helper.logEvent(event, 0, runInBackground, writelog);
					} else {
						event = writeFileFullPath + "Successfully Saved.";
						helper.logEvent(event, 4, runInBackground, writelog);
						totalOutlineFiles++;
					}
				
				}  else {
					event = evalFileFullPath + "Contains no Contours.";
					helper.logEvent(event, 3, runInBackground, writelog);
					event = "No contour image saved.";
					helper.logEvent(event, 3, runInBackground, writelog);
				}
				
			}
			  
		} else {
			event = "Skipping file " + files[iz] + ".";
			helper.logEvent(event, 1, runInBackground, writelog); 
		}
	}
	
	string strTotalFiles = static_cast<ostringstream*>( &(ostringstream() << (totalNumFiles)) )->str();
	if (makeContours) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalContourFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  " Canny counter images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}		
	}
	if (makeThreshold) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalThresholdFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  " Threshold contour images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}
	}
	if (makeHull) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalHullFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  " Convex Hull counter images.";
		if (strSavedFilesCount == strTotalFiles) {
			helper.logEvent(event, 4, runInBackground, writelog);
		} else {
			helper.logEvent(event, 3, runInBackground, writelog);
		}
	}
	if (makeOutline) {
		string strSavedFilesCount = static_cast<ostringstream*>( &(ostringstream() << (totalOutlineFiles)) )->str();
		event = "Successfully saved " + strSavedFilesCount + " of " + strTotalFiles +  " Outline counter images.";
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
