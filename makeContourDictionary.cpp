/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeContourDictionary.cpp
 *
 * Contains classes and functions for building a Visual Word Dictionary
 * based on HU Moment coparison of contour shapes found in an image.
 * Process is to first calculate a matrix of points that define 
 * the designated type of contour (canny, threshold, etc.) and then
 * creates a dictionary. Unlike the usual openCV visual dictionary
 * approach, which is to run all the contours and then cluster them,
 * this builds the dictionary as it goes along.

 * Copyright (C) 2013 Carl Stahmer (cstahmer@gmail.com) 
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

#include "makeContourDictionary.h"

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 200;
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
	string saveVisWordFileDir = VISUAL_WORD_FILE_DIRECTORY;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	bool isDir = false; // set to designate if this is an evaluation directory of a single file to be evaluated
	bool makeContours = false;
	bool makeThreshold = false;
	bool makeHull = false;
	bool makeOutline = false;
	double compEpsilon = CONTOUR_COMPARISSON_EPSILON;
	string visWordString;
	int matchType = CONTOUR_COMPARISON_MATCH_TYPE;
	double minContourArea = CONTOUR_COMPARISSON_MIN_CONTOUR;
	
	vector<Moments> momentsDictionary;
	vector<vector<Point> > pointsDictionary;

	
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
        if (arument == "-epsilon") {
        	compEpsilon = atoi(argv[i + 1]);
        }
        if (arument == "-writevwfiles") {
        	saveVisWordFileDir = argv[i + 1];
        }  
        if (arument == "-matchtype") {
        	matchType = atoi(argv[i + 1]);
        }
        if (arument == "-mincontour") {
        	minContourArea = atof(argv[i + 1]);
        }
        if (arument == "-minthresh") {
        	thresh = atoi(argv[i + 1]);
        }
        if (arument == "-maxthresh") {
        	max_thresh = atoi(argv[i + 1]);
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
			blur( src, src, Size(5,5) );
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
				Canny( src_contours, canny_output, thresh, max_thresh, 3 );
				  
				/// Find contours canny
				findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
				
			
				/// Draw Canny Contours
				Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
				 
				if (contours.size() > 0) {
					
					// save the contours yaml
					//event = "Getting Canny Shapes for " + evalFileFullPath + ".";
					//helper.logEvent(event, 2, runInBackground, writelog); 
					//getShapePoints(contours);
					//shapesYMLCanny = getShapes(contours, hierarchy);
					//event = "Writing Shapes YAML for " + evalFileFullPath + ".";
					//helper.logEvent(event, 2, runInBackground, writelog); 
					//helper.WriteToFile(writeContoursYAML, shapesYMLCanny, "canny_contours");
					
					
					/*
					
					event = "Calculating Moments for " + evalFileFullPath + ".";
					helper.logEvent(event, 2, runInBackground, writelog); 
					vector<Moments> imageMoments = getMoments(contours);
					
					
					event = "Creating Visual Word File for " + evalFileFullPath + ".";
					helper.logEvent(event, 2, runInBackground, writelog); 
					string strVisualWordString = getVisualWords(momentsDictionary, imageMoments, compEpsilon);
					
					event = "Saving Visual Word File for " + evalFileFullPath + ".";
					helper.logEvent(event, 2, runInBackground, writelog); 
					bool fileSaved = saveVisualWordFile(strVisualWordString, "");
					
					if (fileSaved) {
						event = evalFileFullPath + " Successfully Saved.";
						helper.logEvent(event, 2, runInBackground, writelog); 
					} else {
						event = "Error Saving " + evalFileFullPath + "!";
						helper.logEvent(event, 2, runInBackground, writelog); 
					}
					
					
					*/
					
					
					event = "Building Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					visWordString = matchContourShapes(pointsDictionary, contours, compEpsilon, matchType);
					
					event = "Finished Building Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					
					event = "Writing Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					saveVisualWordFile(visWordString, evalFileLessExtension, saveVisWordFileDir);
					
					event = "Writing Visual Word File.";
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
			
			// Threshold, Convex, or Outline
			if (makeThreshold || makeHull || makeOutline) {
				event = "Calculating Threshold Contours for " + evalFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				/// Dectect edges using Threshold
				threshold(src_Threshold, threshold_output, thresh, max_thresh, THRESH_BINARY );
				
				/// Find countours threshold
				findContours( threshold_output, contoursThresh, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );	
				/// Throw out small ones
				vector<vector<Point> > areaContourThresh;
				for( int ica = 0; ica < contoursThresh.size(); ica++ ) {
					double areaCheckVal = fabs( contourArea(contoursThresh[ica]));
					if (areaCheckVal > minContourArea) {
						areaContourThresh.push_back(contoursThresh[ica]);
					}
					
				}
				contoursThresh = areaContourThresh;
				
			}
			
			// if Threshold then draw the threshold image.
			if (makeThreshold) {
				writeFileFullPath = contourImageDirectory + evalFileLessExtension + "_threshold_contours.jpg";
				writeContoursYAML = contourYamlDirectory + evalFileLessExtension + "_threshold_contours.yml";
				event = "Saving Threshold Contours Image " + writeFileFullPath + ".";
				helper.logEvent(event, 2, runInBackground, writelog);
				Mat drawingThresh = Mat::zeros( threshold_output.size(), CV_8UC3 ); 
				
				if (contoursThresh.size() > 0) {
					
					
					
					event = "Building Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					visWordString = matchContourShapes(pointsDictionary, contoursThresh, compEpsilon, matchType);
					
					event = "Finished Building Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					
					event = "Writing Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
					saveVisualWordFile(visWordString, evalFileLessExtension, saveVisWordFileDir);
					
					event = "Writing Visual Word File.";
					helper.logEvent(event, 2, runInBackground, writelog); 
					
				
					//shapesYMLCanny = getShapes(contoursThresh, hierarchy);
					//helper.WriteToFile(writeContoursYAML, shapesYMLCanny, "threshold_contours");
					
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
				
				
				event = "Building Visual Word File.";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				visWordString = matchContourShapes(pointsDictionary, hull, compEpsilon, matchType);
				
				event = "Finished Building Visual Word File.";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				
				event = "Writing Visual Word File.";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				saveVisualWordFile(visWordString, evalFileLessExtension, saveVisWordFileDir);
				
				event = "Writing Visual Word File.";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
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
	
	
	
	/*
	
	string strDictFilePath = "contourDictionary.yml";
	event = "Saving Dictionary.";
	helper.logEvent(event, 2, runInBackground, writelog);
	bool dictSaved = saveDictionary(momentsDictionary, strDictFilePath);
	if (dictSaved) {
		event = "Dictionary " + strDictFilePath + "Saved Successfully.";
		helper.logEvent(event, 2, runInBackground, writelog);
	} else {
		event = "Error Saving Dictionary " + strDictFilePath;
		helper.logEvent(event, 2, runInBackground, writelog);
	}
	
	*/
	
	string stringDictionarySize = static_cast<ostringstream*>( &(ostringstream() << (pointsDictionary.size())) )->str();
	event = "Created dictionary with " + stringDictionarySize + " words.";
	helper.logEvent(event, 2, runInBackground, writelog); 
	
	event = "Contours Process Completed.";
	helper.logEvent(event, 2, runInBackground, writelog);
	event = "Bye, Bye!";
	helper.logEvent(event, 2, runInBackground, writelog);

	return(0);
}

Mat getShapes(vector<vector<Point> > contours, vector<Vec4i> hierarchy) {
	
	Helper helper;
	string evalDirectory = EVAL_DIR;
	string contourImageDirectory = CONTOUR_IMAGE_DIR;
	string contourYamlDirectory = CONTOUR_YAML_DIR;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	
	event = "In getShapes";
	helper.logEvent(event, 2, runInBackground, writelog); 
	
	Mat shapes;
	
	
	// For each contour contours[i] , the elements hierarchy[i][0] , hiearchy[i][1] , hiearchy[i][2] , hiearchy[i][3] 
	// will be set to 0-based indices in contours of the next and previous contours at the same hierarchical level, 
	// the first child contour and the parent contour, respectively. If for some contour i there is no next, previous, 
	// parent or nested contours, the corresponding elements of hierarchy[i] will be negative
	
	
	
	vector<Point> pointList;
	vector<Point> workingPoint;
	vector<vector<Point> >shapePoints; 
	vector<int> alreadyDone;
	bool pointDone;
	
	
	int idx = 0;
	    for( ; idx >= 0; idx = hierarchy[idx][0] )
	
	
	//for( int i = 0; i < hierarchy.size(); i++ ) { 
	for( int i = 0; i < contours.size(); i++ ) { 
		
		pointDone = false;
		if (alreadyDone.size() > 0) {
			for( int iz = 0; iz < alreadyDone.size(); iz++ ) { 
				if (i == alreadyDone[iz]) {
					pointDone = true;
				}
				
			}
		}
		
		if (!pointDone) {
			
			vector<vector<Point> >shapePoints; 
			
			// add this point to the working point
			workingPoint = contours[i];
			event = "Pushing Initial Point";
			helper.logEvent(event, 2, runInBackground, writelog); 
			shapePoints.push_back(workingPoint);
			alreadyDone.push_back(i);
			
			int childPointIndex = i;
			
			// now walk through the child tree from hierarchy
			do {
				//  whatever
			
				childPointIndex = hierarchy[childPointIndex][0];
				string strChildPointIndex = static_cast<ostringstream*>( &(ostringstream() << (childPointIndex)) )->str();
				event = "Child Point Index: " + strChildPointIndex;
				helper.logEvent(event, 2, runInBackground, writelog); 
				if (childPointIndex >= 0) {
					workingPoint = contours[childPointIndex];
					event = "Pushing Child Point";
					helper.logEvent(event, 2, runInBackground, writelog); 
					shapePoints.push_back(workingPoint);
					alreadyDone.push_back(childPointIndex);
				}
				
				
			} while ( childPointIndex >= 0);
			
			
			event = "Pushing points vector to mat";
			helper.logEvent(event, 2, runInBackground, writelog); 
			// at this point, shapePoints should contain all of the points in this shape / level of the hierarchy
			shapes.push_back(shapePoints);
			
			
		}		
		
	}

	return shapes;
	
}

void getShapePoints(vector<vector<Point> > contours) {
	
	Helper helper;
	string eventx;
	
	//Mat shapes;
	
	for( int i = 0; i < contours.size(); i++ ) { 
		eventx = "Shape Number: " + i;
		helper.logEvent(eventx, 2, false, true); 
		vector<Point> thisShape = contours[i];
		for( int iv = 0; iv < thisShape.size(); iv++ ) { 
			Point thispoint = thisShape[iv];
			string strX = static_cast<ostringstream*>( &(ostringstream() << (thispoint.x)) )->str();
			string strY = static_cast<ostringstream*>( &(ostringstream() << (thispoint.y)) )->str();
			event = "     " + strX+ "," + strY;
			helper.logEvent(event, 2, false, true); 
		}
		
	}
	
	
}


vector<Moments> getMoments(vector<vector<Point> > contours) {
	
	vector<Moments> mu(contours.size() );
	for( int i = 0; i < contours.size(); i++ ) {
	   mu[i] = moments( contours[i], false );
	}
	
	return mu;
	
}

string getVisualWords(vector<Moments>& momentsDictionary, vector<Moments> imgMoments, int epsilong) {
	// loop through the vector of moments and process each one
	// by checking to see if there is already a comparable moment in
	// the dictionary.  If so, add the slice/ID of that moment to the 
	// visual words string.  If not, add the moment being tested to the
	// dictionary and then add the slice/id of the newly added moment to
	// the visual words string.
	
	// inialize visual word string return
	string visWordString;


	for( int i = 0; i < imgMoments.size(); i++ ) {
		
		if (momentsDictionary.size() > 0) {
			
			for( int im = 0; im < momentsDictionary.size(); im++ ) {
				// compare the moments here
				// momentsDictionary[im] compare to imgMoments[i]
				
				
				
			}
			
		} else {
			momentsDictionary.push_back(imgMoments[i]);
		}
	}
	

	
}

string matchContourShapes(vector<vector<Point> >& pointsDictionary, vector<vector<Point> > contours, double epsilon, int matchType) {
	Helper helper;
	string event;
	string evalDirectory = EVAL_DIR;
	string contourImageDirectory = CONTOUR_IMAGE_DIR;
	string contourYamlDirectory = CONTOUR_YAML_DIR;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	string strVisWordList;
	bool matchFound;
	string stringVWID;

	strVisWordList = "";
	
	for( int i = 0; i < contours.size(); i++ ) {
		
		matchFound  = false;
		
		Mat contourPointList(contours[i]);
		
		if (pointsDictionary.size() > 0) {
			
			event = "Starting Loop through dictionary.";
			helper.logEvent(event, 2, runInBackground, writelog); 

			for( int id = 0; id < pointsDictionary.size(); id++ ) {
				
				event = "In Loop Through Dictionary.";
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				
				Mat dictionaryPointList(pointsDictionary[id]);
				
				double dblRet = matchShapes(contourPointList, dictionaryPointList, 1, 0);
				string strRet = static_cast<ostringstream*>( &(ostringstream() << (dblRet)) )->str();
				event = "Comparison Return Value: " + strRet;
				helper.logEvent(event, 2, runInBackground, writelog); 
				
				
				
				
				if (dblRet <= epsilon) {
					stringVWID = static_cast<ostringstream*>( &(ostringstream() << (id + 1)) )->str();
					strVisWordList = strVisWordList + stringVWID + " ";
					matchFound = true;
				}	
				
				
			}
			
			
			if (!matchFound) {
				pointsDictionary.push_back(contours[i]);
				stringVWID = static_cast<ostringstream*>( &(ostringstream() << (pointsDictionary.size())) )->str();
				strVisWordList = strVisWordList + stringVWID + " ";
			}
			
		} else {
			pointsDictionary.push_back(contours[i]);
			stringVWID = static_cast<ostringstream*>( &(ostringstream() << (pointsDictionary.size())) )->str();
			strVisWordList = strVisWordList + stringVWID + " ";
			event = "Added First Set of points to dictionary by default.";
			helper.logEvent(event, 2, runInBackground, writelog); 
		}
		
		
	}
	
	return strVisWordList;
	
}


bool saveVisualWordFile(string fileconts, string filenamebase, string saveVisWordFileDir)  {
	
	Helper helper;
	string event;
	string evalDirectory = EVAL_DIR;
	string contourImageDirectory = CONTOUR_IMAGE_DIR;
	string contourYamlDirectory = CONTOUR_YAML_DIR;
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	
	bool retFlag = false;
	string filetoWrite = saveVisWordFileDir;
	filetoWrite.append(filenamebase);
	filetoWrite.append(".txt");
	int fileWriteRet = helper.writeTextFile(fileconts, filetoWrite);
	if (fileWriteRet > 0) {
		event = "Visual Word Document for " + filenamebase + " written successfully to " + filetoWrite + ".";
		helper.logEvent(event, 4, runInBackground, writelog);
		retFlag = true;
	} else {
		event = "Visual Word Document for " + filenamebase + " failed to write!";
		helper.logEvent(event, 0, runInBackground, writelog);
	}		
	
	return retFlag;
}

bool saveDictionary(vector<Moments> dictMoments, string filepath) {
	bool retFlag = false;
	
	return retFlag;
}
