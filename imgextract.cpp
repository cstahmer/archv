/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: imageextract.cpp
 *
 * Contains classes and functions for extracting descriptor
 * information from images
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

#include "imgextract.h"

using namespace std;
using namespace cv;

using std::vector;


void collectclasscentroids(SurfFeatureDetector &detector, Ptr<DescriptorExtractor> &extractor, BOWKMeansTrainer &bowTrainer, string trainingDir, bool runInBackground, bool writelog, bool saveKeypointFile, string keypointFilePath, double dblSizeFilter, double dblResponseFilter) {
	
	IplImage *img;
	vector<string> files = vector<string>();
	Helper helper;
	string event;
	char ch[30];
	
	// should put error correction here to check if directory exists
	
	helper.GetFileList(trainingDir, files);
	
	// randomize the files vector to get a good random sample of impressions out of the very large library
	random_shuffle ( files.begin(), files.end() );
	
	int intIterNum = 0;
	int intIterMax = 500;

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	
        	if (intIterNum < intIterMax) {
	        	string sFileName = trainingDir;
	        	string sFeaturesDir = "/usr/local/share/archive-vision/build/features/";
	        	string sOutputImageFilename = keypointFilePath;
	        	sFileName.append(files[iz]);
	        	sOutputImageFilename.append(files[iz]);
	        	sFeaturesDir.append(files[iz]);
	        	sFeaturesDir.append(".txt");
	        	const char * imageName = sFileName.c_str ();
	        	
				img = cvLoadImage(imageName,0);
				
				
				if (img) {
					string workingFile = files[iz];
					vector<KeyPoint> keypoint;
					detector.detect(img, keypoint);
					
					//calculate average size and response value for image keypoints
					//vector<double> keypointSize;
					double keypointSize = 0;
					double keypointResult = 0;
					for( int icc = 0; icc < keypoint.size(); icc++ ) {
						keypointSize = keypointSize + keypoint[icc].size;
						keypointResult = keypointResult + keypoint[icc].response;
						//keypointSize.push_back(fabs(keypoint[icc].size));
						//keypointResult.push_back(fabs(keypoint[icc].response));
		
					}
					
					event = "Filtering Keypoints for " + workingFile;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					string strOrigKeypointCount = static_cast<ostringstream*>( &(ostringstream() << (keypoint.size())) )->str();
					event = "Starting number of keypoints: " + strOrigKeypointCount;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					// float keypoint_size_average = accumulate( keypointSize.begin(), keypointSize.end(), 0.0 ) / keypointSize.size();
					double keypoint_size_average = keypointSize / keypoint.size();
					string strAvgKeypointSize = static_cast<ostringstream*>( &(ostringstream() << (keypoint_size_average)) )->str();
					event = "Average size of keypoints: " + strAvgKeypointSize;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					
					string strAreaCheckVal = static_cast<ostringstream*>( &(ostringstream() << (dblSizeFilter)) )->str();
					event = "Eliminating keypoints with size < " + strAreaCheckVal;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					vector<KeyPoint> filteredKeypoints;
					for( int ica = 0; ica < keypoint.size(); ica++ ) {
						double areaCheckVal = fabs(keypoint[ica].size);
						
						/*
						if (ica < 20) {
							string strSampleKeypointSize = static_cast<ostringstream*>( &(ostringstream() << (areaCheckVal)) )->str();
							event = "Sample Keypoint Size: " + strSampleKeypointSize;
							helper.logEvent(event, 2, runInBackground, writelog);
						}
						*/
						
						if (areaCheckVal > dblSizeFilter) {
							filteredKeypoints.push_back(keypoint[ica]);
						}
					}
					
		
					string strFilteredKeypointCount = static_cast<ostringstream*>( &(ostringstream() << (filteredKeypoints.size())) )->str();
					event = "Number of Keypoins after applying size filter: " + strFilteredKeypointCount;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					
					keypoint = filteredKeypoints;
					
					
					// float keypoint_response_average = accumulate( keypointResult.begin(), keypointResult.end(), 0.0 )/ keypointResult.size();
					double keypoint_response_average = keypointResult / keypoint.size();
					string strAvgKeypointResponse = static_cast<ostringstream*>( &(ostringstream() << (keypoint_response_average)) )->str();
					event = "Average keypoint result respose value: " + strAvgKeypointResponse;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					string strRespCheckVal = static_cast<ostringstream*>( &(ostringstream() << (dblResponseFilter)) )->str();
					event = "Eliminating keypoints with response value < " + strRespCheckVal;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					
					vector<KeyPoint> filteredRespKeypoints;
					for( int icad = 0; icad < keypoint.size(); icad++ ) {
						double respVal = fabs(keypoint[icad].response);
						
						/*
						if (icad < 20) {
							string strSampleKeypointResponse = static_cast<ostringstream*>( &(ostringstream() << (respVal)) )->str();
							event = "Sample Keypoint Response: " + strSampleKeypointResponse;
							helper.logEvent(event, 2, runInBackground, writelog);
						}
						*/
						
						if (respVal > dblResponseFilter) {
							filteredRespKeypoints.push_back(keypoint[icad]);
						}
					}
					
					
					//string strFilteredRespKeypointCount = static_cast<ostringstream*>( &(ostringstream() << (filteredRespKeypoints.size())) )->str();
					//event = "Number of keypoints after applying response filter: " + strFilteredRespKeypointCount;
					//helper.logEvent(event, 2, runInBackground, writelog);
					
					keypoint = filteredRespKeypoints;
					
					string strNewKeypointCount = static_cast<ostringstream*>( &(ostringstream() << (keypoint.size())) )->str();
					event = "Final Filtered Keypoints to Process: " + strNewKeypointCount;
					helper.logEvent(event, 2, runInBackground, writelog);
					
					if (keypoint.size()) {
						Mat features;
						extractor->compute(img, keypoint, features);
						
						event = "Processing " + workingFile;
						helper.logEvent(event, 2, runInBackground, writelog);
		
						
						if (saveKeypointFile) {
							Mat output;
							drawKeypoints(img, keypoint, output, Scalar(0, 128, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
							imwrite(sOutputImageFilename, output);
						}
						
		
						
						
						// try writing out all the feature, each to its own YML file and see what
						// they look like 
		//				helper.WriteToFile(sFeaturesDir, features, "features");
						
						bowTrainer.add(features);
					} else {
						event = workingFile + "contains no keypoints.";
						helper.logEvent(event, 1, runInBackground, writelog);
					}
				}
				
				intIterNum++;
        	}
        }
    }
	return;
}


/*
 * Note, if I want to class images, I need to do some work on how lables are generated.  Perhaps lookup
 * in a db a class (woodcut_group) for each file name and submit this as the label
 */

vector<Mat> getHistAndLabels(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, int dictionarySize,  double dblSizeFilter, double dblResponseFilter) {
	
	// setup variable and object I need
	IplImage *img2;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, dictionarySize, CV_32FC1);
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Helper helper;
	vector<string> files = vector<string>();
	
	helper.GetFileList(EVAL_DIR, files);

	float labelVal;

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	string sFileName = TRAINING_DIR;
        	sFileName.append(files[iz]);
        	const char * imageName = sFileName.c_str ();
        	
			img2 = cvLoadImage(imageName,0);
			if (img2) {
				detector.detect(img2, keypoint1);
				
				// filter keypoints here
				vector<KeyPoint> filteredKeypoints;
				for( int ica = 0; ica < keypoint1.size(); ica++ ) {
					double areaCheckVal = fabs(keypoint1[ica].size);				
					if (areaCheckVal > dblSizeFilter) {
						filteredKeypoints.push_back(keypoint1[ica]);
					}
				}
				keypoint1 = filteredKeypoints;
				
				vector<KeyPoint> filteredRespKeypoints;
				for( int icad = 0; icad < keypoint1.size(); icad++ ) {
					double respVal = fabs(keypoint1[icad].response);
					if (respVal > dblResponseFilter) {
						filteredRespKeypoints.push_back(keypoint1[icad]);
					}
				}
				keypoint1 = filteredRespKeypoints;

				bowDE.compute(img2, keypoint1, bowDescriptor1);
				trainingData.push_back(bowDescriptor1);
				labelVal = iz+1;
				labels.push_back(labelVal);
			}
			
			
        }
    }	
	
	vector<Mat> retVec;
	retVec.push_back(trainingData);
	retVec.push_back(labels);
	return retVec;
	
}


Mat getHistograms(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, int dictionarySize, vector<string> &collectionFilenames, string evalDir) {
	
	// setup variable and object I need
	IplImage *img2;
	Mat trainingData(0, dictionarySize, CV_32FC1);
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Helper helper;
	vector<string> files = vector<string>();	
	
	helper.GetFileList(evalDir, files);
	
	cout << "Number of Collection Files to Process: " << files.size()-2 << endl;

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	cout << "     Processing " << files[iz] << endl;
        	
        	collectionFilenames.push_back(files[iz]);
        	string sFileName = EVAL_DIR;
        	sFileName.append(files[iz]);
        	const char * imageName = sFileName.c_str ();
        	
			img2 = cvLoadImage(imageName,0);
			if (img2) {
				detector.detect(img2, keypoint1);
				bowDE.compute(img2, keypoint1, bowDescriptor1);
				trainingData.push_back(bowDescriptor1);
			}
			
			
        }
    }	
	
	return trainingData;	
}
Mat getSingleImageHistogram(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, string evalFile, double dblSizeFilter, double dblResponseFilter) {
	
	// setup variable and object I need
	IplImage *img2;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	Helper helper;

    
	int isImage = helper.instr(evalFile, "jpg", 0, true);
    if (isImage > 0) {
 
    	const char * imageName = evalFile.c_str ();
		img2 = cvLoadImage(imageName,0);
		if (img2) {
			detector.detect(img2, keypoint1);
			
			// filter keypoints here
			vector<KeyPoint> filteredKeypoints;
			for( int ica = 0; ica < keypoint1.size(); ica++ ) {
				double areaCheckVal = fabs(keypoint1[ica].size);				
				if (areaCheckVal > dblSizeFilter) {
					filteredKeypoints.push_back(keypoint1[ica]);
				}
			}
			keypoint1 = filteredKeypoints;
			
			vector<KeyPoint> filteredRespKeypoints;
			for( int icad = 0; icad < keypoint1.size(); icad++ ) {
				double respVal = fabs(keypoint1[icad].response);
				if (respVal > dblResponseFilter) {
					filteredRespKeypoints.push_back(keypoint1[icad]);
				}
			}
			keypoint1 = filteredRespKeypoints;
			
			
			
			
			bowDE.compute(img2, keypoint1, bowDescriptor1);
		}	
    }	
	
	return bowDescriptor1;	
}


float getClassMatch(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, IplImage* &img2, int dictionarySize, string sFileName, CvSVM &svm) {
	float response;
	
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;
	Mat evalData(0, dictionarySize, CV_32FC1);
	Mat groundTruth(0, 1, CV_32FC1);
	Mat results(0, 1, CV_32FC1);
	
	
	detector.detect(img2, keypoint2);
	bowDE.compute(img2, keypoint2, bowDescriptor2);
	
	
	//evalData.push_back(bowDescriptor2);
	//groundTruth.push_back((float) classID);
	response = svm.predict(bowDescriptor2);
	//results.push_back(response);

	
	return response;
}