/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: imageextract.h
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

#ifndef IMGEXTRACT_
#define IMGEXTRACT_


#include "cv.h" 
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

/*
 * Pass this funciton a SurfFeatureDetector, a Descriptor Extractor, and an BOWKMeansTrainer (by reference) and 
 * it extracts descriptors from the training set into the BOWKMeansTtrainer.
 */
void collectclasscentroids(SurfFeatureDetector &detector, Ptr<DescriptorExtractor> &extractor, BOWKMeansTrainer &bowTrainer, string trainingDir, bool runInBackground, bool writelog, bool saveKeypointFile, string keypointFilePath, double dblSizeFilter, double dblResponseFilter);

/*
 * Pass this function a SurfFeatureDector, DescriptorExtractor, and a constant designating the size of the vocabulary
 * and it returns two Mats (packaged in a vector).  One contains a collection of histograms for each of the images
 * in the training set, and the other contains a colleciton of class labels generated from file names.
 */
vector<Mat> getHistAndLabels(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, int dictionarySize,  double dblSizeFilter, double dblResponseFilter);

/*
 * This function is similar to the one above, but it returns a Mat of image histograms and puts the filenames by reference in a passed vector.
 */
Mat getHistograms(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, int dictionarySize, vector<string> &collectionFilenames, string evalDir);

/*
 * This function is similar to the one above, but it returns the histogram of single image.
 */
Mat getSingleImageHistogram(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, string evalFile, double dblSizeFilter, double dblResponseFilter);

/*
 * function that tests an eval image against the learned SVM class list and returns closest class match
 */

float getClassMatch(SurfFeatureDetector &detector, BOWImgDescriptorExtractor &bowDE, IplImage* &img2, int dictionarySize, string sFileName, CvSVM &svm);

#endif /*IMGEXTRACT_*/