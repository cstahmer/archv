/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeContourDictionary.h
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


#ifndef MAKECONTOURDICTIONARY_H_
#define MAKECONTOURDICTIONARY_H_



#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>

#include "helper.cpp"


int main(int argc, char* argv[]);

Mat getShapes(vector<vector<Point> > contours, vector<Vec4i> hierarchy);

void getShapePoints(vector<vector<Point> > contours);

vector<Moments> getMoments(vector<vector<Point> > contours);

string getVisualWords(vector<Moments>& momentsDictionary, vector<Moments> imgMoments, int epsilong);

string matchContourShapes(vector<vector<Point> >& pointsDictionary, vector<vector<Point> > contours, double epsilon, int matchType);

bool saveVisualWordFile(string fileconts, string filenamebase, string saveVisWordFileDir);

bool saveDictionary(vector<Moments> dictMoments, string filepath);


#endif /*MAKECONTOURDICTIONARY_H_*/
