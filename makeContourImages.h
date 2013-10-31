/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeContourImages.h
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


#ifndef MAKECONTOURIMAGES_H_
#define MAKECONTOURIMAGES_H_



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

void getShapePoints(vector<vector<Point> > contours) ;


#endif /*MAKECONTOURIMAGES_H_*/
