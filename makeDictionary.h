/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: make-Dictionary.h
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

#ifndef MAKEDICTIONARY_H_
#define MAKEDICTIONARY_H_

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


/*void collectclasscentroids();*/

int main(int argc, char* argv[]);


#endif /*MAKEDICTIONARY_H_*/

