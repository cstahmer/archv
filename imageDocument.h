/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: imageDocument.h
 *
 * Contains classes and functions for converting an image visual 
 * word histogram to an image document format and saving the image
 * document to disk.
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

#ifndef IMAGE_DOC_H
#define IMAGE_DOC_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace std;


#endif

#pragma once
class ImgDoc
{

public:

	ImgDoc();
	~ImgDoc();

	/*
	 * This function takes a histogram for an image convertis it to a String representation.
	 * String is unordered with respect to the spatial orientation of objects in the original image.
	 * If boolean "fequency" is set to 1/on, then the number of times that a given word is added
	 * to the output string matches the number of times the histogram indicates that it appears
	 * in the image.  If the boolean "frequency" is set to false, then a visual word is added to
	 * output string only one time regardless of how many times the histogram indicates that it
	 * appears in the image.  the "depth" field tells affects the output a lot if frequency is set
	 * to true.  Histogram items are stored in scientific notation, which means they have to be
	 * multiplied by 10 to some power in order to make them whole numbers instead of decimals.
	 * The depth represents the exponent to multiply 10 to before multiplying the histogram
	 * number by (10 to the power of "level"). The hightes value that can be sent is 10.  
	 * I'll have to do some tweaking of the system in order to find out what works best.               
	 */
	string makeVWString(vector<float> histogram, bool frequency, int depth, string handle, string &histogramsCSV);
	
	string makeWeightedVWString(vector<float> histogram, bool frequency, int depth, string handle, string &histogramsCSV, int dictionarySize);
	
	void test();

};