/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: imageDocument.cpp
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

#include "imageDocument.h"
using namespace cv;
using namespace std;


ImgDoc::ImgDoc()
{
}


ImgDoc::~ImgDoc()
{
}


/* need to modify this so that it first reads the scientific notation as
 * a string and looks to see what the exponent is, then it should use
 * this exponent to figure out the multiplier in int multiplier = pow(10, depth);
 * Then, convert the scientific notation to an actual number.  Then create to files,
 * one for indexing, and one for querying.  The indexing file should use the "frequency"
 * kind of write out, where it repeats the word/number one time for every time it is
 * in the histogram.  The querying one shoudl construct a boolean query with AND
 * separators.  This might solve my problem with Lucene.
 */

string ImgDoc::makeVWString(vector<float> histogram, bool frequency, int depth, string handle, string &histogramsCSV) {
	string retString;
	int multiplier;
	string dictWordID;
	stringstream out;
	int wholeNumberHist = 0;
	vector<int> vectorExponents;
	int medianExponentValue;
	int exp;
	float significand;

	
	
	if (histogram.size() > 0) {	
		
/*
		cout << "Create Vector of Histograms" << endl;
		for(int iq=0; iq<histogram.size(); iq++){
			if (histogram[iq] != 0) {
				significand = frexp(histogram[iq], &exp);
				exp = exp * -1;
				if (exp > 0) {
					vectorExponents.push_back(exp);
				}	
			}
		}
		cout << "End Creating Vecor of Histograms" << endl;
*/		

		/*
		
		size_t n = vectorExponents.size() / 2;
		nth_element(vectorExponents.begin(), vectorExponents.begin()+n, vectorExponents.end());
		medianExponentValue = vectorExponents[n];
		
		*/
		
		//cout << handle << " ";
		//histogramsCSV.append(handle);
		//histogramsCSV.append(",");
		for(int i=0; i<histogram.size(); i++){
			out << i+1;
			dictWordID = out.str();
			if (histogram[i] > 0) {
				for (float fi=0; fi<histogram[i]; fi++) {
					//cout << "[" << i << ":" << fi << "]";
					retString.append(dictWordID);
					retString.append(" ");
				}
			}
			
			/*
			
			significand = frexp(histogram[i], &exp);
			if (significand > 0) {
				
				stringstream ssSig (stringstream::in | stringstream::out);
				stringstream ssExp (stringstream::in | stringstream::out);
				ssSig << significand;
				ssExp << exp;
				
				cout << significand << "e" << exp;
				histogramsCSV.append(ssSig.str());  //// errors here.  need to convert "significand" and "exp" to strings before I can assign them.
				histogramsCSV.append("e");
				histogramsCSV.append(ssExp.str());
			} else {
				cout << "0";
				histogramsCSV.append("0");
			}
			if (i < (histogram.size() - 1)) {
				cout << " ";
				histogramsCSV.append(",");
			}
			exp = exp * -1;
//			if (exp >= (medianExponentValue -1)) {
				wholeNumberHist =  (significand * pow(2, exp));
				for (int ix=0; ix<wholeNumberHist; ix++) {
					retString.append(dictWordID);
					retString.append(" ");
				}
//			}
			
			*/
			
			/*
			stringstream ss (stringstream::in | stringstream::out);
			ss << histogram[i];
			string test = ss.str();
			cout << test << " ";
			/*
			
			/*
			
			 if (histogram[i] >= 0) {
				 
				 wholeNumberHist = (int) ((histogram[i] * multiplier) + 0.5); 
			 } else {
				 
				 wholeNumberHist = (int) ((histogram[i] * multiplier) - 0.5);
			 }
			
			if (wholeNumberHist > 0) {
				if (frequency) {
					for (int ix=0; ix<wholeNumberHist; ix++) {
						retString.append(dictWordID);
						retString.append(" ");
					}
				} else {
					retString.append(dictWordID);
					retString.append(" ");	
				}
			}
			
			*/
			
			
			//clears the outstr before the next read in of ID
			out.str("");
				
		}
		
		//cout << endl;
		//histogramsCSV.append("\r\n");
		
	}

	
	return retString;
}


string ImgDoc::makeWeightedVWString(vector<float> histogram, bool frequency, int depth, string handle, string &histogramsCSV, int dictionarySize) {
	string retString;
	int multiplier;
	string dictWordID;
	stringstream out;
	int wholeNumberHist = 0;
	vector<int> vectorExponents;
	int medianExponentValue;
	int exp;
	float significand;

	
	
	if (histogram.size() > 0) {	
		
/*
		cout << "Create Vector of Histograms" << endl;
		for(int iq=0; iq<histogram.size(); iq++){
			if (histogram[iq] != 0) {
				significand = frexp(histogram[iq], &exp);
				exp = exp * -1;
				if (exp > 0) {
					vectorExponents.push_back(exp);
				}	
			}
		}
		cout << "End Creating Vecor of Histograms" << endl;
*/		

		/*
		
		size_t n = vectorExponents.size() / 2;
		nth_element(vectorExponents.begin(), vectorExponents.begin()+n, vectorExponents.end());
		medianExponentValue = vectorExponents[n];
		
		*/
		
		//cout << handle << " ";
		//histogramsCSV.append(handle);
		//histogramsCSV.append(",");
		int numberOfOccurrances = 0;
		for(int i=0; i<histogram.size(); i++){
			out << i+1;
			dictWordID = out.str();
			numberOfOccurrances = histogram[i] * dictionarySize;
			if (numberOfOccurrances > 0) {
				for (int fi=0; fi < numberOfOccurrances; fi++) {
					//cout << "[" << i << ":" << fi << "]";
					retString.append(dictWordID);
					retString.append(" ");
				}
			}
			
			/*
			
			significand = frexp(histogram[i], &exp);
			if (significand > 0) {
				
				stringstream ssSig (stringstream::in | stringstream::out);
				stringstream ssExp (stringstream::in | stringstream::out);
				ssSig << significand;
				ssExp << exp;
				
				cout << significand << "e" << exp;
				histogramsCSV.append(ssSig.str());  //// errors here.  need to convert "significand" and "exp" to strings before I can assign them.
				histogramsCSV.append("e");
				histogramsCSV.append(ssExp.str());
			} else {
				cout << "0";
				histogramsCSV.append("0");
			}
			if (i < (histogram.size() - 1)) {
				cout << " ";
				histogramsCSV.append(",");
			}
			exp = exp * -1;
//			if (exp >= (medianExponentValue -1)) {
				wholeNumberHist =  (significand * pow(2, exp));
				for (int ix=0; ix<wholeNumberHist; ix++) {
					retString.append(dictWordID);
					retString.append(" ");
				}
//			}
			
			*/
			
			/*
			stringstream ss (stringstream::in | stringstream::out);
			ss << histogram[i];
			string test = ss.str();
			cout << test << " ";
			/*
			
			/*
			
			 if (histogram[i] >= 0) {
				 
				 wholeNumberHist = (int) ((histogram[i] * multiplier) + 0.5); 
			 } else {
				 
				 wholeNumberHist = (int) ((histogram[i] * multiplier) - 0.5);
			 }
			
			if (wholeNumberHist > 0) {
				if (frequency) {
					for (int ix=0; ix<wholeNumberHist; ix++) {
						retString.append(dictWordID);
						retString.append(" ");
					}
				} else {
					retString.append(dictWordID);
					retString.append(" ");	
				}
			}
			
			*/
			
			
			//clears the outstr before the next read in of ID
			out.str("");
				
		}
		
		//cout << endl;
		//histogramsCSV.append("\r\n");
		
	}

	
	return retString;
}

void ImgDoc::test() {
	cout<<"I'm here"<<endl;
}
