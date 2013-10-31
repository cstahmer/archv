/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: visualWordGenerator.cpp
 *
 * Reads a library of images and outputs visual word documents
 * for the library.
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

#include "visualWordGenerator.h"

using namespace cv; 
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;


char ch[30];


//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
SurfFeatureDetector detector(100); // original value 500 documentation says somewhere between 300-500 is good depending on sharpness and contrast





//---dictionary size=number of cluster's centroids
int dictionarySize = 8000; // originally set to 1500
//TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
TermCriteria tc(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100000000, 0.000000001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);

Mat dictionary;


int main(int argc, char* argv[]) {
	
	

	int i,j;  //once I get everyrhing moved to other funcitons and classes, this can go away I think
	IplImage *img2;
	Helper helper;
	ImgDoc imagedoc;
	
	helper.StartClock();
	
	cout<<"Vector quantization..."<<endl;
	
	// Call the collectclasscentroids function from imgextract.cpp
	collectclasscentroids(detector, extractor, bowTrainer, TRAINING_DIR);
	
	//cout << "Temporary end of program.  Good Bye!" << endl;
	//return(0);
	
	// Make a cv Matrix <Mat> to hold the descriptors from the training set
	// in this code this is loaded directly from the BOWTrainer that was just
	// produced.  I ultimately want to create the BOWTrainer, then extract the 
	// dictionary to some kind of file or, better yet, the db.  Then this function would
	// load the training data from the db.
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	
	// get count of number of total features
	int count=0;
	for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++) {
		count+=iter->rows;
	}
	helper.PrintElapsedClock();
	cout<<"Clustering "<<count<<" features..."<<endl;
	
	
	
	//cluster the descriptors int a dictionary
	dictionary = bowTrainer.cluster();
	cout<<"Saving Dictionary File..."<<endl;
	helper.WriteToFile("clustered_dictionary.yaml", dictionary, "clustered_dictionary");
	
	helper.PrintElapsedClock();
	
	// set the dictionary for the BOW Description Extractor
	cout<<"Setting the dictionary for the BOWImgDescriptorExtractor (bowDE) that will be used for matching..."<<endl;
	bowDE.setVocabulary(dictionary);

	
	helper.PrintElapsedClock();
	
	/*
	 * getHistAndLabels function in imgextract.cpp file called to extract
	 * histograms of each image in the training set (based on dictionary) and
	 * store them in a Mat and also to store of list of labels that are
	 * diesignated by file name (in this case) and store them in a Mat
	 */
	vector<string> collectionFiles;
	string evalDir = EVAL_DIR;
	Mat collectionHistograms =  getHistograms(detector, bowDE, dictionarySize, collectionFiles, evalDir);
	cout << "collectionHistograms.rows: " << collectionHistograms.rows << endl;
	cout<<"Saving Collection Histograms MAT..."<<endl;
	helper.WriteToFile("collection_histograms.yaml", collectionHistograms, "collectionHistograms");
	
	helper.PrintElapsedClock();
	
	/*
	 * Convert the histogram of each image to a Visual Word document
	 * and write to file system
	 */
	
	cout << "Creating Visual Word Documents..." << endl;
	
	vector<float> histRow;
	string histogramsCSV;
	string vwd;
	
	int colVal;
	for( int yz = 0; yz < collectionHistograms.rows; yz++ )
	{
		histRow = collectionHistograms.row(yz);
		vwd = imagedoc.makeVWString(histRow, true, 4, collectionFiles[yz], histogramsCSV);
		string filetoWrite = COLLECTION_VW_FILES_DIR;
		filetoWrite.append(collectionFiles[yz]);
		filetoWrite.append(".txt");
		int fileWriteRet = helper.writeTextFile(vwd, filetoWrite);
		//cout << "VWFile for " << collectionFiles[yz] << endl << vwd << endl << endl;
		if (fileWriteRet > 0) {
			cout<<"   Image Document for "<<collectionFiles[yz]<<" written successfully!"<<endl;
		} else {
			cout<<"   Image Document for "<<collectionFiles[yz]<<" failed to write!"<<endl;
		}
	    histRow.clear();
	}

	cout << "Wrapping Up..." << endl;
	
	helper.StopAndPrintClock();
	
	cout << "Bye, Bye!" << endl;
	
	return 0;

}