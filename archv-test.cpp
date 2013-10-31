/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: archv-test.cpp
 *
 * Contains classes and functions for testing various aspects
 * of the BOW approach to image classification
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

#include "archv-test.h"

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
SurfFeatureDetector detector(300); // original value 500 documentation says somewhere between 300-500 is good depending on sharpness and contrast

//---dictionary size=number of cluster's centroids
int dictionarySize = 2000; // originally set to 1500
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
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
	cout<<"Vector quantization..."<<endl;
	
	// Call the collectclasscentroids function from imgextract.cpp
	collectclasscentroids(detector, extractor, bowTrainer, TRAINING_DIR);
	
	// Make a cv Matrix <Mat> to hold the descriptors from the training set
	// in this code this is loaded directly from the BOWTrainer that was just
	// produced.  I ultimately want to create the BOWTrainer, then extract the 
	// dictionary to some kind of file or, better yet, the db.  Then this function would
	// load the training data from the db.
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	
	// iterate through the Mat that holds the decriptors and do something
	// right now it is just counting them
	int count=0;
	for(vector<Mat>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++) {

		count+=iter->rows;

	}
	cout<<"Clustering "<<count<<" features"<<endl;
	
	
	
	//cluster the descriptors int a dictionary
	dictionary = bowTrainer.cluster();
	cout<<"Saving Dictionary File"<<endl;
	helper.WriteToFile("clustered_dictionary.yaml", dictionary, "clustered_dictionary");
	

	// set the dictionary for the BOW Description Extractor
	cout<<"Setting the dictionary for the BOWImgDescriptorExtractor (bowDE) that will be used for matching"<<endl;
	bowDE.setVocabulary(dictionary);
	
	
	/*
	 * Extracting histogram in the form of bow for each image 
	 * this section seems to go back through each of the items that was
	 * used to create the dictionary and then re-computes the dictionary 
	 * words for each based upon running it against the clustered 
	 * dictionary, then add these to a a new MAT.  There is a parallel
	 * MAT that contains the file names, pushed in the same order.
	 * I think this loop would, as such, produced the normaalized word
	 * description of each image in the training set.
	 */
	
	cout<<"extracting histograms in the form of BOW for each image "<<endl;

	/*
	 * getHistAndLabels function in imgextract.cpp file called to extract
	 * histograms of each image in the training set (based on dictionary) and
	 * store them in a Mat and also to store of list of labels that are
	 * diesignated by file name (in this case) and store them in a Mat
	 */
	vector<Mat> histLabelsVec =  getHistAndLabels(detector, bowDE, dictionarySize);
	Mat trainingData = histLabelsVec[0];
	Mat labels = histLabelsVec[1];

	cout<<"Saving Training Data MAT"<<endl;
	helper.WriteToFile("base_trainingdata.yaml", trainingData, "trainingData");
	cout<<"Saving Labels"<<endl;
	helper.WriteToFile("base_labels.yaml", labels, "labels");
	
	/* 
	 * testing getting just the hitograms returned into a Mat.  This is what
	 * I will do in the real implementation.  Once I get this Mat, I will use
	 * it to write visual word files for each image.
	 */
	string evalDir = EVAL_DIR;
	vector<string> collectionFiles;
	Mat collectionHistograms =  getHistograms(detector, bowDE, dictionarySize, collectionFiles, evalDir);
	cout<<"Saving Collection Histograms MAT"<<endl;
	helper.WriteToFile("collection_histograms.yaml", collectionHistograms, "collectionHistograms");
	
	/*
	 * testing converting the histogram to a string representation
	 */
	
	vector<float> histRow;
	string histogramsCSV;
	string vwd;
	
	int colVal;
	for( int yz = 0; yz < collectionHistograms.rows; yz++ )
	{
		histRow = collectionHistograms.row(yz);
		vwd = imagedoc.makeVWString(histRow, false, 4, collectionFiles[yz], histogramsCSV);
		string filetoWrite = COLLECTION_VW_FILES_DIR;
		filetoWrite.append(collectionFiles[yz]);
		filetoWrite.append(".txt");
		int fileWriteRet = helper.writeTextFile(vwd, filetoWrite);
		if (fileWriteRet > 0) {
			cout<<"Image Document for "<<collectionFiles[yz]<<" written successfully!"<<endl;
		} else {
			cout<<"Image Document for "<<collectionFiles[yz]<<" failed to write!"<<endl;
		}
	    histRow.clear();
	}

	
	/*
	 * SVM code below is used to train the "classes" of objects being found
	 * (ie, car, plane, etc.) based upon the training histograms.  This is the 
	 * part that I won't really need for my initial implementation
	 */
	
	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type=CvSVM::RBF;
	params.svm_type=CvSVM::C_SVC;
	params.gamma=0.50625000000000009;
	params.C=312.50000000000000;
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
	CvSVM svm;


	printf("%s\n","Training SVM classifier");
	bool res=svm.train(trainingData,labels,cv::Mat(),cv::Mat(),params);
	

	/*
	Mat c = abs(labels);
	cout<<"Mat C type: "<<c.type()<<endl;
	
	vector<double> labelsRowX;
	double labelValX;
	for( int yzxt = 0; yzxt < c.rows; yzxt++ )
	{
		labelsRowX = c.row(yzxt);
		labelValX = labelsRowX[0];
		cout<<"Label double"<<labelValX<<":"<<endl;
		labelsRowX.clear();
	}	
	*/
	
	
	/*
	 * Save the labels into a simple vector
	 */
	vector<float> labelsRow;
	vector<float> lables_stack;
	float labelVal;
	for( int yzx = 0; yzx < labels.rows; yzx++ )
	{
		labelsRow = labels.row(yzx);
		labelVal = labelsRow[0];
		lables_stack.push_back(labelVal);		
		labelsRow.clear();
	}
	
	
	cout<<"Processing evaluation data..."<<endl;
	int k=0;
	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, dictionarySize, CV_32FC1);
	k=0;
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;
	Mat results(0, 1, CV_32FC1);;
	
	
	vector<string> files = vector<string>();	
	
	helper.GetFileList(EVAL_DIR, files);

    for (unsigned int iz = 0;iz < files.size();iz++) {
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	
        	string sFileName = EVAL_DIR;
        	sFileName.append(files[iz]);
        	const char * imageName = sFileName.c_str ();
        	
			img2 = cvLoadImage(imageName,0);
			if (img2) {
				
				
				/*
				 * this call is to a function in imgextract.ccp that returns a number indicating the closest class match for the image.
				 */
				
				
				float matchClassResults = getClassMatch(detector, bowDE, img2, dictionarySize, sFileName, svm);
				
				cout << "Image " << sFileName << " belongs to class " << matchClassResults << endl;
				
				/*
				for (unsigned int izitz = 0;izitz < lables_stack.size();izitz++) {
					double matchClassResults = getClassMatch(detector, bowDE, img2, dictionarySize, sFileName, lables_stack[izitz], svm);
				}
				*/

				
				
				
				detector.detect(img2, keypoint2);
				bowDE.compute(img2, keypoint2, bowDescriptor2);
				
				string fileEvalDescriptors;
				fileEvalDescriptors.append(EVAL_DESCRIPTOR_DIR);
				fileEvalDescriptors.append(files[iz]);
				fileEvalDescriptors.append(".yaml");
				cout << "Saving Eval Descriptors Matrix for " << files[iz] << endl;
				helper.WriteToFile(fileEvalDescriptors, bowDescriptor2, "bowDescriptor2");
				
				
				
				/*
				 * note, the line "groundTruth.push_back((float) 5)" is here artificially
				 * assigning a class to check this image against as class 5.  The original
				 * code used file names both on the way in and on the way out as ways of
				 * assigning labels to classes.  What I would really need to do is for each
				 * evaluation image loop trough all classes a check against each one
				 * where the result is negative, it isn't in that class, and where positive
				 * it is.  Not exactly sure how to manage this with the groundTrough array.
				 */
				
				
				evalData.push_back(bowDescriptor2);
				groundTruth.push_back((float) 5);  //I've just hard coded a class to test againt, and the system is correctly predicting the error rate of evaluating againg this class
				float response = svm.predict(bowDescriptor2);
				results.push_back(response);
				
				
			}
			
			
        }
    }	

	//calculate the number of unmatched classes 
	double errorRate = (double) countNonZero(groundTruth- results) / evalData.rows;
	printf("%s%f","Error rate is ",errorRate);
	cout<<endl;
	return 0;

}


