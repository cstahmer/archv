#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include <iostream>
#include <fstream>
#include "helper.cpp"

using namespace cv;
using namespace std;

string defaultDetectorType = "SURF";
string defaultDescriptorType = "SURF";
string defaultMatcherType = "FlannBased";
//string defaultQueryImageName = "/web/sites/beeb/finished-woodcut-images/20159-30.jpg";
//string defaultEvalDir = "/web/sites/beeb/finished-woodcut-images/";
//string defaultDirToSaveResImages = "/web/sites/beeb/finished-woodcut-images-res/";
string defaultQueryImageName = "/web/sites/beeb/DiD/seed/Broken_r.jpg";
string defaultEvalDir = "/web/sites/beeb/DiD/stack/";
string defaultDirToSaveResImages = "/web/sites/beeb/DiD/examples/";

/// blur variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 3;
int MAX_FILES_TO_SEARCH = 0;

// make a helper object
Helper helper;
bool runInBackground = RUN_IN_BACKGROUND;
bool writelog = WRITE_LOG;

void doGaussianBlur(Mat src, Mat &dst) {
	
	 for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 ) { 
		 GaussianBlur( src, dst, Size( i, i ), 0, 0 );
	 }
	
}

void printPrompt() {
	cout << "/*\n"
	<< " * This is a sample on matching descriptors detected on one image to descriptors detected in image set.\n"
	<< " * So we have one query image and several train images. For each keypoint descriptor of query image\n"
	<< " * the one nearest train descriptor is found the entire collection of train images. To visualize the result\n"
	<< " * of matching we save images, each of which combines query and train image with matches between them (if they exist).\n"
	<< " * Match is drawn as line between corresponding points. Count of all matches is equel to count of\n"
	<< " * query keypoints, so we have the same count of lines in all set of result images (but not for each result\n"
	<< " * (train) image).\n"
	<< " */\n" << endl;

    cout << endl << "Format:" << endl;
    cout << "./matching_to_many_images -detector [detectorType] -descriptor [descriptorType] -matcher [matcherType] -image [queryImage] -match [fileWithTrainImages] -write [dirToSaveResImages] -maxfiles [max number of files to search against: 0 = all]" << endl;
    cout << endl;

    cout << "\nExample:" << endl
         << "./matching_to_many_images -detector " << defaultDetectorType << " -descriptor " << defaultDescriptorType << " -matcher" << defaultMatcherType << " -image "
         << defaultQueryImageName << " -match " << defaultEvalDir << " -write " << defaultDirToSaveResImages << " -maxfiles 250" << " -caption " << DELAY_CAPTION 
         << " -blur " << DELAY_BLUR << " -maxkernel " << MAX_KERNEL_LENGTH  << endl;
}


double maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask ) {
	double numMatches = 0;
	
	string event = "Building Match Mask...";
	helper.logEvent(event, 2, runInBackground, writelog);
	
	mask.resize( matches.size() );
    fill( mask.begin(), mask.end(), 0 );
    for( size_t i = 0; i < matches.size(); i++ ) {
        if( matches[i].imgIdx == trainImgIdx ) {
            mask[i] = 1;
            numMatches++;
        }
    }
    
    return numMatches;
}

void readTrainFilenames( string dirName, vector<string>& trainFilenames, int intMaxNumFilesToQuery ) {
	bool foundTrainingImages = false;
	
	// change this whole routing so that it reads the directory passed in dirName and returns a vector of filenames that are .jpg files
	
	vector<string> files = vector<string>();	
	helper.GetFileList(dirName, files);
	
	// here loop through the files vector checking for type jpg and if correct push to trainFilenames
	trainFilenames.clear();
	
	
	
	int numToProcess;
	if (intMaxNumFilesToQuery == 0) {
		numToProcess = files.size();
	} else {
		numToProcess = intMaxNumFilesToQuery;
	}
	int numProcessed = 0;
	for (unsigned int iz = 0;iz < files.size();iz++) {
		int isImage = helper.instr(files[iz], "jpg", 0, true);
		if (isImage > 0) {
			if (numProcessed < numToProcess) {
				trainFilenames.push_back(files[iz]);
				foundTrainingImages = true;
				numProcessed++;
			}
		}
	}
	
}

bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,
                                      Ptr<FeatureDetector>& featureDetector,
                                      Ptr<DescriptorExtractor>& descriptorExtractor,
                                      Ptr<DescriptorMatcher>& descriptorMatcher ) {
	
	string event = "Creating feature detector, descriptor extractor and descriptor matcher ...";
    helper.logEvent(event, 2, runInBackground, writelog);
    featureDetector = FeatureDetector::create( detectorType );
    descriptorExtractor = DescriptorExtractor::create( descriptorType );
    descriptorMatcher = DescriptorMatcher::create( matcherType );
    
    bool fdIsCreated = !(featureDetector.empty());
    bool deIsCreated = !(descriptorExtractor.empty());
    bool dmIsCreated = !(descriptorMatcher.empty());
    
    if ( !fdIsCreated ) {
    	event = "Could not create Feature Dector of type " + detectorType + ".";
    	helper.logEvent(event, 0, runInBackground, writelog);
    }
    if ( !deIsCreated ) {
    	event = "Could not create Descriptor Extractor of type " + descriptorType + ".";
    	helper.logEvent(event, 0, runInBackground, writelog);
    }
    if ( !dmIsCreated ) {
    	event = "Could not create Descriptor Matcher of type " + matcherType + ".";
    	helper.logEvent(event, 0, runInBackground, writelog);
    }
    


    bool isCreated = !( featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty() );

    return isCreated;
}

bool readImages( const string& queryImageName, const string& trainDirName, Mat& queryImage, vector <Mat>& trainImages, vector<string>& trainImageNames, int maxFilesToSearchAgainst ) {
	
	string event = "Reading Images...";
	helper.logEvent(event, 2, runInBackground, writelog);
	
    Mat origQueryImage = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
    queryImage = origQueryImage.clone();
    doGaussianBlur(origQueryImage, queryImage);
    if( queryImage.empty() )
    {
    	event = "Query Image " + queryImageName + " does not exist or you do not have the correct permissions!";
    	helper.logEvent(event, 0, runInBackground, writelog);
        return false;
    }
    
   
    readTrainFilenames( trainDirName, trainImageNames, maxFilesToSearchAgainst );
    if( trainImageNames.empty() )
    {
		event = "Unable to load evaluation images!";
	    helper.logEvent(event, 0, runInBackground, writelog);
        return false;
    }
    
    int readImageCount = 0;
    for( size_t i = 0; i < trainImageNames.size(); i++ ) {
        string filename = trainDirName + trainImageNames[i];
        Mat imgStart = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
        Mat img = imgStart.clone();
        doGaussianBlur(imgStart, img);
        if( img.empty() ) {
        	event = "Unable to read " + queryImageName + ".";
        	helper.logEvent(event, 0, runInBackground, writelog);
        } else{ 
            readImageCount++;
            trainImages.push_back( img );
        }
    }
    
    if( !readImageCount ) {
    	event = "Comparison image matrix empty. NO evaluation images read!";
    	helper.logEvent(event, 0, runInBackground, writelog);
        return false;
    } else {
    	string strReadImageCount = static_cast<ostringstream*>( &(ostringstream() << (readImageCount)) )->str();
    	event = "Loaded " + strReadImageCount + " evaluation images into image comparison matrix.";
    	helper.logEvent(event, 4, runInBackground, writelog);
    }

    return true;
}

void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, Ptr<FeatureDetector>& featureDetector ) {
	string event = "Extracting Query Image Keypoints...";
	helper.logEvent(event, 2, runInBackground, writelog);
    featureDetector->detect( queryImage, queryKeypoints );
	event = "Extracting Evaluation Image Keypoints...";
	helper.logEvent(event, 2, runInBackground, writelog);
    featureDetector->detect( trainImages, trainKeypoints );
}

void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
                         const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
                         Ptr<DescriptorExtractor>& descriptorExtractor ) {
	
	string event = "Computing Query Image Descriptors...";
	helper.logEvent(event, 2, runInBackground, writelog);
    descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
    event = "Computing Evaluation Image Descriptors...";
    helper.logEvent(event, 2, runInBackground, writelog);
    descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );
    
}

void matchDescriptors( const Mat& queryDescriptors, const vector<Mat>& trainDescriptors,
                       vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher ) {
 
	string event = "Adding evaluation descriptors to description matcher...";
	helper.logEvent(event, 2, runInBackground, writelog);
    descriptorMatcher->add( trainDescriptors );
    event = "Building Matches vector...";
    helper.logEvent(event, 2, runInBackground, writelog);
    descriptorMatcher->match( queryDescriptors, matches );
    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

}

void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints,
                       const vector<DMatch>& matches, const vector<string>& trainImagesNames, const string& resultDir ) {
	
	string event;
	double totalMatchesOnImage = 0;
	double numQueryKeypoints = queryKeypoints.size();
    Mat drawImg;
    vector<char> mask;
    for( size_t i = 0; i < trainImages.size(); i++ ) {
        if( !trainImages[i].empty() ) {
        	event = "Processing " + trainImagesNames[i] + "...";
        	helper.logEvent(event, 2, runInBackground, writelog);
        	totalMatchesOnImage = maskMatchesByTrainImgIdx( matches, i, mask );
            drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
                         matches, drawImg, Scalar::all(-1), Scalar::all(-1), mask );
            string filename = resultDir + "res_" + trainImagesNames[i];
            
            double percentMatch = (totalMatchesOnImage / numQueryKeypoints) * 100;
            string strTotalMatchesOnImage = static_cast<ostringstream*>( &(ostringstream() << (totalMatchesOnImage)) )->str();
            string strPercentMatch = static_cast<ostringstream*>( &(ostringstream() << (percentMatch)) )->str();
            event = "Matched " + strTotalMatchesOnImage + " keypoints in " + trainImagesNames[i] +": " + strPercentMatch + "%.";
            helper.logEvent(event, 2, runInBackground, writelog);
            
            if( !imwrite( filename, drawImg ) ) {
            	event = "Error Saving " + filename + ".";
            	helper.logEvent(event, 0, runInBackground, writelog);
            } else {
            	event = filename + "Successfully saved.";
            	helper.logEvent(event, 4, runInBackground, writelog);
            }
        }
    }
    
}

int main(int argc, char** argv) {
    string detectorType = defaultDetectorType;
    string descriptorType = defaultDescriptorType;
    string matcherType = defaultMatcherType;
    string queryImageName = defaultQueryImageName;
    string dirWithEvalImages = defaultEvalDir;
    string dirToSaveResImages = defaultDirToSaveResImages;
    string event;

    for (int i = 1; i < argc; i++) { 
    	string arument = argv[i];
        if (arument == "-detector") {
        	detectorType = argv[i + 1];
        }
        if (arument == "-descriptor") {
        	descriptorType = argv[i + 1];
        }
        if (arument == "-matcher") {
        	matcherType = argv[i + 1];
        }
        if (arument == "-image") {
        	queryImageName = argv[i + 1];
        }
        if (arument == "-match") {
        	dirWithEvalImages = argv[i + 1];
        }
        if (arument == "-write") {
        	dirToSaveResImages = argv[i + 1];
        }
        if (arument == "-log") {
        	writelog = true;
        }
        if (arument == "-back") {
        	runInBackground = true;
        }
        if (arument == "-maxfiles") {
        	string strMaxFiles = argv[i + 1];
        	MAX_FILES_TO_SEARCH = atoi(strMaxFiles.c_str());
        }   
        if (arument == "-caption") {
        	string strDelay = argv[i + 1];
        	DELAY_CAPTION = atoi(strDelay.c_str());
        } 
        if (arument == "-maxkernel") {
        	string strMaxKernal = argv[i + 1];
        	MAX_KERNEL_LENGTH = atoi(strMaxKernal.c_str());
        } 
        if (arument == "-blur") {
        	string strBlur = argv[i + 1];
        	DELAY_CAPTION = atoi(strBlur.c_str());
        } 
        if (arument == "-help") {
        	printPrompt();
        	exit(0);
        } 
    }
    
	event = "Starting matching_to_many_images execuatable.";
	helper.logEvent(event, 2, runInBackground, writelog);
    event = "Using Detector Type: " + detectorType;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Using Descriptor Type: " + descriptorType;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Using Matcher Type: " + matcherType;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Query Image: " + queryImageName;
    helper.logEvent(event, 2, runInBackground, writelog); 
    event = "Eval Directory: " + dirWithEvalImages;
    helper.logEvent(event, 2, runInBackground, writelog);
    string strQueryEventNumber;
    if (MAX_FILES_TO_SEARCH > 0) {
    	strQueryEventNumber = static_cast<ostringstream*>( &(ostringstream() << (MAX_FILES_TO_SEARCH)) )->str();
    } else {
    	strQueryEventNumber = "all";
    }
    event = "Searching against " + strQueryEventNumber + " images in the training directory.";
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Destination Directory for Match Images: " + dirToSaveResImages;
    helper.logEvent(event, 2, runInBackground, writelog);

    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;
    Ptr<DescriptorMatcher> descriptorMatcher;
    if( !createDetectorDescriptorMatcher( detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher ) )
    {
    	event = "Error Initializing Descriptor Extractor and Matching Objects.";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
    } else {
    	event = "Descriptor Extractor and all matching objects successfully instantiated.";
    	helper.logEvent(event, 4, runInBackground, writelog);
    }

    Mat queryImage;
    vector<Mat> trainImages;
    vector<string> trainImagesNames;
    if( !readImages( queryImageName, dirWithEvalImages, queryImage, trainImages, trainImagesNames,  MAX_FILES_TO_SEARCH) )   {
    	event = "Error reading images.";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
    } else {
    	event = "Query Image and Training Images successfully loaded.";
    	helper.logEvent(event, 4, runInBackground, writelog);    
    }
    
    vector<KeyPoint> queryKeypoints;
    vector<vector<KeyPoint> > trainKeypoints;
    detectKeypoints( queryImage, queryKeypoints, trainImages, trainKeypoints, featureDetector );
    event = "Keypoint extraction completed.";
    helper.logEvent(event, 4, runInBackground, writelog); 
    
    Mat queryDescriptors;
    vector<Mat> trainDescriptors;
    computeDescriptors( queryImage, queryKeypoints, queryDescriptors,
                        trainImages, trainKeypoints, trainDescriptors,
                        descriptorExtractor );
    event = "Descriptors successfully computed.";
    helper.logEvent(event, 4, runInBackground, writelog);  

    // START HERE
    
    
    event = "Beginning match process...";
    helper.logEvent(event, 2, runInBackground, writelog);
    vector<DMatch> matches;
    matchDescriptors( queryDescriptors, trainDescriptors, matches, descriptorMatcher );
    event = "Saving Results...";
    helper.logEvent(event, 2, runInBackground, writelog);
    saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints, matches, trainImagesNames, dirToSaveResImages );
    
    return 0;
}


