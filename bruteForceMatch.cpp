#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"


#include <iostream>
#include <fstream>
#include "helper.cpp"

using namespace cv;
using namespace std;

//string defaultDetectorType = "SURF";
//string defaultDescriptorType = "SURF";
string defaultDetectorType = "FAST";
string defaultDescriptorType = "SURF";
string defaultMatcherType = "FlannBased";
string defaultQueryImageName = "/web/sites/beeb/DiD/seed/Broken_r.jpg";
string defaultEvalDir = "/web/sites/beeb/DiD/stack/";
string defaultDirToSaveResImages = "/web/sites/beeb/DiD/examples/";

/// blur variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 3;
int MAX_FILES_TO_SEARCH = 0;
int MATCH_DISTANCE_CONSTANT = 2; //Draw only "good" matches (i.e. whose distance is less than 2*min_dist )

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
         << defaultQueryImageName << " -match " << defaultEvalDir << " -write " << defaultDirToSaveResImages << " -maxfiles 250" <<endl;
}

double maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask ) {
	double numMatches = 0;
	
	//matches.clear();
	//mask.clear();
	
	string event = "Building Match Mask...";
	//helper.logEvent(event, 2, runInBackground, writelog);
	
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

bool readImages( const string& ImageName, Mat& Image ) {
	
	string event = "Reading " + ImageName + "...";
	//helper.logEvent(event, 2, runInBackground, writelog);
	
    Mat origQueryImage = imread( ImageName, CV_LOAD_IMAGE_GRAYSCALE);
    Image = origQueryImage.clone();
    doGaussianBlur(origQueryImage, Image);
    if( Image.empty() )
    {
    	event = "Image " + ImageName + " does not exist or you do not have the correct permissions!";
    	helper.logEvent(event, 0, runInBackground, writelog);
        return false;
    }

    return true;
}

void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Ptr<FeatureDetector>& featureDetector, string strType ) {
	string event = "Extracting " + strType + " Image Keypoints...";
	//helper.logEvent(event, 2, runInBackground, writelog);
    featureDetector->detect( queryImage, queryKeypoints );
}

void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors, Ptr<DescriptorExtractor>& descriptorExtractor, string strType ) {
	
	string event = "Computing " + strType + " Image Descriptors...";
	//helper.logEvent(event, 2, runInBackground, writelog);
    descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
    
}

void matchDescriptors( const Mat& queryDescriptors, const Mat& trainDescriptors,
                       vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher ) {
 
	
	vector<Mat> evalDescriptors;
	evalDescriptors.push_back(trainDescriptors);
	string event = "Adding evaluation descriptors to description matcher...";
	//helper.logEvent(event, 2, runInBackground, writelog);
    descriptorMatcher->add( evalDescriptors );
    event = "Building Matches vector...";
    //helper.logEvent(event, 2, runInBackground, writelog);
    descriptorMatcher->match( queryDescriptors, matches );
    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

}

void simpleMatch(Mat& queryDescriptors, Mat& evalDescriptors, vector<DMatch>& matches) {
	FlannBasedMatcher matcher;
	matcher.match(queryDescriptors, evalDescriptors, matches);
}

void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const Mat& trainImages, const vector<KeyPoint>& trainKeypoints,
                       const vector<DMatch>& matches, const string trainImagesNames, const string& resultDir ) {
	
	string event;
	double totalMatchesOnImage = 0;
	double numQueryKeypoints = queryKeypoints.size();
    Mat drawImg;
    vector<char> mask;
 //   for( size_t i = 0; i < trainImages.size(); i++ ) {
 //       if( !trainImages[i].empty() ) {
        	event = "Processing " + trainImagesNames + "...";
        	//helper.logEvent(event, 2, runInBackground, writelog);
        	totalMatchesOnImage = maskMatchesByTrainImgIdx( matches, 0, mask );
            drawMatches( queryImage, queryKeypoints, trainImages, trainKeypoints,
                         matches, drawImg, Scalar::all(-1), Scalar::all(-1), mask );
            string filename = resultDir + "res_" + trainImagesNames;
            
            double percentMatch = (totalMatchesOnImage / numQueryKeypoints) * 100;
            string strTotalMatchesOnImage = static_cast<ostringstream*>( &(ostringstream() << (totalMatchesOnImage)) )->str();
            string strPercentMatch = static_cast<ostringstream*>( &(ostringstream() << (percentMatch)) )->str();
            event = "Matched " + strTotalMatchesOnImage + " keypoints in " + trainImagesNames +": " + strPercentMatch + "%.";
            event = "Matched \t" + trainImagesNames +"\t" + strPercentMatch + "\t %.";
            helper.logEvent(event, 2, runInBackground, writelog);
            
            if( !imwrite( filename, drawImg ) ) {
            	event = "Error Saving " + filename + ".";
            	//helper.logEvent(event, 0, runInBackground, writelog);
            } else {
            	event = filename + "Successfully saved.";
            	//helper.logEvent(event, 4, runInBackground, writelog);
            }
//        }
 //   }
    
}


void saveResultImagesTwo( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
                       const Mat& evalImages, const vector<KeyPoint>& evalKeypoints,
                       const vector<DMatch>& matches, const string evalImagesName, const string& resultDir ) {
	
	string event;
    Mat drawImg;
    drawMatches(queryImage, queryKeypoints, evalImages, evalKeypoints, matches, drawImg);
    string filename = resultDir + "res_" + evalImagesName;
    if( !imwrite( filename, drawImg ) ) {
    	event = "Error Saving " + filename + ".";
    	//helper.logEvent(event, 0, runInBackground, writelog);
    } else {
    	event = filename + "Successfully saved.";
    	//helper.logEvent(event, 4, runInBackground, writelog);
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
        if (arument == "-distance") {
        	string strBlur = argv[i + 1];
        	MATCH_DISTANCE_CONSTANT = atoi(strBlur.c_str());
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

   
    
    // do query image
    Mat queryImage;
    if( !readImages( queryImageName, queryImage) )   {
    	event = "Error reading images.";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
    } else {
    	event = "Query Image successfully loaded.";
    	helper.logEvent(event, 4, runInBackground, writelog);    
    }
    
    vector<KeyPoint> queryKeypoints;
    detectKeypoints( queryImage, queryKeypoints, featureDetector, "Query" );
    event = "Query Keypoint extraction completed.";
    helper.logEvent(event, 4, runInBackground, writelog); 
    
    Mat queryDescriptors;
    computeDescriptors( queryImage, queryKeypoints, queryDescriptors, descriptorExtractor, "Query" );
    event = "Query Descriptors successfully computed.";
    helper.logEvent(event, 4, runInBackground, writelog);  

    // NOW LOOP THROUGH EVAL FILES AND CALCULATE MATCH FOR EACH ON
    vector<string> evalImagesNames;
    readTrainFilenames( dirWithEvalImages, evalImagesNames, MAX_FILES_TO_SEARCH );
    
    for( size_t i = 0; i < evalImagesNames.size(); i++ ) {
    	
    	
    	//build the eval image objects
    	
        string filename = dirWithEvalImages + evalImagesNames[i];
        
        Mat evalImage;
        if( !readImages( filename, evalImage) )   {
        	event = "Error reading image.";
    	    helper.logEvent(event, 0, runInBackground, writelog);
    	    event = "Operation Aborted!";
    	    helper.logEvent(event, 0, runInBackground, writelog);
    		exit(0);
        } else {
        	event = "Eval Image successfully loaded.";
        	//helper.logEvent(event, 4, runInBackground, writelog);    
        }
        
        vector<KeyPoint> evalKeypoints;
        detectKeypoints( evalImage, evalKeypoints, featureDetector, "Eval" );
        event = "Eval Keypoint extraction completed.";
        //helper.logEvent(event, 4, runInBackground, writelog); 
        
        Mat evalDescriptors;
        computeDescriptors( evalImage, evalKeypoints, evalDescriptors, descriptorExtractor, "Eval" );
        event = "Eval Descriptors successfully computed.";
        //helper.logEvent(event, 4, runInBackground, writelog);  
        
        // now do the match
        event = "Beginning match process...";
        //helper.logEvent(event, 2, runInBackground, writelog);
        vector<DMatch> matches;
        simpleMatch(queryDescriptors, evalDescriptors, matches);
        
        //calculate max/min distances
        double max_dist = 0; double min_dist = 100;
        for( int imd = 0; imd < queryDescriptors.rows; imd++ ) { 
        	double dist = matches[imd].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        //build match set of "goox" matches
        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
        //-- PS.- radiusMatch can also be used here.
        std::vector< DMatch > good_matches;
        double totalGoodMatches = 0;
        cout << "Building Match Points" << endl;
        for( int igm = 0; igm < queryDescriptors.rows; igm++ ) { 
        	int qIndex = matches[igm].queryIdx;
        	int eIndex = matches[igm].trainIdx;
        	int imgIndex = matches[igm].imgIdx;
        	double ddistance = matches[igm].distance;
        	cout << "Match Slice " << igm << "qIndex: [" << qIndex << "] eIndex: [" << eIndex << "] imgIndex: [" << imgIndex << "] Distance: [" << ddistance << "]" << endl;
        	if( matches[igm].distance == 0 ) { 
        	//if( matches[igm].distance > (MATCH_DISTANCE_CONSTANT * min_dist) ) { 
        		good_matches.push_back( matches[igm]); 
        		totalGoodMatches++;
        	}
        }
        cout << "END Building Match Points" << endl;
        
    	double numQueryKeypoints = queryKeypoints.size();
    	double percentMatch = (totalGoodMatches / numQueryKeypoints) * 100;
    	string strTotalMatchesOnImage = static_cast<ostringstream*>( &(ostringstream() << (totalGoodMatches)) )->str();
    	string strPercentMatch = static_cast<ostringstream*>( &(ostringstream() << (percentMatch)) )->str();
    	//event = "Matched " + strTotalMatchesOnImage + " keypoints in " + trainImagesNames +": " + strPercentMatch + "%.";
    	event = "Matched \t" + evalImagesNames[i] +"\t" + strPercentMatch + "\t %.";
    	helper.logEvent(event, 2, runInBackground, writelog);
        
        
        //matchDescriptors( queryDescriptors, evalDescriptors, matches, descriptorMatcher );
        event = "Saving Results...";
        //helper.logEvent(event, 2, runInBackground, writelog);
        
        saveResultImagesTwo( queryImage, queryKeypoints, evalImage, evalKeypoints, good_matches, evalImagesNames[i], dirToSaveResImages );
        
    }   
    
    
    return 0;
}