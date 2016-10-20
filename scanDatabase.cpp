/* ===============================================================================================
   scanDatabase.cpp		          	Version 2               10/16/2016               		Arthur Koehl

   This program reads in an image file, the directory of images to compare it with, the keypoint
   files of those image (the directory), an output image file, as well as the parameters that 
   made those keypoint files. Then, it finds the matches, filters those matches using homography
   (the ratio, symmetry and ransac tests) and displays the best three matches and their Distance
   (the number refering to the remaining number (higher numbers being better)).

   =============================================================================================== */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <fstream>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include <algorithm>  // for sort algorithm
using namespace cv;
using namespace std;


int  usage();
void read_flags(int argc, char** argv, string *imgfile, string *imgdir, string *infodir, string *output, string *param);
void read_surfparams(string param, int *minh, int *octaves, int *layers, int *sizemin, double *responsemin);
int GetFileList(string directory, vector<string> &files);

void filter_keypoints (vector<KeyPoint>& keypoints, int sizemin, double responsemin);
void showkeypts(vector<KeyPoint>& keypoints, Mat& drawImg);

int ratioTest(vector<vector<cv::DMatch> > &matches, double ratio);
void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2,vector<DMatch>& symMatches);
Mat ransacTest(const vector<cv::DMatch>& matches,const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& outMatches);

Mat CombineImages(int nimage, Mat images[], string titles[]);
vector<int> ordered(vector<double> const & values, int nval);


int main(int argc, char** argv)
{

/* ===============================================================================================
   Show usage if needed
   =============================================================================================== */
  if (argc < 2)
    return usage();

  string input = argv[1];
  if( input == "-h" || input == "-help" )
    return usage();


/* ===============================================================================================
   (1) Initialize all varaibles and surf parameters (2) parse command line (3) read in parameters 
   =============================================================================================== */
  string imgfile, imgdir, infodir, output, param;
  int minh = 2000;
  int octaves = 8;
  int layers = 8;
  int sizemin = 50;
  double responsemin = 100;
  double scale = 1;
  double ratio = 0.8;

  read_flags (argc, argv, &imgfile, &imgdir, &infodir, &output, &param);

  read_surfparams (param, &minh, &octaves, &layers, &sizemin, &responsemin);

/* ===============================================================================================
   Create all structures that are needed to process the images:
        - use SURF for key points detection and feature extraction
        - use BFMatcher (Brute force matcher) for assigning a point to its two nearest
          neighbours in another image
   =============================================================================================== */
  SurfFeatureDetector detector( minh, octaves, layers );  // Define SURF detector

  Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor(); // Feature extractor from keypoints


  vector<KeyPoint> keypoints1;                                         // Vector of keypoints
  vector<KeyPoint> keypoints2;                                         // Vector of keypoints

  Mat descriptors1, descriptors2;

  BFMatcher matcher;

  vector < vector<DMatch> > matches1;
  vector < vector<DMatch> > matches2;
  vector <DMatch> sym_matches;
  vector <DMatch> matches;

/* ===============================================================================================
   Process image:
	- detect keypoints
	- filter keypoints
   =============================================================================================== */
  Mat img1;

  img1 = imread(imgfile);

  detector.detect( img1, keypoints1);
  filter_keypoints (keypoints1, sizemin, responsemin);
  extractor->compute(img1, keypoints1, descriptors1);

/* ===============================================================================================
   Go to directory containing images, check it exists, get file list, select those files
   that contain an image (.jpg extension).
   =============================================================================================== */
  struct stat sb;

  const char * dirName;
  dirName = imgdir.c_str ();
  if (stat(dirName, &sb) != 0 || !S_ISDIR(sb.st_mode))
  {
	  cout << dirName << " does not exist, or is not a directory; try again"<<endl;
    return -1;
   }

  vector<string> allfiles = vector<string>();                 // vector containing names of all files

  int ierr = GetFileList(imgdir, allfiles);
  if (ierr != 0) 
  {
	  cout << " Problem while trying to read in list of histograms; check the directory!" <<endl;
    return -1;
  }

  vector<string> files;
  string filename;
  int counter = 0;

  for(int i = 0; i < allfiles.size(); i++)
  {
	  filename = allfiles[i];
	  if(filename.substr(filename.find_last_of(".") + 1) == "jpg")
	  {
		  files.push_back(filename);
		  counter++;
	  }
  }

/* ===============================================================================================
   Now loop over all image files:
   =============================================================================================== */
  vector<double> distval(files.size());
  vector<int>    indices(files.size());

  string infofile;
  int pos;

  counter = 0;
  int ndist = 0;

  for(int i = 0; i < files.size(); i++)
  {
/*      =========================================================================================
        Generate filenames
        ========================================================================================== */
	  if((counter+1) % 100 ==0 || counter == files.size()-1) cout << "Processing image # " << counter+1 << " out of " << files.size() << " images in the database"<<endl;

	  infofile = files[i];
    pos = 0;
    pos = infofile.find("jpg",pos);
    infofile.replace(pos,3,"yml");
    filename = infodir;
    if(*filename.rbegin() != '/') filename.append("/");
    filename.append(infofile);

/*      =========================================================================================
        Read keypoints from file
        ========================================================================================== */
  	keypoints2.clear();

	  FileStorage fs(filename, FileStorage::READ);
	  FileNode kptFileNode = fs["keypoints"];
	  read( kptFileNode, keypoints2 );

/*      =========================================================================================
        Process image2
        ========================================================================================== */
    if(keypoints1.size()>0 && keypoints2.size() > 0 ) 
    {

   	  fs["descriptors"] >> descriptors2;

/*  	=========================================================================================
    	Find matches based on descriptors: 
	- first from img1 to img2 (with 2 NN), then from img2 to img1
	- filter based on ratio test
	- filter for symmetry
	- filter by RANSAC
    	======================================================================================== */

		  matches1.clear();
		  matches2.clear();
 		  matcher.knnMatch(descriptors1,descriptors2,matches1,2);
 		  matcher.knnMatch(descriptors2,descriptors1,matches2,2);

 		  int removed= ratioTest(matches1,ratio);
 		  removed= ratioTest(matches2,ratio);

		  sym_matches.clear();
		  matches.clear();

 		  symmetryTest(matches1,matches2,sym_matches);

 		  Mat fundamental= ransacTest(sym_matches,keypoints1,keypoints2, matches);
		  distval[counter] = matches.size();
	 }

	  else
		  distval[counter] = 0;

	  fs.release();
	  counter++;
  }
    ndist = counter;

/* ===============================================================================================
   Sort array of distances: get indices of sorted values
   =============================================================================================== */
    indices = ordered(distval,ndist);

/* ===============================================================================================
   Recover the top three images
   =============================================================================================== */
   int nimg = 4;
   int nimgmax = 10;

   Mat image;
   Mat drawImg[nimgmax];
   Mat scaledImg[nimgmax];

   string imfile;
   string titles[nimgmax];
   stringstream str[nimgmax];

   img1.copyTo(drawImg[0]);
   str[0] << "Input Image: " << imgfile.substr(0,imgfile.find_last_of(".")) ;
   getline(str[0],titles[0]);

   int idx;
   double dist;
   for(int i=0; i < 3; i++)
   {

	  idx = indices[i];
	  dist = distval[idx];

	  imfile = files[idx];
	  filename = imgdir;
	  if(*filename.rbegin() != '/') filename.append("/");
	  filename.append(imfile);
	  image=imread(filename.c_str());
	  image.copyTo(drawImg[i+1]);

 	  str[i+1] << "Image: " << files[idx].substr(0,files[idx].find_last_of(".")) << " dist = " << dist;
 	  getline(str[i+1],titles[i+1]);
   }


/* ===============================================================================================
   Now combine all images into a single, large one
   =============================================================================================== */
  int interpolation=INTER_LINEAR;
  resize(drawImg[0],scaledImg[0], Size(),scale,scale,interpolation);
  resize(drawImg[1],scaledImg[1], Size(),scale,scale,interpolation);
  resize(drawImg[2],scaledImg[2], Size(),scale,scale,interpolation);
  resize(drawImg[3],scaledImg[3], Size(),scale,scale,interpolation);

  Mat CombinedImage = CombineImages(nimg,scaledImg,titles);

/* ===============================================================================================
   Write out ordered list of images, with number of matches
   =============================================================================================== */
  string distfile=output;
  pos = 0;
  pos = distfile.find("jpg",pos);
  distfile.replace(pos,3,"txt");
  ofstream outfile(distfile.c_str());
  outfile << "Input Image: " << imgfile.substr(0,imgfile.find_last_of(".")) << endl ;
  outfile << "\nImages scanned and number of significant matches: " << endl;

  for(int i=0; i < ndist; i++)
  {
	  idx = indices[i];
	  dist = distval[idx];
   	outfile << "Image: " << files[idx].substr(0,files[idx].find_last_of(".")) << " dist = " << dist << endl;
  }
  outfile.close();

/* ===============================================================================================
   Save image of all the top matches into a window
   =============================================================================================== */
  imwrite( output, CombinedImage );

  return 0;
}

/* ===============================================================================================
   Usage
   =============================================================================================== */
int usage()
{
    cout << "\n\n" <<endl;
    cout << "     " << "================================================================================================"  << endl;
    cout << "     " << "================================================================================================"  << endl;
    cout << "     " << "=                                                                                              ="  << endl;
    cout << "     " << "=                                      ScanImageDatabase                                       ="  << endl;
    cout << "     " << "=                                                                                              ="  << endl;
    cout << "     " << "=     This program reads in an image, extracts keypoints, generate descriptors for those       ="  << endl;
    cout << "     " << "=     keypoints, compare those descriptors with the descriptors of images from a database.     ="  << endl;
    cout << "     " << "=     Each comparison (image vs database image) is done using a robust filter, that checks     ="  << endl;
    cout << "     " << "=     for sensitivity, symmetry, as well as geometric proximity of the matches. Images in      ="  << endl;
    cout << "     " << "=     are ranked based on the number of matches with the input image. The top three hits       ="  << endl;
    cout << "     " << "=     are displayed.                                                                           ="  << endl;
    cout << "     " << "=                                                                                              ="  << endl;
    cout << "     " << "=     Usage is:                                                                                ="  << endl;
    cout << "     " << "=                 scanDatabase.exe                                                             ="  << endl;
    cout << "     " << "=                                 -i        <path to file containing test image>               ="  << endl;
    cout << "     " << "=                                 -d        <path to directory with corresponding images>      ="  << endl;
    cout << "     " << "=                                 -k        <path to directory with keypoints of images>       ="  << endl;
    cout << "     " << "=                                 -o        <path to output file>                              ="  << endl;
    cout << "     " << "=                                 -p        <path to param file for SURF>                      ="  << endl;
    cout << "     " << "=                                                                                              ="  << endl;
    cout << "     " << "================================================================================================"  << endl;
    cout << "     " << "================================================================================================"  << endl;
    cout << "\n\n" <<endl;

  return -1;
}

/* ===============================================================================================
   Procedure to read in flag values
   =============================================================================================== */
void read_flags(int argc, char** argv, string *imgfile, string *imgdir, string *infodir, string *output, string *param)
{
  string input;
  for(int i = 1; i < argc; i++)
  {
    input = argv[i];
    if (input == "-i") 
      *imgfile = argv[i + 1];
    if (input == "-d") 
      *imgdir = argv[i + 1];
    if (input == "-k") 
      *infodir = argv[i + 1];
    if (input == "-o") 
      *output = argv[i + 1];
    if (input == "-p") 
      *param = argv[i + 1];
  }
}

/* ===============================================================================================
   Procedure to read parameters for SURF
   =============================================================================================== */
void read_surfparams(string param, int *minHessian, int *octaves, int *octaveLayers, int *SizeMin, double *RespMin)
{
  ifstream inFile;
  inFile.open(param.c_str());
	string record;
	stringstream ss;

	while ( !inFile.eof () ) {    
		getline(inFile,record);
		if (record.find("minHessian") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *minHessian;
			ss.str("");
			ss.clear();
		}
		if (record.find("octaves") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *octaves;
			ss.str("");
			ss.clear();
		}
		if (record.find("octaveLayers") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *octaveLayers;
			ss.str("");
			ss.clear();
		}
		if (record.find("min Size") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *SizeMin;
			ss.str("");
			ss.clear();
		}
		if (record.find("min Resp") != std::string::npos) {
			ss<<record.substr(record.find_last_of(":") + 1);
			ss>> *RespMin;
			ss.str("");
			ss.clear();
		}
	}
}

/* ===============================================================================================
   Procedure to extract list of files from a directory
   =============================================================================================== */
int GetFileList(string directory, vector<string> &files) 
{
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(directory.c_str())) == NULL) {
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL) {
    files.push_back(string(dirp->d_name));
  }
  closedir(dp);
  return 0;
}

/* ===============================================================================================
   Procedure to filter keypoints based on size and response
   =============================================================================================== */
void filter_keypoints (vector<KeyPoint>& keypoints, int sizemin, double responsemin)
{
	vector<KeyPoint> temp;
	int npoints=keypoints.size();
	int size;
	double response;

  //filter based on size and reponse size
 	for(int i = 0; i < npoints; i++)
	{
		size = keypoints[i].size;
    response = keypoints[i].response;
		if(size > sizemin && response > responsemin)
			temp.push_back(keypoints[i]);
	}

	keypoints.clear();
	keypoints = temp;

  return;
}

/* ===============================================================================================
   Procedure to draw positions of key points on image using circles
   =============================================================================================== */

void showkeypts(vector<KeyPoint>& keypoints, Mat& drawImg)
{
	int thickness = 2;
	int lineType = 8;
	int shift = 0;
	int radius;

	int npoints=keypoints.size();

    	for(int i = 0; i < npoints; i++)
	{
		radius=keypoints[i].size/5;
		circle(drawImg,keypoints[i].pt,radius,Scalar(0,0,255),thickness,lineType,shift);
	}
}

/* ===============================================================================================
   This function combines multiple images into a single image
   =============================================================================================== */

Mat CombineImages(int nimage, Mat images[], string titles[])
{
	Mat NewImage;
	Mat myimg;
	Mat img;
	string title;

	int i;
	int offset_x, offset_y;
	int width,height;
	int width_max, height_max;
	int spacer_x,spacer_y;
	int pos_x,pos_y;

/* ===============================================================================================
	img_row : (maximum) number of images per row
	col_max : (maximum) number of images per column
   =============================================================================================== */

	int img_row, img_col;

/* ===============================================================================================
	Define the geometry of the new image: number of rows and columns,
	as well as size of each individual image
   =============================================================================================== */

	if (nimage == 2) {
		img_row = 1;
		img_col = 1;
	}
	else if (nimage == 4) {
		img_row = 2;
		img_col = 2;
	}
	else if (nimage == 6)  {
		img_row = 3;
		img_col = 2;
	}
	else {
		img_row = 4;
		img_col = 2;
	}

/* ===============================================================================================
	spacer_x and spacer_y are the spaces (# of pixels) left empty between images on row 1
	pos_x and pos_y is the position of the text for the first title
   =============================================================================================== */

	spacer_x = 20;
	spacer_y = 20;

	pos_x   = 2*spacer_x;
	pos_y   = spacer_y/2;

/* ===============================================================================================
	Get information from images: size (width and height)
   =============================================================================================== */

	width_max = 0;
	height_max = 0;
	for(i = 0; i < nimage; i++) {
		if(images[i].cols > width_max) width_max = images[i].cols;
	}
	double scale;
	i = 0;
	int h;
	while (i < nimage)
	{
		scale = ((float) width_max)/images[i].cols;
		height = images[i].rows*scale;
		h = height;
		i++;
		scale = ((float) width_max)/images[i].cols;
		height = images[i].rows*scale;
		if(height>h) h = height;
		i++;
		height_max=height_max + h;
	}
		
/* ===============================================================================================
	Create new image and size it so that it can contain all the images
	Generate image with white background
   =============================================================================================== */

	int size_x = spacer_x*(img_row+1) + width_max*img_row;
	int size_y = spacer_y*(img_col+1) + height_max;

	NewImage = Mat( cvSize(size_x,size_y), CV_8UC3, Scalar(255,255,255));

/* ===============================================================================================
	Now place all images into the New Image
   =============================================================================================== */

	Mat TargetROI;


	offset_x = spacer_x;
	offset_y = spacer_y;

   	int interpolation=INTER_LINEAR;
	i = 0;

	while (i < nimage)
	{
/* 		==================================================================================
		First put image on the left
   		================================================================================== */

		img = images[i];
		title = titles[i];

		scale = ((float) width_max)/img.cols;
		resize(img,myimg, Size(),scale,scale,interpolation);
		width = myimg.cols;
		height = myimg.rows;
		h = height;

/* 		==================================================================================
		- Set a rectangle where we want to put the image;
		- Define a "Region Of Interest" corresponding to this rectangle
		- Copy the image in that rectangle
   		================================================================================== */
 
		Rect ROI(offset_x, offset_y, width, height);
		TargetROI = NewImage(ROI);
		myimg.copyTo(TargetROI);

/* 		==================================================================================
		Write a title above the image: indicate the minHessian, octaves, and octaveLayers
		that were used for that image
		Title is written in black
   		================================================================================== */

		putText(NewImage, title, Point(pos_x,pos_y), CV_FONT_HERSHEY_PLAIN, 0.7, Scalar(0,0,0));

		i++;

/* 		==================================================================================
		Repeat for image representing the histogram, on the right
   		================================================================================== */
/* 		==================================================================================
		Reset the offsets in X and Y: move along rows.
   		================================================================================== */

		offset_x += width_max + spacer_x;
		pos_x    += width_max + spacer_x;

		img = images[i];
		title = titles[i];

		scale = ((float) width_max)/img.cols;
		resize(img,myimg, Size(),scale,scale,interpolation);
		width = myimg.cols;
		height = myimg.rows;
		if(height>h) h = height;

		Rect ROI2(offset_x, offset_y, width, height);
		TargetROI = NewImage(ROI2);
		myimg.copyTo(TargetROI);

		putText(NewImage, title, Point(pos_x,pos_y), CV_FONT_HERSHEY_PLAIN, 0.7, Scalar(0,0,0));

		i++;

/* 		==================================================================================
		Switch to new row
   		================================================================================== */

		offset_x = spacer_x;
		offset_y += spacer_y + h;
		pos_x = 2*spacer_x;
		pos_y += h +spacer_y;
	}

/* ===============================================================================================
	Now return New Image to main program
   =============================================================================================== */

	return NewImage;

}

/* ===============================================================================================
   Procedure to sort an array (in descending order) by providing the indices of the sorted positions
   (does not change the array)
   Use simple bubble sort! Could be easily improved....
   =============================================================================================== */

vector<int> ordered(vector<double> const & values, int nval)
{
	vector<int> indices(values.size());
	for(int i =0; i < nval ; i++)
	{
		indices[i] = i;
	}
	int idx;
	for(int i = 0; i < nval-1; i++)
	{
		for(int j = i+1; j< nval; j++)
		{
			if(values[indices[i]] < values[indices[j]])
			{
				idx = indices[i];
				indices[i]=indices[j];
				indices[j]=idx;
			}
		}
	}

    	return indices;
}

/* ===============================================================================================
   This function perform a ratio test: 
   - Clear matches for which NN ratio is > than threshold
   - return the number of removed points
   =============================================================================================== */

int ratioTest(vector<vector<cv::DMatch> > &matches, double ratio) 
{
  int removed=0;

	for (vector<vector<cv::DMatch> >::iterator
		matchIterator= matches.begin();
		matchIterator!= matches.end(); ++matchIterator) 
	{
/*      	==================================================================================
                if 2 NN have been identified
        	================================================================================== */

		if (matchIterator->size() > 1)
    {
/*      	==================================================================================
                check distance ratio
        	================================================================================== */

			if ((*matchIterator)[0].distance/ (*matchIterator)[1].distance > ratio)
      {
				matchIterator->clear(); // remove match
				removed++;
     	}

		} 

/*      	==================================================================================
                does not have 2 neighbours, then remove match
        	================================================================================== */
		else 
    {
			matchIterator->clear(); // remove match
			removed++;
		}
	}

	return removed;
}

/* ===============================================================================================
   This function perform a symmetry test:
   matches from 1 to 2 should match with matches from 2 to 1
   =============================================================================================== */

void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches) 
{

/*     	=========================================================================================
        for all matches image 1 -> image 2
       	========================================================================================= */

	for (vector<vector<cv::DMatch> >:: const_iterator matchIterator1= matches1.begin();
           matchIterator1!= matches1.end(); ++matchIterator1) 
	{
         	// ignore deleted matches
         	if (matchIterator1->size() < 2) continue;

/*     		=================================================================================
        	for all matches image 2 -> image 1
       		================================================================================= */

		for (vector<vector<cv::DMatch> >:: const_iterator matchIterator2= matches2.begin();
		matchIterator2!= matches2.end(); ++matchIterator2) 
		{

			// ignore deleted matches
			if (matchIterator2->size() < 2) continue;

/*     			=========================================================================
        		Symmetry test
       			========================================================================= */

			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
			(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) 
			{

				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
				(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));

				break; // next match in image 1 -> image 2

			}
		}
	}
}

/* ===============================================================================================
   Identify good matches using RANSAC; return fundamental matrix
   =============================================================================================== */

Mat ransacTest(const vector<cv::DMatch>& matches,const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2, vector<cv::DMatch>& outMatches) 
{

/*     	=========================================================================================
        Convert keypoints1 and keypoints2 into Point2f
       	========================================================================================= */

	vector<cv::Point2f> points1, points2;

	Mat fundemental;

	for (vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it) 
	{

		double x= keypoints1[it->queryIdx].pt.x;
		double y= keypoints1[it->queryIdx].pt.y;
		points1.push_back(Point2f(x,y));

		x= keypoints2[it->trainIdx].pt.x;
		y= keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x,y));
	}

/*     	=========================================================================================
        Compute fundamental matrix using RANSAC
       	========================================================================================= */

	vector<uchar> inliers(points1.size(),0);

	double confidence = 0.99;
	double distance = 3.0;
	int refineF = 1;

	if (points1.size()>0&&points2.size()>0)
	{

		Mat fundemental= cv::findFundamentalMat(
                        cv::Mat(points1),cv::Mat(points2), // matching points
                        inliers,       // match status (inlier or outlier)
                        CV_FM_RANSAC, // RANSAC method
                        distance,      // distance to epipolar line
                        confidence); // confidence probability

		// extract the surviving (inliers) matches

		vector<uchar>::const_iterator itIn= inliers.begin();
		vector<cv::DMatch>::const_iterator itM= matches.begin();

		for ( ;itIn!= inliers.end(); ++itIn, ++itM) 
		{
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}

		if (refineF) {

			// The F matrix will be recomputed with all accepted matches
			// Convert keypoints into Point2f for final F computation

			points1.clear();
			points2.clear();

			for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();it!= outMatches.end(); ++it) 
			{
				double x= keypoints1[it->queryIdx].pt.x;
				double y= keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x,y));

				x= keypoints2[it->trainIdx].pt.x;
				y= keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x,y));
			}

			// Compute 8-point F from all accepted matches

			if (points1.size()>0&&points2.size()>0){

				fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), // matches
						CV_FM_8POINT); // 8-point method

			}
		}
	}

 return fundemental;
}
