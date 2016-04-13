#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <algorithm>

using namespace cv;
using namespace std;

int usage (char **argv);
void read_flags (int argc, char **argv, string *imgfile1, string *output, string *param, double *scale, double *ratio);
void read_surfparams (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin);
int get_filelist (string path, vector <string> &allfiles);
void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin);
void showkeypts (vector <KeyPoint> &keypoints, Mat &drawImg);
Mat CombineImageHist (int nimage, Mat images[], string titles[]);
vector <int> ordered (vector<double> const &values, int nval);
Mat DrawMatch(Mat& image1, vector<KeyPoint>& keypoints1, Mat& image2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches1to2);
int ratioTest(vector<vector<DMatch> > &matches, double ratio);
void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches);
Mat ransacTest(const vector<DMatch>& matches,const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches);

int main(int argc, char** argv)
{
/* ===============================================================================================
   Show usage if needed
   =============================================================================================== */
   if (argc < 2)
    return usage(argv);
   string argument = argv[1];
   if (argument == "-h" || argument == "-help")
    return usage(argv);

/* ===============================================================================================
   Initialize parameters for SURF // set scale = 0.8
   =============================================================================================== */
   int minhessian, octaves, layers, SizeMin;
   double RespMin;
   double scale, ratio;
   string param, output, imgdir;
   string imgfile1, imgfile2;

/* ===============================================================================================
   Read in parameters for run
   =============================================================================================== */
   read_flags (argc, argv, &imgfile1, &output, &param, &scale, &ratio);

/* ===============================================================================================
   Read in parameters for SURF
   =============================================================================================== */
   read_surfparams (param, &minhessian, &octaves, &layers, &SizeMin, &RespMin);

/* ===============================================================================================
   Create all structures that are needed to process the images:
        - use SURF for key points detection and feature extraction
   =============================================================================================== */
   vector <KeyPoint> keypoints1;
   vector <KeyPoint> keypoints2;
   SurfFeatureDetector detector (minhessian, octaves, layers);          // SURF detector
   Ptr <DescriptorExtractor> extractor = new SurfDescriptorExtractor(); // Feature Extractor from Keypoints

   BFMatcher matcher;
   vector <vector<DMatch>> matches1;
   vector <vector<DMatch>> matches2;
   vector <DMatch> sym_matches;
   vector <DMatch> matches;

/*  ===============================================================================================
    Process images:
	- detect keypoints
	- filter keypoints
    =============================================================================================== */
   Mat img1;
   img1 = imread (imgfile1);

   detector.detect (img1, keypoints1);

   filter_keypoints (keypoints1, SizeMin, RespMin);

   Mat descriptors1;
   Mat descriptors2;
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

   int ierr = get_filelist(imgdir, allfiles);
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
   cout << " Number of images : " << counter << endl;

/* ===============================================================================================
   Now loop over all image files:
   =============================================================================================== */
  vector<double> distval(files.size());
  vector<int>    indices(files.size());
  Mat img2;

  counter = 0;
  int ndist = 0;
  for(int i = 0; i < files.size(); i++)
  {
/*      =========================================================================================
        Generate filenames
        ========================================================================================== */
	
	if((counter+1) % 100 ==0 || counter == files.size()-1) cout << "Processing image # " << counter+1 << " out of " << files.size() << " files"<<endl;

	filename = imgdir;
	if(filename.back() != '/') filename.append("/");
	filename.append(files[i]);

/*      =========================================================================================
        Read image from file
        ========================================================================================== */

	    img2 = imread(filename);

/*      =========================================================================================
        Process image2
        ========================================================================================== */

	    keypoints2.clear();
    	detector.detect( img2, keypoints2);
    	filter_keypoints(keypoints2,SizeMin,RespMin);

      if(keypoints1.size()>0 && keypoints2.size() > 0 ) 
      {
		    extractor->compute(img2, keypoints2, descriptors2);

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
   str[0] << "Input Image: " << imgfile1.substr(0,imgfile1.find_last_of(".")) ;
   getline(str[0],titles[0]);

   int idx;
   int pos;
   double dist;
   for(int i=0; i < 3; i++)
   {
	   idx = indices[i];
	   dist = distval[idx];

	  imfile = files[idx];
	  filename = imgdir;
	  if(filename.back() != '/') filename.append("/");
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

    Mat CombinedImage = CombineImageHist(nimg,scaledImg,titles);

/* ===============================================================================================
   Save image 
   =============================================================================================== */

    imwrite( output, CombinedImage );

    return 0;
}




int usage (char **argv)
{
  cout << "\n this program finds if one image can be found within another, and then displays them side by side" << endl;
  cout << "./detectimages.exe -i1 <img file 1> -i2 <img file 2> -o <output img file> -p <param file> -s <scale> -r <ratio>" << endl;
  return 1;
}

void read_flags (int argc, char **argv, string *imgfile1, string *output, string *param, double *scale, double *ratio)
{
  string input;
  for (int i = 1; i < argc; i++)  
  {
    input = argv[i];
    if (input == "-i1")
      *imgfile1 = argv[i+1];
    if (input == "-o")
      *output = argv[i+1];
    if (input == "-p")
      *param = argv[i+1];
    if (input == "-r")
      *ratio = atof(argv[i+1]);
    if (input == "-s")
      *scale = atof(argv[i+1]);
  }
}

void read_surfparams (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin)
{
  ifstream inFile;
  inFile.open(param);
  string record;
  stringstream ss;

  while (!inFile.eof())
  {
    getline (inFile, record);
    if (record.find("minhessian") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *minhessian;
      ss.str("");
      ss.clear();
    }
    if (record.find("octaves") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *octaves;
      ss.str("");
      ss.clear();
    }
    if (record.find("layers") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *layers;
      ss.str("");
      ss.clear();
    }
    if (record.find("min Size") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *SizeMin;
      ss.str("");
      ss.clear();
    }
    if (record.find("min Resp") !=std::string::npos)
    {
      ss << record.substr(record.find_last_of(":") + 1);
      ss >> *RespMin;
      ss.str("");
      ss.clear();
    }
  }
}

int get_filelist (string path, vector <string> &allfiles)
{
  DIR *dp;
  struct dirent *dirp;
  if ((dp = opendir (path.c_str())) == NULL)
    return errno;
  
  while ((dirp = readdir(dp)) != NULL)
  {
    allfiles.push_back (string (dirp->d_name));
  }

  closedir (dp);
  return 0;
}

void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin)
{
  vector <KeyPoint> keypoint2;
  int npoints = keypoints.size();
  int size;
  double response;

/*    ==============================================================================================
      filter keypoints based on size
      ============================================================================================== */
      for (int i = 0; i < npoints; i++)
      {
        size = keypoints[i].size;
        if (size > SizeMin)
          keypoint2.push_back(keypoints[i]);
      }

      keypoints.clear();
      keypoints = keypoint2;
      npoints = keypoints.size();
      keypoint2.clear();

/*    ==============================================================================================
      filter keypoints based on size
      ============================================================================================== */
      for (int i = 0; i < npoints; i++)
      {
        response = keypoints[i].response;
        if (response > RespMin)
          keypoint2.push_back(keypoints[i]);
      }

      keypoints.clear();
      keypoints = keypoint2;
}

void showkeypts (vector <KeyPoint> &keypoints, Mat &drawImg)
{
/* =====================================================================================
	all variables needed for openCV circle function, n is number of keypoints 
  ===================================================================================== */
  int radius, thickness = 1, lineType = 8, shift = 0;
  int n = keypoints.size();

/* =====================================================================================
	for each keypoint: draw a circle with radius scaled to keypoints size around
  ===================================================================================== */
  for (int i = 0; i < n; i++)
  {
    radius = keypoints[i].size / 4;
    circle (drawImg, keypoints[i].pt, radius, Scalar (255,0,0), thickness, lineType, shift); 
  }
  return;
}

Mat CombineImageHist(int nimage, Mat images[], string titles[])
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
	for(i = 0; i < nimage; i++) {
		scale = ((float) width_max)/images[i].cols;
		height = images[i].rows*scale;
		if(height > height_max) height_max = height;
	}
		
/* ===============================================================================================
	Create new image and size it so that it can contain all the images
	Generate image with white background
   =============================================================================================== */

	int size_x = spacer_x*(img_row+1) + width_max*img_row;
	int size_y = spacer_y*(img_col+1) + height_max*img_col;

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

		Rect ROI2(offset_x, offset_y, width, height);
		TargetROI = NewImage(ROI2);
		myimg.copyTo(TargetROI);

		putText(NewImage, title, Point(pos_x,pos_y), CV_FONT_HERSHEY_PLAIN, 0.7, Scalar(0,0,0));

		i++;

/* 		==================================================================================
		Switch to new row
   		================================================================================== */

		offset_x = spacer_x;
		offset_y += spacer_y + height;
		pos_x = 2*spacer_x;
		pos_y += height +spacer_y;
	}

/* ===============================================================================================
	Now return New Image to main program
   =============================================================================================== */
	return NewImage;
}

Mat DrawMatch(Mat& image1, vector<KeyPoint>& keypoints1, Mat& image2, vector<KeyPoint>& keypoints2, vector<DMatch>& matches1to2)
{
  Mat newimage;
  int offset_x;
  int width, height;
  int spacer_x;

  // get information from the images
  spacer_x = 20;
  width = image1.cols + image2.cols + spacer_x;
  height = image1.rows;
  if (image2.rows > height)
    height = image2.rows;

  // create newimage with white background and size it such that it can contain both
  newimage = Mat (cvSize (width, height), CV_8UC3, Scalar (255,255,255));


  // place all the images onto the newimage
  Mat TargetROI;
  offset_x = 0;

    /* first put image on the left: 
       - Set a rectangle where we want to put the image
       - Define a "Region of Interest" corresponding to this rectangle
       - Copy the image in that rectangle */
       Rect ROI (offset_x, 0, image1.cols, image1.rows);
       TargetROI = newimage (ROI);
       image1.copyTo (TargetROI);

    // repeat for image representing the histogram on the right
      offset_x = offset_x + image1.cols + spacer_x;
      Rect ROI2 (offset_x, 0, image2.cols, image2.rows);
      TargetROI = newimage (ROI2);
      image2.copyTo (TargetROI);


  // now add lines between the matched keypoints
  int thickness = 4;
  int lineType = 8;
  int shift = 0;

  for (int m = 0; m < matches1to2.size(); m++)
  {
    int i1 = matches1to2[m].queryIdx;
    int i2 = matches1to2[m].trainIdx;
    CV_Assert (i1 >= 0 && i1 < static_cast<int> (keypoints1.size()));
    CV_Assert (i2 >= 0 && i2 < static_cast<int> (keypoints2.size()));

    Point2f pt1 = keypoints1[i1].pt;
    Point2f pt2 = keypoints2[i2].pt;
    pt2.x = pt2.x + offset_x;

    line (newimage, pt1, pt2, Scalar(255,0,0), thickness, lineType, shift);
  }

  return newimage;
}

/* This function performs a ratio test:
  - clear matches for which NN ratio is > than threshold
  - return the number of removed points
*/
int ratioTest(vector<vector<cv::DMatch> > &matches, double ratio)
{
  int removed = 0;
  for (vector <vector<cv::DMatch>>::iterator matchIterator = matches.begin(); matchIterator != matches.end(); ++matchIterator)
  {
    // if 2 NN have been identified
    if (matchIterator->size() > 1)
    {
      // check distance ratio
      if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio)
      {
        matchIterator->clear(); // remove match
        removed++;
      }
    }

    // if does not have 2 neighbours, then remove match
    else 
    {
      matchIterator->clear(); // remove match
      removed++;
    }
  }

  return removed;
}

void symmetryTest(const vector<vector<DMatch> >& matches1,const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches)
{
/*	=========================================================================================
    for all matches image 1 -> image 2
   	========================================================================================= */
	for (vector<vector<cv::DMatch> >:: const_iterator matchIterator1= matches1.begin(); matchIterator1!= matches1.end(); ++matchIterator1) 
	{
   	// ignore deleted matches
  	if (matchIterator1->size() < 2) continue;

/* 		=================================================================================
     	for all matches image 2 -> image 1
   		================================================================================= */

		for (std::vector<std::vector<cv::DMatch> >:: const_iterator matchIterator2= matches2.begin();
		matchIterator2!= matches2.end(); ++matchIterator2) 
		{
			// ignore deleted matches
			if (matchIterator2->size() < 2) continue;
/* 			=========================================================================
     		Symmetry test
   			========================================================================= */
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) 
			{
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx, (*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

Mat ransacTest(const vector<DMatch>& matches,const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches)
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

vector<int> ordered (vector<double> const &values, int nval)
{
  vector<int> indices (values.size());
  for (int i = 0; i < nval; i++)
    indices[i] = i;
  int idx;
  for (int i = 0; i < nval -1; i++)
  {
    for (int j = i + 1; j < nval; j++)
    {
      if (values[indices[i]] < values[indices[j]])
      {
        idx = indices[i];
        indices[i] = indices[j];
        indices[j] = idx;
      }
    }
  }
  return indices;
}

    

