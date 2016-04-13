#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

int usage();
void read_flags (int argc, char **argv, string *file1, string *file2, string *output, string *dictfile, string *param, double *scale);
void read_surfparameters (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin);
void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin);
void draw_keypoints (vector <KeyPoint> &keypoint, Mat &image);
Mat CombineImageHist (int nimage, Mat images[], string titles[]);

int main (int argc, char **argv)
{

/* ==========================================================================================
   Show usage if needed
   ========================================================================================== */
  if (argc < 2)
    return usage();
  string check = argv[1];
  if (check == "-help" || check == "-h")
    return usage();

/* ==========================================================================================
   Initialize Surf Parameters
   ========================================================================================== */
  string file1, file2, output, dictfile, param;
  int minhessian, octaves, layers, SizeMin;
  double RespMin;
  double scale = 1;

/* ==========================================================================================
   Read in Parameters for the run and for SURF
   ========================================================================================== */
  read_flags (argc, argv, &file1, &file2, &output, &dictfile, &param, &scale);
  read_surfparameters (param, &minhessian, &octaves, &layers, &SizeMin, &RespMin);

/* ==========================================================================================
   Create all structure needed to process images: SURF detection, FLANN matching
   ========================================================================================== */
  SurfFeatureDetector detector (minhessian, octaves, layers); Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create ("FlannBased");
  Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();

  vector <KeyPoint> keypoints1;
  vector <KeyPoint> keypoints2;

  int retries = 1;
  BOWImgDescriptorExtractor bowDE (extractor, matcher);

  Mat img1, img2;
  Mat histogram1, histogram2;
  Mat histogram1S, histogram2S;
  Mat BOW_Descriptors1, BOW_Descriptors2;
  vector< vector<int> > pointIdxsOfClusters1;
  vector< vector<int> > pointIdxsOfClusters2;
  img1 = imread (file1);
  img2 = imread (file2);

  Mat dictionary;

  /* ==========================================================================================
     Read in the Dictionary
     ========================================================================================== */
  FileStorage fs (dictfile, FileStorage::READ);
  fs["Dictionary"] >> dictionary;
  fs.release();

  int dictsize = dictionary.rows;
  bowDE.setVocabulary (dictionary);
  

/* ==========================================================================================
   Process image1:    MAKE THIS A FUNCTION
   1 - detect keypoints
   2 - filter keypoints
   3 - make histogram by finding which words match in dictionary and their frequencies
   ========================================================================================== */
  detector.detect (img1, keypoints1);
  filter_keypoints (keypoints1, SizeMin, RespMin);

  bowDE.compute (img1, keypoints1, histogram1, &pointIdxsOfClusters1, &BOW_Descriptors1);

  //comment this out if you want normalized histograms
  histogram1 *= BOW_Descriptors1.rows;

  /* =======================================================================================
     3: only keep keypoints with correspondance in dictionary
     ======================================================================================= */
  vector<int> flag (keypoints1.size());
  vector<KeyPoint> temp;
  int npoints;
  for (int i = 0; i < dictsize; i++)
  {
    npoints = pointIdxsOfClusters1[i].size(); 
    for (int j = 0; j < npoints; j++)
      flag[pointIdxsOfClusters1[i][j]] = 1;
  }

  for (int i = 0; i < keypoints1.size(); i++)
  {
    if (flag[i] == 1)
      temp.push_back(keypoints1[i]);
  }
  keypoints1.clear();
  keypoints1 = temp;
  temp.clear();

/* ==========================================================================================
   repeat for the second image
   ========================================================================================== */
  detector.detect (img2, keypoints2);
  filter_keypoints (keypoints2, SizeMin, RespMin);

  bowDE.compute (img2, keypoints2, histogram2, &pointIdxsOfClusters2, &BOW_Descriptors2);

  //comment this out if you want normalized histogram
  histogram2 *= BOW_Descriptors2.rows;

  vector<int> flag2(keypoints2.size());
  for (int i = 0; i < dictsize ; i++)
  {
   npoints = pointIdxsOfClusters2[i].size();
   for (int j = 0; j < npoints; j++)
     flag2[pointIdxsOfClusters2[i][j]] = 1;
  }

  for (int i = 0; i < keypoints2.size(); i ++)
  {
    if (flag[2] == 1)
      temp.push_back(keypoints2[i]);
  }
  keypoints2.clear();
  keypoints2 = temp;
  temp.clear();

   
/* ===============================================================================================
   Generate an image that displays the histogram of words for the input image1. This image will be
   shown next to the original image
   =============================================================================================== */

   double factor = dictsize/img1.cols;

   int hist_w1 = cvRound(factor*img1.cols); int hist_h1 = cvRound(factor*img1.rows);
   int histSize1 = dictsize;

   int bin_w1 = cvRound( (double) hist_w1/histSize1 );

   Mat histImage1( hist_h1, hist_w1, CV_8UC3, Scalar( 0,0,0) );

/* ===============================================================================================
   Normalize the histogram so that it fits in the window: [ 0, histImage.rows ]
   =============================================================================================== */

   normalize(histogram1, histogram1S, 0, histImage1.rows, NORM_MINMAX, -1, Mat() );

/* ===============================================================================================
   Basic plot of the histogram
   =============================================================================================== */

   int thickness = 4;
   int lineType = 8;
   int shift = 0;

   for( int i = 1; i < histSize1; i++ )
   {
	line( histImage1, Point( bin_w1*(i-1), hist_h1 - cvRound(histogram1S.at<float>(i-1)) ) ,
		Point( bin_w1*(i), hist_h1 - cvRound(histogram1S.at<float>(i)) ),
		Scalar( 0, 0, 255), thickness, lineType, shift  );
   }

   Mat histImage1S;
   factor = 1.0/factor;
   int interpolation=INTER_LINEAR;

   resize(histImage1,histImage1S, Size(),factor,factor,interpolation);

/* ===============================================================================================
   Repeat for image 2
   =============================================================================================== */

   factor = dictsize/img2.cols;

   int hist_w2 = cvRound(factor*img2.cols); int hist_h2 = cvRound(factor*img2.rows);
   int histSize2 = dictsize;

   int bin_w2 = cvRound( (double) hist_w2/histSize2 );

   Mat histImage2( hist_h2, hist_w2, CV_8UC3, Scalar( 0,0,0) );

   normalize(histogram2, histogram2S, 0, histImage2.rows, NORM_MINMAX, -1, Mat() );

   for( int i = 1; i < histSize2; i++ )
   {
	line( histImage2, Point( bin_w2*(i-1), hist_h2 - cvRound(histogram2S.at<float>(i-1)) ) ,
		Point( bin_w2*(i), hist_h2 - cvRound(histogram2S.at<float>(i)) ),
		Scalar( 0, 0, 255), thickness, lineType, shift  );
   }

   Mat histImage2S;
   factor = 1.0/factor;
   resize(histImage2,histImage2S, Size(),factor,factor,interpolation);

/* ===============================================================================================
   Generate:
	- the original image with keypoints drawn on it, scaled
	- the image showing the histogram, scaled
   Generate title for both
   =============================================================================================== */

   int nimg = 4;

   Mat drawImg[10];
   Mat scaledImg[10];
   img1.copyTo(drawImg[0]);
   histImage1S.copyTo(drawImg[1]);
   img2.copyTo(drawImg[2]);
   histImage2S.copyTo(drawImg[3]);

   string titles[10];
   stringstream str[10];

// Draw keypoints
   draw_keypoints (keypoints1,drawImg[0]);
   draw_keypoints (keypoints2,drawImg[2]);

// Generate titles for the four images

   str[0] << "Image: " << file1.substr(0,file1.find_last_of(".")) ;
   getline(str[0],titles[0]);

   str[1] << "Histogram: number of words = " << dictsize;
   getline(str[1],titles[1]);

   str[2] << "Image: " << file2.substr(0,file2.find_last_of(".")) ;
   getline(str[2],titles[2]);

   str[3] << "Histogram: number of words = " << dictsize;
   getline(str[3],titles[3]);

/* ===============================================================================================
   Now combine all images into a single, large one
   =============================================================================================== */

   resize(drawImg[0],scaledImg[0], Size(),scale,scale,interpolation);
   resize(drawImg[1],scaledImg[1], Size(),scale,scale,interpolation);
   resize(drawImg[2],scaledImg[2], Size(),scale,scale,interpolation);
   resize(drawImg[3],scaledImg[3], Size(),scale,scale,interpolation);

    Mat CombinedImage = CombineImageHist(nimg,scaledImg,titles);

/* ===============================================================================================
   Histogram-based distance between the two images
   =============================================================================================== */

    double dist;
    double x,y;
    for(int i = 0; i < dictsize; i++)
    {
	    x = histogram1.at<float>(i);
	    y = histogram2.at<float>(i);
	    dist = dist + (x-y)*(x-y);
    }

    dist = sqrt(dist);
 
    cout << "\nThe histogram-based distance between the two images is : " << dist <<endl;

/* ===============================================================================================
   Save image
   =============================================================================================== */

    imwrite( output, CombinedImage );

    return 0;
}


  




int usage()
{
    cout << "\nThis program reads in an image, detects key points, and compare them with a dictionary of key points.\n" <<endl;
    cout << "Usage  is:\n  Compare2Images.exe -i1 <image file 1> -i2 <image file 2> -o <output image file> -d <dictionary file> -p <param file for SURF> -s <scale factor>\n" <<endl;
  return -1;
}

void read_flags (int argc, char **argv, string *file1, string *file2, string *output, string *dictfile, string *param, double *scale)
{
  string parser;
  for (int i = 0; i < argc; i++)
  {
    parser = argv[i];
    if (parser == "-i1")
      *file1 = argv[i+1];
    if (parser == "-i2")
      *file2 = argv[i+1];
    if (parser == "-o")
      *output = argv[i+1];
    if (parser == "-p")
      *param = argv[i+1];
    if (parser == "-d")
      *dictfile = atoi(argv[i+1]);
    if (parser == "-s")
      *scale = atof(argv[i+1]);
  }
}

void read_surfparameters (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin)
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

void draw_keypoints (vector <KeyPoint> &keypoint, Mat &image)
{
  int radius, thickness = 1, lineType = 8, shift = 0;
  int n = keypoint.size();

  for (int i = 0; i < n; i++)
  {
    radius = keypoint[i].size / 4;
    circle (image, keypoint[i].pt, radius, Scalar (255, 0, 0), thickness, lineType, shift);
  }
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
