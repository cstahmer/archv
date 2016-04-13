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

using namespace cv;
using namespace std;

int usage();
void read_flags (int argc, char **argv, string *path2dir, string *path2out, string *dictfile, string *param);
void read_surfparameters (string param, int *minhessian, int *octaves, int *layers, int *SizeMin, double *RespMin);
int get_filelist (string path, vector <string> &allfiles);
void filter_keypoints (vector <KeyPoint> &keypoints, int SizeMin, double RespMin);

int main(int argc, char** argv)
{
/* ===============================================================================================
   Show usage if needed
   =============================================================================================== */
  if (argc < 2)
    return usage();
  string argument = argv[1];
  if (argument == "-h" || argument == "-help")
    return usage();

/* ===============================================================================================
   Initialize parameters for SURF
   =============================================================================================== */
  int minhessian = 2000, octaves = 5, layers = 4, SizeMin = 100;
  double RespMin = 2000;
  string param;
  string path2dir, path2out, dictfile;

/* ===============================================================================================
   Read in parameters for run
   =============================================================================================== */
  read_flags (argc, argv, &path2dir, &path2out, &dictfile, &param);

/* ===============================================================================================
   Read in parameters for SURF
   =============================================================================================== */
  read_surfparameters (param, &minhessian, &octaves, &layers, &SizeMin, &RespMin);

/* ===============================================================================================
   Create all structures that are needed to process the images (for SURF and Bag of Words (BOW)
   =============================================================================================== */
  Mat dictionary;                                                         // dictionary stored as matrix
  vector <KeyPoint> keypoints;                                            // keypoints stored in vector
  SurfFeatureDetector detector (minhessian, octaves, layers);             // define SURF detector
  Ptr <DescriptorExtractor> extractor = new SurfDescriptorExtractor();    // Feature extractor from keypoints
  Ptr <DescriptorExtractor> matcher = DescriptorMatcher::create("Flannbased");

  BOWImgDescriptorExtractor bowDE (extractor, matcher);       // define Bag of Words

/* ===============================================================================================
   Read in dictionary and assign it to the Bag of Word
   =============================================================================================== */
   FileStorage fs (dictfile, FileStorage::READ);
   fs["Dictionary"] >> dictionary;
   fs.release();

   int dictsize = dictionary.rows;
   bowDE.setVocabulary (dictionary);

/* ===============================================================================================
   Go to directory containing images, check it exists, get file list, select those files
   that contain an image (.jpg extension) 
   =============================================================================================== */
  bool isDir = false;
  struct stat sb;
  const char *dirName = path2dir.c_str();
  int counter = 0;

  if (stat (dirName, &sb) != 0 || !S_ISDIR(sb.st_mode))
  {
    cout << dirName << " does not exist, or isn't a directory; try again" << endl;
    return -1;
  }

  vector <string> allfiles = vector <string>(); // vector containing all the filenames
  int ierr = get_filelist (path2dir, allfiles);
  if (ierr != 0)
  {
    cout << "Problem reading in image files from directory" << endl;
    return -1;
  }

  vector <string> files;                        // only for files that contain an image
  string filename;                              

  for (int i = 0; i < allfiles.size(); i++)
  {
    filename = allfiles[i];
    if (filename.substr (filename.find_last_of(".") + 1) == "jpg")
    {
      files.push_back (filename);
      counter++;
    }
  }

  cout << " Number of images : " << counter << endl;

/* ===============================================================================================
   Now loop over all image files:
	- detect keypoints
	- filter keypoints
	- extract features
  - generate their histograms
   =============================================================================================== */
  const char * image;
  Mat img;
  string outfile, fileout;
  int pos = 0;
  counter = 0;

  Mat Histogram;
  Mat BOW_Descriptors;
  vector <vector<int>> pointIdxsOfClusters;

  for (int i =0; i < files.size(); i++)
  {

/*      =========================================================================================
        generate filenames
        ========================================================================================== */
      filename = path2dir;
      if (filename.back() != '/') 
      filename.append("/");
      filename.append(files[i]);

      if ((counter + 1) % 100 == 0 || counter == files.size() -1)
        cout << "Processing image # " << counter + 1 << "out of " << files.size() << " files" << endl;
      fileout = files[i];
      pos = 0;
      pos = fileout.find ("jpg", pos);
      fileout.replace (pos, 3, "yml");
      outfile = path2out;

      if (outfile.back() != '/')
      outfile.append ("/");
      outfile.append (fileout);
          
/*      =========================================================================================
        Open jpg file
        ========================================================================================== */
      image = filename.c_str();
      img = imread (image);

/*      =========================================================================================
        Detect keypoints and filter them
        ========================================================================================== */
      detector.detect (img, keypoints);
      filter_keypoints (keypoints, SizeMin, RespMin);

/*      =========================================================================================
        Extract features from keypoints, compare to dictionary of Bag of Words
        ========================================================================================== */
      bowDE.compute (img, keypoints, Histogram, &pointIdxsOfClusters, &BOW_Descriptors);

/*      =========================================================================================
        In current state the Histogram is normalized, if you want it un-normalized, uncomment following line
        ========================================================================================== */
     // Histogram *= BOW_Descriptors.rows;

/*      =========================================================================================
        Save histogram into file
        ========================================================================================== */
      FileStorage fs (outfile, FileStorage::WRITE);
      fs << "histogram" << Histogram;
      fs.release();

      counter++;
  }


 return 0;
}




int usage()
{
  cout << endl;
  cout << "This program reads in a collection of images, finds keypoints, computes their features with a Bag of Words and stores the corresponding histograms that describe the images" << endl;
  cout << "Usage is: " << endl;
  cout << "./histogram.exe -inputdir /path/to/images/ -outputdir /path/to/putfiles/ -dictfile /path/to/dictionary.yml -p paramfile" << endl;
  cout << endl;
  return -1;
}

void read_flags (int argc, char **argv, string *path2dir, string *path2out, string *dictfile, string *param)
{
  string parser;
  for (int i = 0; i < argc; i++)
  {
    parser = argv[i];
    if (parser == "-inputdir")
      *path2dir = argv[i+1];
    if (parser == "-outputdir")
      *path2out = argv[i+1];
    if (parser == "-dictfile")
      *dictfile = argv[i+1];
    if (parser == "-param")
      *param = argv[i+1];
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
