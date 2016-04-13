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
#include <algorithm>    // for random_shuffle and min

using namespace cv;
using namespace std;

int usage();
void read_flags (int argc, char **argv, string *path, int *DictionarySize, string *name, string *param, int *tcmax, double *tcepsilon, string *logfile);
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

/* ===============================================================================================
   Initialize parameters for Bag of Words and Dictionary
   =============================================================================================== */
  int DictionarySize = 8000;
  string path, name, logfile;
  int retries = 1;
  int flags = KMEANS_PP_CENTERS;
  int tcmax = 10;
  double tcepsilon = 0.01;

/* ===============================================================================================
   Read in parameters for run
   =============================================================================================== */
  read_flags (argc, argv, &path, &DictionarySize, &name, &param, &tcmax, &tcepsilon, &logfile);

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

  TermCriteria tc (CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, tcmax, tcepsilon); // set criteria for iterations 
  BOWKMeansTrainer bowTrainer (DictionarySize, tc, retries, flags);       // define Bag of Words

/* ===============================================================================================
   Open log file and writes parameters
   =============================================================================================== */
  ofstream log;
  log.open (logfile);

  cout << endl << " SURF Parameters for extracting keypoints from images..." << endl;
  cout << "      - MinHessian : " <<minhessian <<endl;
  cout << "      - Octaves    : " <<octaves <<endl;
  cout << "      - Layers     : " <<layers <<endl;
  cout << "      - SizeMin    : " <<SizeMin <<endl;
  cout << "      - RespMin    : " <<RespMin <<endl;
  cout << endl;

  log << endl << " SURF Parameters for extracting keypoints from images..." << endl;
  log << "      - MinHessian : " <<minhessian <<endl;
  log << "      - Octaves    : " <<octaves <<endl;
  log << "      - Layers     : " <<layers <<endl;
  log << endl;

  log << " Parameters for filtering keypoints detected by SURF ..." << endl;
  log << "      - SizeMin    : " <<SizeMin <<endl;
  log << "      - RespMin    : " <<RespMin <<endl;
  log << endl;

  log << " Parameters for Bag of Words and Dictionary ..." << endl;
  log << "      - DictSize   : " <<DictionarySize <<endl;
  log << "      - NiterMax   : " <<tcmax <<endl;
  log << "      - epsilon    : " <<tcepsilon <<endl;
  log << endl;

/* ===============================================================================================
   Go to directory containing images, check it exists, get file list, select those files
   that contain an image (.jpg extension), and randomize that list.
   Randomizing the file list removes biases coming from the lexicographic ordering of the files 
   =============================================================================================== */
  bool isDir = false;
  struct stat sb;
  const char *dirName = path.c_str();

  if (stat (dirName, &sb) != 0 || !S_ISDIR(sb.st_mode))
  {
    cout << dirName << " does not exist, or isn't a directory; try again" << endl;
    return -1;
  }

  vector <string> allfiles = vector <string>(); // vector containing all the filenames
  int ierr = get_filelist (path, allfiles);
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
      files.push_back (filename);
  }

  random_shuffle (files.begin(), files.end());   // randomize the files 

/* ===============================================================================================
   Now loop over all image files:
	- detect keypoints
	- filter keypoints
	- extract features
	- add to Bag-of-Word
   =============================================================================================== */
  const char * image;
  Mat img;
  int counter = 0;

  cout << endl << "Processing all files in the image library:\n" << endl;
  for (int i =0; i < files.size(); i++)
  {

/*      =========================================================================================
        Open image file (make sure it is a jpg!)
        ========================================================================================== */
      filename = path;
      if (filename.back() != '/') filename.append("/");
      filename.append(files[i]);
      if (filename.substr (filename.find_last_of(".") + 1) == "jpg")
      {
        if ((counter + 1) % 100 == 0 || counter == files.size() -1)
          cout << "Processing image # " << counter + 1 << "out of " << files.size() << " files" << endl;
        if (counter < 100000) // maximum number of images
        {
          counter ++;
          log << "Processing file: " << files[i] << endl; // should this be i?
          image = filename.c_str();
          img = imread (image);
          
/*      =========================================================================================
        Detect keypoints and filter them
        ========================================================================================== */
          detector.detect (img, keypoints);
          log << "Number of keypoints (before filtering) : " << keypoints.size() << endl;
          filter_keypoints (keypoints, SizeMin, RespMin);
          log << "Number of keypoints (after filtering) : " << keypoints.size() << endl;

/*      =========================================================================================
        Extract features from keypoints, and add to Bag of Words
        ========================================================================================== */
         if (keypoints.size())
         {
           Mat features;
           extractor->compute (img, keypoints, features);
           bowTrainer.add(features);
         }

         keypoints.clear();
         log << endl;
      }
    }
  }

/* ===============================================================================================
   Cluster the descriptors stored in the Bag of Words -> dictionary
   =============================================================================================== */
  int ncount = bowTrainer.descripotorsCount();
  log << "\n Number of descriptors in Bag of Word: " << ncount << endl;
  cout << "\n Number of descriptors in Bag of Word: " << ncount << endl;

  if (ncount > 0 && ncount > DictionarySize)
  {
    cout << endl;
    cout << "Clustering BOWs to generate a dictionary of size: " << DictionarySize << endl;
    cout << endl;
    dictionary = bowTrainer.cluster();
  }

/* ===============================================================================================
   Save dictionary into the output file
   =============================================================================================== */
  if (ncount > 0 && ncount > DictionarySize)
  {
    log << "Dictionary saved into file: " << name << endl;

    try
    {
      FileStorage fs (name, FileStorage::WRITE);
      fs << "Dictionary" << dictionary;
      fs.release();
    } catch (int e)
    {
      cout << "Unable to write to File " << name << ". Exception " << e << "." << endl; 
    }
  }
 else
 {
   log << "\n# of descriptors stored in the bow (" <<ncount<<") dictionary size (" << DictionarySize << ")!" << endl;
   log << "Dictionary was not created\n" << endl;
   cout << endl;
   cout << "\n# of descriptors stored in the bow (" <<ncount<<") dictionary size (" << DictionarySize << ")!" << endl;
   cout << "Dictionary was not created\n" << endl;
 }
 log.close();

 return 0;
}




int usage()
{
  cout << endl;
  cout << "builds a dictionary by clustering the keypoints over a random selection from the imageset" << endl;
  cout << "Usage is: " << endl;
  cout << "./dictionary.exe -d /path/to/dir -s size -n name -p paramfile -tcmax tcmax -tcepsilon tcepsilon" << endl;
  cout << endl;
  return -1;
}

void read_flags (int argc, char **argv, string *path, int *DictionarySize, string *name, string *param, int *tcmax, double *tcepsilon, string *logfile)
{
  string parser;
  for (int i = 0; i < argc; i++)
  {
    parser = argv[i];
    if (parser == "-d")
      *path = argv[i+1];
    if (parser == "-s")
      *DictionarySize = atoi(argv[i+1]);
    if (parser == "-n")
      *name = argv[i+1];
    if (parser == "-p")
      *param = argv[i+1];
    if (parser == "-tcmax")
      *tcmax = atoi(argv[i+1]);
    if (parser == "-tcepsilon")
      *tcepsilon = atof(argv[i+1]);
    if (parser == "-log")
      *logfile = argv[i+1];
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
