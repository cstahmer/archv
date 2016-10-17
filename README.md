# README #

This README would normally document whatever steps are necessary to get your application up and running.

### ARCHIVE-VISION ###
	
Arch-v is a collection of computer vision programs written in C++ with functions from the OpenCV library to perform analysis on large sets of images. The primary function is to locate recurring patterns within the images. Given a seed image the code base can locate similar features from that image within the rest of the set and output the images with the most similarity. There are four distinct programs however their uses are all related. The first program, processImages.cpp, is used to generate text files containing the keypoints and their mathematical descriptors. With these keypoints, analysis can be done to compare images and find matches. The second program, homography.cpp, is used to find the images that are most similar to a seed image provided by the user. The remaining program is best used when the best matches have already been found. drawMatches.cpp compares two images, finds their matches based on homography and then draws the keypoints and their relative match.



### SETTING IT UP###

In order to compile the source code, you will need several programs. The installation of OpenCV will require all you need for ArchV.  For OpenCV you need gcc, g++, cmake and several video and image libraries specified on their site.

***Installing OpenCV
***

These are a couple useful links for installing OpenCV –

http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html
https://help.ubuntu.com/community/OpenCV

Provided is a simplified version of the process for building OpenCV on Unix based systems –
	Download all dependencies required for OpenCV
	Download the zipped OpenCV file from their website
	Unzip OpenCV
	Go to directory and build using cmake
	Set up user system to correctly link to OpenCV library
	Check that library is linked correctly by testing some of the OpenCV sample programs

***Compiling the code***

Once OpenCV is installed and the libraries are included, go to the ArchV directory and run make all. You should be left with a .exe version of each of the four programs, process.exe, homography.exe, match.exe, and detect.exe. 


### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact