# README #


### ARCHIVE-VISION ###
	
Arch-v is a collection of computer vision programs written in C++ with functions from the OpenCV library to perform analysis on large sets of images. The primary function is to locate recurring patterns within the images. Given a seed image the code base can locate similar features from that image within the rest of the set and output the images with the most similarity. The first program, processImages.cpp, is used to generate text files containing the keypoints and their mathematical descriptors. With these keypoints, analysis can be done to compare images and find matches. The second program, scanDatabase.cpp, is used to find the images that are most similar to a seed image provided by the user. The remaining program is best used when the best matches have already been found. Lastly, drawMatches.cpp compares two images, finds their matches based on homography and then draws the keypoints and their relative match.

This app is best used for finding similar images to a seed image. The general method is therefore to first process the data set that you will compare your image to. Then run scanDatabase to compare your seed image with the dataset to find the best matches. Finally, it can be useful to see the matching parts between your seed image and the best match. To do so, run drawMatches.

The method:

* processImages
* scanDatabase
* drawMatches


### SETTING IT UP###

In order to compile and run this project you will need to install the OpenCV library.  

***Installing OpenCV
***

For OpenCV you need several dependencies; gcc, g++, cmake and several video and image libraries specified on their site.


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

Once OpenCV is installed and the libraries are included, go to the ArchV directory and run make all. You should be left with a .exe version of each program; processImages.exe, scanDatabase.exe, and drawMatches.exe.


### PROCESS IMAGES ###

processImages when executed reads in a parameter file (for SURF), an input directory that contains the images to be processed, and an output directory to put the YAML files containing the discovered keypoints and their descriptors. 

***Using processImages***

./process.exe -i <input directory> -o <output directory> -p <path to parameter file>

When executes it should immediately start processing the files one by one and outputting the information to the terminal. This is the most computationally intensive part of Arch-v and should take several minutes to complete. For each image that was found in the directory it will output the number of keypoints found and then the remaining keypoints after they have been filtered. It will also output the image number it is on for every hundred images it processes, i.e. image 0, 100, 200 … to the last one.
![run.png](https://bitbucket.org/repo/7RRn64/images/2882487937-run.png)

When done, you should see within the output directory a unique.yml file for each image that was in the input directory.
![filescropped.png](https://bitbucket.org/repo/7RRn64/images/422956241-filescropped.png)

Within each file there are all the keypoints and descriptors. The first part of the file is the keypoints, and then after that matrix there is a matrix of all of the descriptors for those keypoints. These files will then be read in for the homography matching part of ArchV. The files should look like this if any keypoints were found.
![ymlcropped.png](https://bitbucket.org/repo/7RRn64/images/1008876333-ymlcropped.png)

After this step has been completed, you can run the second program to find matches for your seed image within the image set.


### SCAN DATABASE ###

* Repo owner or admin
* Other community or team contact