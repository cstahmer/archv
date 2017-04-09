#### README ####

## ARCHIVE-VISION ##

Archive-Vision (archv or arch-v) is a collection of computer vision programs written in C++ which utilizes functions from the OpenCV library to perform analysis on large image sets. The primary function is to locate recurring patterns within each image in a set of images. *Arch-v* locates features from a given seed image within an imageset and outputs the image(s) with the most similarities. The first program, **processImages.cpp**, generates text files containing the keypoints and their mathmatical descriptors; with the keypoints, analysis can be done to compare images and find matches. The second program, **scanDatabase.cpp**, finds the images that are most similar to a given seed image. The third program, **drawMatches.cpp**, compares two images, locates their matches based on homography, then draws the keypoints and their relative match; this is most useful when the best matches have already been found.


The best use for *arch-v* is to find images which are similar to a seed image. The standard **method** is to process the image set that you will compare your seed image to, scan through the generated dataset with a given seed image to find the best matches, then draw identifiers for matching features between the seed image and its best match.

Therefore, the **method** is as follows:

1. **processImages**
2. **scanDatabase**
3. **drawMatches**

### SETTING UP ARCH-V ###

In order to compile and run this project you will need to install the OpenCV library.

***Note:*** click [here](https://bitbucket.org/digitalscholarship/documentation/src/0d45e95a99e6aa490a4c6c659f7182390b3ee41a?at=master) to download the documentation for setting up arch-v on **macOS**.

***Installing OpenCV***

For OpenCV, you need several dependencies: gcc, g++, cmake and several video and image libraries specified on their site. For Linux, use these links to install OpenCV:

* [Introduction to OpenCV](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)
* [Installation on Linux](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation)
* [Ubuntu Documentation: OpenCV](https://help.ubuntu.com/community/OpenCV)

Provided is a simplified version of the process for building OpenCV on Unix based systems:

* Download all of the dependices required for OpenCV
* Download the zipped OpenCV file from their website
* Unzip the OpenCV file
* Go into the unzipped OpenCV directory and built using cmake
* Set up the configurations to link the OpenCV library
* Verify that the library has been linked correctly by following this tutorial to [Load and Display an Image](http://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html#display-image)

***Compiling Arch-v***

Once OpenCV is installed and the libraries are included, go to your arch-v directory and run `make all`. You should be left with an executable (.exe) version of each program: processImages.exe, scanDatabase.exe, and drawMatches.exe.

***Note:*** all image files should be `.jpg` and click [here](http://benjaminpauley.net/BL-Flickr.tar.gz) if you wish to download the  same imageset that this documentation will be using.

### PROCESS IMAGES ###

**processImages** reads in a parameter file (for SURF), an input directory that contains the images to be processed, and an output directory to put the YAML files containing their discovered keypoints and their descriptors. 

***Using processImages***

When running processImages, the `<path to input directory>` is the imageset that you are trying to process. The output directory, which is where the `.yml` files will be stored, must already exist. This program is the most computationally intensive component of arch-v and should take several minutes to complete.

	$ ./processImages.exe -i <path to input directory> -o <path to output directory> -p <path to SURF parameter file>

... 

	$ ./processImages.exe -i imageset/ -o keypoints/ -p param
	Processed all 1067 images, and placed the .yml files in keypoints/
	$ 

Taking a look at the input and output directories, each image has a corresponding `.yml` file.

	$ ls imageset/
	10998090545_ba532dc156_o.jpg	10998857066_8e73d5d435_o.jpg	10999455433_17a0db32b6_o.jpg
   	10998095795_483363ebc7_o.jpg	10998859196_b863e10978_o.jpg	10999456085_90141fcbdf_o.jpg
   	10998096905_60b65e863b_o.jpg	10998859965_2b6ea731f5_o.jpg	10999459045_abb11c05ca_o.jpg
   	10998100245_835ea9f601_o.jpg	10998864205_8b3b560385_o.jpg	10999459103_ff81e309b1_o.jpg
   	10998121025_708152d1b0_o.jpg	10998865296_cb00afbbac_o.jpg	10999463225_ee672db6f5_o.jpg
   	...
   	10998975233_1ba7fd59cc_o.jpg	10999363305_db9784db93_o.jpg	10999654263_bf18a3a94f_o.jpg
   	$ 

...

	$ ls keypoints/
	10998090545_ba532dc156_o.yml	10998857066_8e73d5d435_o.yml	10999455433_17a0db32b6_o.yml
	10998095795_483363ebc7_o.yml	10998859196_b863e10978_o.yml	10999456085_90141fcbdf_o.yml
	10998096905_60b65e863b_o.yml	10998859965_2b6ea731f5_o.yml	10999459045_abb11c05ca_o.yml
	10998100245_835ea9f601_o.yml	10998864205_8b3b560385_o.yml	10999459103_ff81e309b1_o.yml
	10998121025_708152d1b0_o.yml	10998865296_cb00afbbac_o.yml	10999463225_ee672db6f5_o.yml
	...
	10998975233_1ba7fd59cc_o.yml	10999363305_db9784db93_o.yml	10999654263_bf18a3a94f_o.yml
	$ 

These files will then be read in for the homography matching component of arch-v. Looking at the first `.yml` file, the first matrix contains the keypoints and the second matrix contains the descriptors for the keypoints:

	$ cat 11000210893_335dee8657_o.yml
	%YAML:1.0
	keypoints: [ 3.2808068847656250e+02, 4.8052941894531250e+02, 56.,
		1.5546127319335938e+02, 2.0784480468750000e+04, 1, 1,
		3.2770791625976562e+02, 4.7961071777343750e+02, 62.,
		1.5044647216796875e+02, 1.7364101562500000e+04, 2, 1,
		7.3369165039062500e+02, 3.8075772094726562e+02, 58.,
		3.4650259399414062e+02, 1.7311123046875000e+04, 1, 1,
		...
		2.8078506469726562e+02, 5.0304754638671875e+02, 2, 1 ]
	descriptors: !!opencv-matrix
		rows: 503
		cols: 64
		dt: f
		data: [ 1.21101424e-04, -7.75693916e-03, 6.75657764e-03,
			9.65651684e-03, 2.51447931e-02, -2.49528252e-02, 2.57605817e-02,
			2.52846610e-02, 1.91994593e-04, 5.52531055e-05, 6.01771404e-04,
			4.99580929e-04, -5.58408658e-07, 6.68875509e-05, 1.16100106e-04,
			8.46642070e-05, -5.39382035e-03, -2.57135532e-03, 1.99829433e-02,
			5.59122600e-02, 8.40003043e-02, -2.42966130e-01, 3.75171691e-01,
			...
			1.17994336e-04, 1.03991253e-04 ]
	$ 


After this step has been completed, you can run the second program to find matches for your seed image within the image set.

### SCAN DATABASE ###

**scanDatabase** reads in a seed image, the directory of `.yml` files, a filepath to an output json (text) file, and the path to the SURF parameter file.

The program reads in your seed image, extracts the keypoints and descriptors like processImages had, and compares that information with the keypoints and descriptors from every `.yml` file; this is essentially comparing the seed image against every image in the imageset. Each comparison is done using a robust filter, that checks for sensitivity, symmetry, as well as geometric proximity of the matches. Images are then ranked based on the number of matches they have with the seed image. The top three matches are then displayed with the number of matches they contained. There will be two output files: `output.jpg` and `output.txt`.

***Using scanDatabase***


	$ ./scanDatabase.exe -i <path to seed image> -d <path to input directory> -k <path to keypoints directory> -o <path to output file> -p <path to SURF parameter file>

While scanDatabase is running, it will print its progress for every hundred images that it has processed.

	$ ./scanDatabase.exe -i imageset/11000210893_335dee8657_o.jpg -d imageset/ -k keypoints/ -o output.jpg -p param
	Processing image # 100 out of 1067 images in the database
	Processing image # 200 out of 1067 images in the database
	Processing image # 300 out of 1067 images in the database
	Processing image # 400 out of 1067 images in the database
	...
	Processing image # 1067 out of 1067 images in the database
	$ 

When the program finishes, it will have saved the output in json from to the text file with the names that you had specified `<path to output file>`. Combining the top hits should look similar to the following image:

![output.jpg](https://bitbucket.org/repo/7RRn64/images/3554904158-output.jpg)
The seed image is in the top left, the best match is immediately to the right (being the seed image itself), the second best is the first image in the second row, and so on. The filename and distance are included on top of each image. The distance refers to the remaining number of matches.

### DRAW MATCHES ###

**drawMatches** takes as input two images, the path to an output image file as well as the path to the parameter file. It is best to use similar parameters to what was used in the first two steps to find these two images that are known to be similar. The code is also self contained so you can input any two images and any SURF parameter files to find the keypoints that match and have passed the robust homography filter.

***Using drawMatches***

	$ ./drawMatches.exe -i1 <path to seed image> -i2 <path to image for comparison> -o <path to output image> -p <path to SURF parameter file>

The following execution of drawMatches is between the seed image that we've been using throughout this documentation and its best match.

	$ ./drawMatches.exe -i1 imageset/11000210893_335dee8657_o.jpg -i2 imageset/11000152114_551839b72c_o.jpg -o match.jpg -p param
	Number of keypoints 1 : 3735 After filter : 503
	Number of keypoints 2 : 3518 After filter : 454
	number of remaining matches after homography: 38
	$ 

When the program finishes, it will have saved the output image and text file with the names that you had specified `<path to output image>`. The output image should look similar to the following image:

![match.jpg](https://bitbucket.org/repo/7RRn64/images/3795577038-match.jpg)
The red circles are the keypoints with their radii equal their size and the blues lines connect the matching keypoints between the two images.

### PARAMETER FILE ###

The parameter file should be a `.txt` file that follows this format:

	$ cat param
	minHessian: 500
	octaves:  4
	octaveLayers: 4
	min Size: 50
	min Response: 100
	$ 

#### CONTACT ####

Developed by [Carl Stahmer](http://www.carlstahmer.com) and [Arthur Koehl](avkoehl@ucdavis.edu) at the Data Science Initiative of the University of California Davis. Documentation authored by [Henry Le](hutle@ucdavis.edu).
