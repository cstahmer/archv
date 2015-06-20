# README #

Archv Bag Of Visual Words Engine (The Arch-V Bow-Wow) is a collection of tools built upon the OpenCV library
for creating bags of visual words (BOVW) representations of images of printed materials.  It works in conjunction 
with the Arch-V Java Toolset  (https://bitbucket.org/cstahmer/archv_java) to provide scalable visual search capability for archives of printed materials. Development of the codebase was initially supported by a Start-Up Grant from 
the National Endowment for the Humanities (NEH.) Continued development is a labor of love and necessity 
(pending future funding.)

The code is made available under a creative commons Attribution Share-Alike (CC BY-SA 4.0) License
https://creativecommons.org/licenses/by-sa/4.0/

The codebase is currently under heave development and there is presently no user documentation. This is one of the 
things we hope to provide with future funding. For information of how to implement or contribute, contact Carl Stahmer 
at cstahmer@ucdavis.edu

***********************************************************************************************************************************
In lieu of proper documentation, below is a snippet from my worklog that shows the commands that you need to run to 
create a dictionary and subsequently create a set of visual word files (BoVWs) for 
each image your library:

To make the dictionary:
./makeDictionary -d /path_to_your_image_library/ -n nameYourDictionaryFile -s 100000 -minhessian 1000 -octaves 15 -octavelayers 10 -tcmax 100000000 -tcepsilon 0.000000001 -sizefilter 100 -responsefilter 2000 -back -log &

You can see that there are several aspects of the run that can be tweaked, and 
each will produce a different result.  Getting things to work on your particular 
image set will require dome trial and error.  Here's a brief description of each 
parameter:

-d: The directory containing the images you want to build a dictionary for

-n: The name of the dictionary you want to create.  It will write the dictionary 
	out to this filename relative to the ./build directory of your Arch-V install. 

-s: The number of words to include in your dictionary.  I'd probably shrink this 
	to 30,000 - 50,000 for your first tests.

-minhessian: Threshold for the keypoint detector. Only features, whose hessian is 
				larger than -minhessian are retained by the detector. Therefore, the 
				larger the value, the fewer keypoints you will get. A good default 
				value for you to start with could be from 300 to 500, depending from 
				the image contrast. (Note that for the ballad images, we determined 
				that a much higher -minhessian was in order, but I'd start with 500 
				for your type of resource.)  Also, in case you aren't familiar with 
				hessian values, see the answer to the question at: 
				http://stackoverflow.com/questions/18744051/opencv-surf-hessian-minimum-threshold

-octaves: The number of a gaussian pyramid octaves that the detector uses. It is set 
			to 4 by default. If you want to get very large features, use the larger 
			value. If you want just small features, decrease it.  This setting will 
			affect your results greatly.  For the type of matching you are wanting to do, 
			I'd recommend starting with a smaller value.

-octavelayers: The number of images within each octave of a gaussian pyramid. For 
				recognition of things in photographs, a small value, such as 2 
				(the OpenCV default) works best; however, our experience dealing 
				with printed materials is that an higher value works better.  
				Note, however, that setting too high a value can put the system into 
				an infinite loop of looking for sub regions that don't exist.  So 
				start smaller and then walk your way up, looking at your results 
				along the way to find the best value.

-tcmax: The maximum number to times to iterate through a sequence of blurring and 
			then extracting when creating feature points.  If you recall, the paper 
			on the newspaper poetry project showed how they were working from 4 
			different levels of blurred objects.  This value affects this, but as a 
			representation of gaussian blur distance as opposed to number to times to 
			iterate.  

-tcepsilon: The way dictionary building works, the system pulls all of the feature 
				points from a selection of images and then compares them (through 
				a system known on quantization) to create the dictionary.  Each 
				time a feature point is found, the system checks all other feature 
				points that already exist in the dictionary.  If it finds a match, 
				it moves on.  If it doesn't, it adds it to the dictionary.  The -tcepsilon 
				value give the system some parameters on how different two feature points 
				can be and still be considered to the same when building the dictionary.  
				The larger the value, the greater the difference can be and still be 
				considered the same.  

-sizefilter: Sets a minimum feature point size to be considered when placing things 
				in the dictionary.

-responsefilter: Sets a response strength filter that limits the types of features 
					to be included.

For a good, quick explanation of how feature points are determined, see 
http://stackoverflow.com/questions/10328298/what-does-size-and-response-exactly-represent-in-a-surf-keypoint.  
It will give you better idea of the system overall and also of how -sizefilter 
and -responsefilter work.


--

Once you have your dictionary, you will need to now build your BoVW files for your library.
To do this, I run the following command
% ./makeWordHistograms -eval /path_to_your_image_library/ -write /path_where_you_want_to_output_the_BoVW_files/ -dict /path/name_of_your_dictionary.yml -minhessian 1000 -octaves 15 -octavelayers 10 -tcmax 100000000 -tcepsilon 0.000000001 -sizefilter 100 -responsefilter 2000 -back -log &

For the feature point extraction variables, use the same values that you used when 
creating the dictionary.  The values that you will have to alter are the -eval 
directory, which tells the system where the images you want process are and the 
-write directory, which tells it where to put the BoVW files.

--

When you are done with the above process, you will have a set of BoVW files for each 
image in your library.  You can compare them by hand, or index them with lucene to see 
how you did.


Carl G Stahmer
Director of Digital Scholarship
University of California Davis Library
www.carlstahmer.com
@cstahmer