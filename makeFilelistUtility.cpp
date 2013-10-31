/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: makeFileListUtility.cpp
 *
 * Reads a directory of images and outputs sql or csv file for inserting of data into
 * the BIA set of tables.  Set the starting ID number for the sequence from the bia_impressions table
 * and whether or not to format as SQL
 * for the library.
 *
 * Copyright (C) 2012 Carl Stahmer (cstahmer@gmail.com) 
 * Early Modern Center, University of California, Santa Barbara
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Creative Commons licence, version 3.
 *    
 * See http://creativecommons.org/licenses/by/3.0/legalcode for the 
 * complete licence.
 *    
 * This program is distributed WITHOUT ANY WARRANTY; 
 * without even the implied warranty of MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for 
 * more details.
 * 
 */

#include "cv.h" 
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

#include "helper.cpp"
#include "imgextract.cpp"
#include "imageDocument.cpp"

using namespace cv; 
using namespace std;

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;

char ch[30];

Mat dictionary;


int main(int argc, char* argv[]) {
	
	Helper helper;
	ImgDoc imagedoc;
	string event;
	
	// set default directories
	string evalDirectory = "/home/cstahmer/bia_batch_jpeg/";
	string writeToDirectory = "/web/sites/beeb/FileLists/";
	
	bool runInBackground = RUN_IN_BACKGROUND;
	bool writelog = WRITE_LOG;
	bool writeMysql = true;
	int maxheight = 200;
	int maxwidth = 200;
	int startID = 5500;
	
	bool isDir = false; // set to designate if this is an evaluation directory of a single file to be evaluated

	
	//"Usage is -eval <dirctory of evaluation files> -write <directory to which histogram files should be saved> -dict <dictionary file to load>"
	
    for (int i = 1; i < argc; i++) { 
    	string arument = argv[i];
        if (arument == "-eval") {
        	evalDirectory = argv[i + 1];
        }
        if (arument == "-write") {
        	writeToDirectory = argv[i + 1];
        }
        if (arument == "-log") {
        	writelog = true;
        }
        if (arument == "-back") {
        	runInBackground = true;
        }
        if (arument == "-help") {
            cout << "Usage: ./makeThumbnailsUtility -eval <dirctory of base files> -write <directory to save thumbnails> -height <max height of thumbnails> -width <max width of the thumbnails> -back [flag to run in background mode] -log [flag to run in log mode]"<<endl;
            exit(0);
        } 
    }
    
	event = "Starting makeFilelistUtility execuatable.";
	helper.logEvent(event, 2, runInBackground, writelog);
    event = "Evaluation Directory: " + evalDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    event = "Filelist will be saved to: " + writeToDirectory;
    helper.logEvent(event, 2, runInBackground, writelog);
    
	
	//make sure that the eval directory exists
	struct stat sb;
	const char * fc = evalDirectory.c_str();
	if (stat(fc, &sb) == 0 && S_ISDIR(sb.st_mode)) {
		isDir = true;
	}
	if (!isDir) {
		event = evalDirectory + " is not a valid directory or you do not have the correct permissions!";
	    helper.logEvent(event, 0, runInBackground, writelog);
	    event = "Operation Aborted!";
	    helper.logEvent(event, 0, runInBackground, writelog);
		exit(0);
	}
	

	helper.StartClock();
	//helper.PrintElapsedClock();

	event = "Reading Eval Directory...";
	helper.logEvent(event, 2, runInBackground, writelog);
	vector<string> files = vector<string>();	
	int totalEvalFiles = 0;
	string numFilesToProcess = "0";
	helper.GetFileList(evalDirectory, files);
	totalEvalFiles = files.size()-2;
	numFilesToProcess = static_cast<ostringstream*>( &(ostringstream() << (totalEvalFiles)) )->str();
	
	event = "Processing " + numFilesToProcess + " files in evaluation directory.";
	helper.logEvent(event, 2, runInBackground, writelog);	
	
	
	int numSuccessFiles = 0;
	int wfcount = 0;
	int impID = startID;
	string evalFileBase;
	string fileToConvert;
	string impListCsvString;
	string fileAssocCsvString;
	string delimString;
	string impListSqlString;
	string fileAssocSqlString;
	for (unsigned int iz = 0;iz < files.size();iz++) {
		fileToConvert = evalDirectory + files[iz];
		event = "Evaluating File " + fileToConvert;
		helper.logEvent(event, 2, runInBackground, writelog);	
    	int isImage = helper.instr(files[iz], "jpg", 0, true);
        if (isImage > 0) {
        	
        	event = fileToConvert + " is a JPG file. ";
        	helper.logEvent(event, 2, runInBackground, writelog);
        	
        	// make sure we have a correct file mask
        	int isDelim = helper.instr(files[iz], "-", 0, true);
        	if (isDelim > 0) {
        		
        		event = fileToConvert + " has the proper filename mask. ";
        		helper.logEvent(event, 2, runInBackground, writelog);
        		
        		// determine wheter to use a single or double split delimiter
        		bool blnGoodFile = true;
        		int isDoubleDelim = helper.instr(files[iz], "--", 0, true);
        		if (isDoubleDelim > 0) {
        			blnGoodFile = false;    		
        		}
        		
        		delimString = "-";
        		if (blnGoodFile) {
        			
        			event = fileToConvert + " is a good file.";
        			helper.logEvent(event, 2, runInBackground, writelog);
        			
	        		vector<string> fileNameParts = helper.Split(delimString, files[iz]);
	        		string balladID = fileNameParts[0];
	        		string orderPart = fileNameParts[1];
	        		orderPart.erase (orderPart.begin() + (orderPart.rfind('.') - 1), orderPart.end());
	        		
	        		event = "All parts extracted for " + fileToConvert;
	        		helper.logEvent(event, 2, runInBackground, writelog);
	        		
	        		string strImpID = static_cast<ostringstream*>( &(ostringstream() << (impID)) )->str();
	        		
	        		impListCsvString = strImpID + "," + files[iz] + "\n";
	        		impListSqlString = "INSERT INTO bia_impressions (BIA_IMP_ID, BIA_IMP_File) VALUES (" + strImpID  + ", '" + files[iz] + "');\n";
	        		fileAssocCsvString = strImpID + "," + balladID + "," + strImpID + "," + orderPart + "\n";
	        		fileAssocSqlString = "INSERT INTO bia_balladImpressions (BIA_BDI_BDID, BIA_BDI_IMPID, BIA_BDI_Number) VALUES (" + balladID  + ", "  + strImpID + ", " + orderPart  + ");\n";
	        		
	        		event = fileToConvert + " exportList 1: " + impListCsvString;
	        		helper.logEvent(event, 2, runInBackground, writelog);
	        		event = fileToConvert + " exportAssoc 2: " + fileAssocCsvString;
	        		helper.logEvent(event, 2, runInBackground, writelog);
	        		
	        		
	        		string strImpressionsFileToWriteTo;
	        		string strAssocFileToWriteTo;
	        		
	        		if (writeMysql) {
	        			strImpressionsFileToWriteTo = writeToDirectory + "impressions.sql";
	        			strAssocFileToWriteTo = writeToDirectory + "balladImpressions.sql";
	        		} else {
	        			strImpressionsFileToWriteTo = writeToDirectory + "impressions.txt";
	        			strAssocFileToWriteTo = writeToDirectory + "balladImpressions.txt";
	        		}
	        		
	        		event = "Trying to write to files [" + strImpressionsFileToWriteTo + "] and [" + strAssocFileToWriteTo + "]";
	        		helper.logEvent(event, 2, runInBackground, writelog);
	        		
	        		if (writeMysql) {
	        			helper.writeStingToFile(impListSqlString, strImpressionsFileToWriteTo);
	        			helper.writeStingToFile(fileAssocSqlString, strAssocFileToWriteTo);
	        			
	        		} else {
	        			helper.writeStingToFile(impListCsvString, strImpressionsFileToWriteTo);
	        			helper.writeStingToFile(fileAssocCsvString, strAssocFileToWriteTo);
	        		}
	        		
	        		event = "Files [" + strImpressionsFileToWriteTo + "] and [" + strAssocFileToWriteTo + "] written successfully.";
	        		helper.logEvent(event, 2, runInBackground, writelog);
        		
        		}
        		
        	}
        	
        	impID++;

        }
	}



	event = "./makeFilelistUtility Process Completed.";
	helper.logEvent(event, 2, runInBackground, writelog);
	event = "Bye, Bye!";
	helper.logEvent(event, 2, runInBackground, writelog);

	return 0;

}