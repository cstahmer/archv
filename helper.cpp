/*
 * ArchiveVision: Compter Vision Engine for Digital Archives
 * 
 * file: helper.cpp
 *
 * Contains classes and functions for various file system
 * operations
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

#include "helper.h"
#include <fstream>

#ifdef __APPLE__
	#include <unistd.h>
#endif

using namespace std;


Helper::Helper()
{
}


Helper::~Helper()
{
}

/*
 * The method writes Mat object to disk.
 * 
 *  fileName - Name of the file to be written to disk.
 *                                                             
 *  structure - Mat object to be written to disk.
 *  structureName - structure name to be used to write the file.
 *
 *  PreCondition - The Mat object to be written to disk is and the filename are passed in as arguments.
 *                 
 *  PostCondition - The Mat object is written to file and saved as the file name passed in as argument.
 *                  
 */
void Helper::WriteToFile(string fileName, Mat structure, string structureName)
{
	try {
		FileStorage fs(fileName, FileStorage::WRITE);
		fs<< structureName << structure;
		fs.release();
	} catch (int e) {
	    cout << "Unable to write file " << fileName << ".  Exception " << e << "." << endl << "Operation Aborted!" << endl;
	}
}


/*
 * The method reads Mat object from disk.
 * 
 *  fileName - Name of the file to be read from disk.
 *                                                             
 *  structure - Mat object to read data in.
 *
 *  structureName - structure name to be used to read the file.
 *
 *  return Mat read from the disk.
 *
 *  PreCondition - The Mat object to be read from disk and the filename are passed in as arguments.
 *                 
 *  PostCondition - The Mat object is read from file and returned.
 *                  
 */

Mat Helper::ReadFromFile(string fileName, Mat structure, string structureName)
{
	FileStorage fs(fileName, FileStorage::READ);
	fs[structureName] >> structure;
	fs.release();
	return structure;
}

Mat Helper::ReadMatFromFile(string fileName, string structureName)
{
	Mat structure;
	FileStorage fs(fileName, FileStorage::READ);
	fs[structureName] >> structure;
	fs.release();
	return structure;
}

/*
 * The method prints the elapsed time.
 * 
 *  PreCondition - The time t1 is already set by StartClock method.
 *                 
 *  PostCondition - The method prints the difference between start time and end time.
 *                  
 */
clock_t t1, t2, t3;
void Helper::StopAndPrintClock() 
{
	t2=clock();
	float diff = ((float)t2 - (float)t1) / 1000000.0F;
	cout << "  Total Processing Time: " << diff << " seconds" << endl;
	//cout << " Time took : " << diff << " seconds\n\n" << endl;
}

/*
 * The method prints the elapsed time.
 * 
 *  PreCondition - The time t1 is already set by StartClock method.
 *                 
 *  PostCondition - The method prints the difference between start time and end time.
 *                  
 */
void Helper::PrintElapsedClock() 
{
	t3=clock();
	float diff = ((float)t3 - (float)t1) / 1000000.0F;
	cout << "  Elapsed Processing Time: " << diff << " seconds" << endl;
}


/*
 * The method captures the start time as t1 to calculate total elapsed time.
 * 
 *  PreCondition - The time t1 object is created.
 *                 
 *  PostCondition - The time t1 is set to current time.
 *                  
 */
void Helper::StartClock() 
{
	t1 = clock();
}

int Helper::GetFileList(string directory, vector<string> &files) {
	
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

int Helper::instr( string str, string toFind, int start = 0, bool ignoreCase = true ) {

	/*
	size_t found;
	found=str.find(toFind);
	int foundInt = int(found);
	cout << "Find '" << toFind << "' in '" << str << "'result is: " << foundInt << endl;
	return int(foundInt);
	*/
	
		for( int i(start); i < str.length(); i++ ) {
			if( !ignoreCase ) {
				if( int(str[i]) == int(toFind[0]) ) {
				
					bool found = true;
					int counter = 0;
					for( int j(1); j < toFind.length(); j++ ) { 
						
						if( (i+j) < str.length() ) {
							if( int(str[i+j]) != int(toFind[0+j]) )
								found = false;
	                    } else
							counter++;
						}
								
					if( found && !counter ) return i;	
				}
			} else {
			
				if( int(tolower(str[i])) == int(tolower(toFind[0])) ) {
					
					bool found = true;
					int counter = 0;
					
					for( int j(1); j < toFind.length(); j++ ) { 
						if( (i+j) < str.length() ) { 
							if( int(tolower(str[i+j])) != int(tolower(toFind[0+j])) )
								found = false;
								
	                    } else
							counter++;		
	                }
				
					if( found && !counter ) return i;
				}		
			}
		}
		
		return -1;
	
}


//code that writes out MAT data as yml text file
int Helper::writeTextFile(string strdata, string filename) {
	int success = 0;
	const char * fileToCreate = filename.c_str ();
	const char * fileData = strdata.c_str ();
	FILE * pFile;
	pFile = fopen (fileToCreate,"w");
	if (pFile!=NULL) {
		fputs (fileData,pFile);
		fclose (pFile);
		success = 1;
	}
	
	return success;
}

//logging class class
void Helper::logEvent(string strEvent, int eType, bool background, bool writelog) {
	string eventType;
	if (eType == 0) {
		eventType = "ERROR!!";
	} else if (eType == 1) {
		eventType = "WARNING";
	} else if (eType == 2) {
		eventType = "SYSSTAT";
	} else if (eType == 3) {
		eventType = "FAILURE";
	} else if (eType == 4) {
		eventType = "SUCCESS";
	} else {
		eventType = "UNKNOWN";
	}

	
	time_t t = time(NULL);
	tm* timePtr = localtime(&t);
	
	string yearstring = static_cast<ostringstream*>( &(ostringstream() << ((timePtr->tm_year) + 1900)) )->str();
	string monthstring = static_cast<ostringstream*>( &(ostringstream() << ((timePtr->tm_mon) + 1)) )->str();
	string dayofmonthstring = static_cast<ostringstream*>( &(ostringstream() << ((timePtr->tm_mday) + 1)) )->str();
	string hourstring = static_cast<ostringstream*>( &(ostringstream() << ((timePtr->tm_hour) + 1)) )->str();
	string minutesstring = static_cast<ostringstream*>( &(ostringstream() << ((timePtr->tm_min) + 1)) )->str();
	string secondsstring = static_cast<ostringstream*>( &(ostringstream() << ((timePtr->tm_sec) + 1)) )->str();
	string logFileName = LOG_PATH + yearstring + monthstring + ".txt";
	
	string appendString;
	if (hourstring.length() == 1) {
		appendString = "0";
		hourstring = appendString + hourstring;
	}
	if (minutesstring.length() == 1) {
		appendString = "0";
		minutesstring = appendString + minutesstring;
	}
	if (secondsstring.length() == 1) {
		appendString = "0";
		secondsstring = appendString + secondsstring;			
	}
	
	string timeStamp = monthstring + "/" + dayofmonthstring + "/" + yearstring + "\t" + hourstring + ":" + minutesstring + ":" + secondsstring;
	
	string strScreenEvent = timeStamp + "\t" + eventType + "\t" + strEvent;
	string strLogEvent = strScreenEvent + "\n";
	
	if (writelog) {
		const char * fileToWrietTo = logFileName.c_str();
		const char * fileData = strLogEvent.c_str();
		FILE * pFile;
		if (access (logFileName.c_str (), F_OK) != 0) {
			pFile = fopen(fileToWrietTo,"w");
		} else {
			pFile = fopen(fileToWrietTo,"a");
		}
		if (pFile!=NULL) {
			fputs (fileData,pFile);
			fclose (pFile);
		}	
	}
	
	if (!background) {
		cout << strScreenEvent << endl;
	}
	
}

// split: receives a char delimiter; returns a vector of strings
// By default ignores repeated delimiters, unless argument rep == 1.
vector<string> Helper::Split(string delim, string baseString) {
	vector<string> flds;
    string work = baseString;
    string buff = "";
    int i = 0;
    bool newWord = true;
    while (i < work.length()) {
    	stringstream ss;
    	string thisChar;
    	ss << work[i];
    	ss >> thisChar;
    	if (thisChar != delim) {
    		buff = buff + thisChar;
        } else {
        	if (buff.length() > 0) {
        		flds.push_back(buff);
        		buff = "";
        	}
        }
        i++;
    }
	if (buff.length() > 0) {
		flds.push_back(buff);
	}   
    
    //if (!buf.empty())
      //  flds.push_back(buff);
    return flds;
}


// simple file write
//logging class class
void Helper::writeStingToFile(string strData, string strFilename) {
	
		const char * fileToWrietTo = strFilename.c_str();
		const char * fileData = strData.c_str();
		FILE * pFile;
		if (access (fileToWrietTo, F_OK) != 0) {
			pFile = fopen(fileToWrietTo,"w");
		} else {
			pFile = fopen(fileToWrietTo,"a");
		}
		if (pFile!=NULL) {
			fputs (fileData,pFile);
			fclose (pFile);
		}	

	
}


