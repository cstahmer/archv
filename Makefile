#
EXT=.exe
NAME=showkeypoints
NAME2=multi
NAME3=calibrate
NAME4=dictionary
NAME5=histograms
NAME6=compare2images
NAME7=detectimages
DIR=.
NAMEFUL=$(DIR)/$(NAME)$(EXT)
NAMEFUL2=$(DIR)/$(NAME2)$(EXT)
NAMEFUL3=$(DIR)/$(NAME3)$(EXT)
NAMEFUL4=$(DIR)/$(NAME4)$(EXT)
NAMEFUL5=$(DIR)/$(NAME5)$(EXT)
NAMEFUL6=$(DIR)/$(NAME6)$(EXT)
NAMEFUL7=$(DIR)/$(NAME7)$(EXT)

CC = g++
CFLAGS = -c -O -std=c++11
LDFLAGS = -O 
LIBS=-L/usr/local/lib
LIBRARIES=-lopencv_core -lopencv_nonfree -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_flann -lopencv_contrib -lopencv_ml -lopencv_objdetect -lopencv_video -lopencv_videostab -lopencv_calib3d -lopencv_ocl -lopencv_photo -lopencv_stitching

.cpp.o :
	$(CC) $(CFLAGS) $<

OBJECTS = \
$(NAME).o 

OBJECTS2 = \
$(NAME2).o 

OBJECTS3 = \
$(NAME3).o 

OBJECTS4 = \
$(NAME4).o 

OBJECTS5 = \
$(NAME5).o 

OBJECTS6 = \
$(NAME6).o 

OBJECTS7 = \
$(NAME7).o 

$(NAMEFUL) : $(OBJECTS)
	$(CC) -o $(NAMEFUL) $(LDFLAGS) $(OBJECTS) $(LIBS) $(LIBRARIES)

$(NAMEFUL2) : $(OBJECTS2)
	$(CC) -o $(NAMEFUL2) $(LDFLAGS) $(OBJECTS2) $(LIBS) $(LIBRARIES)

$(NAMEFUL3) : $(OBJECTS3)
	$(CC) -o $(NAMEFUL3) $(LDFLAGS) $(OBJECTS3) $(LIBS) $(LIBRARIES)

$(NAMEFUL4) : $(OBJECTS4)
	$(CC) -o $(NAMEFUL4) $(LDFLAGS) $(OBJECTS4) $(LIBS) $(LIBRARIES)

$(NAMEFUL5) : $(OBJECTS5)
	$(CC) -o $(NAMEFUL5) $(LDFLAGS) $(OBJECTS5) $(LIBS) $(LIBRARIES)

$(NAMEFUL6) : $(OBJECTS6)
	$(CC) -o $(NAMEFUL6) $(LDFLAGS) $(OBJECTS6) $(LIBS) $(LIBRARIES)

$(NAMEFUL7) : $(OBJECTS7)
	$(CC) -o $(NAMEFUL7) $(LDFLAGS) $(OBJECTS7) $(LIBS) $(LIBRARIES)

all: $(OBJECTS) $(OBJECTS2) $(OBJECTS3) $(OBJECTS4) $(OBJECTS5) $(OBJECTS6) $(OBJECTS7)
	$(CC) -o $(NAMEFUL) $(LDFLAGS) $(OBJECTS) $(LIBS) $(LIBRARIES)
	$(CC) -o $(NAMEFUL2) $(LDFLAGS) $(OBJECTS2) $(LIBS) $(LIBRARIES)
	$(CC) -o $(NAMEFUL3) $(LDFLAGS) $(OBJECTS3) $(LIBS) $(LIBRARIES)
	$(CC) -o $(NAMEFUL4) $(LDFLAGS) $(OBJECTS4) $(LIBS) $(LIBRARIES)
	$(CC) -o $(NAMEFUL5) $(LDFLAGS) $(OBJECTS5) $(LIBS) $(LIBRARIES)
	$(CC) -o $(NAMEFUL6) $(LDFLAGS) $(OBJECTS6) $(LIBS) $(LIBRARIES)
	$(CC) -o $(NAMEFUL7) $(LDFLAGS) $(OBJECTS7) $(LIBS) $(LIBRARIES)

clean:
	touch junk.o; rm -f *.o *.exe

