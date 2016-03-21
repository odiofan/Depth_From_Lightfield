TEMPLATE	= app
CONFIG		+= qt debug
QT          += widgets
CONFIG      += link_pkgconfig

QT_CONFIG -= no-pkg-config
CONFIG    += link_pkgconfig

PKGCONFIG += opencv

HEADERS		= src/*.h\
			  src/gco/*.h\
			  src/WMF/*.h			  
SOURCES		= src/*.cpp\
			  src/gco/*.cpp\
                             
TARGET		= lf2depth


# Compiler flags tuned for my system
QMAKE_CXXFLAGS +=  -O3 -pipe -g -Wall -frounding-math -fsignaling-nans -fopenmp
linux-g++: QMAKE_CXXFLAGS += -O99 

LIBS +=  -lgsl -lgslcblas -lhdf5 -lhdf5_hl -lz -fopenmp
LIBS +=  -L/usr/local/lib

OBJECTS_DIR = obj
DESTDIR     = bin

