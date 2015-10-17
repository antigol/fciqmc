TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11
LIBS += -pthread

SOURCES += hubbard.cc

HEADERS += fciqmc.hh \
	mpi_data.hh

QMAKE_CXXFLAGS_RELEASE -= -O
QMAKE_CXXFLAGS_RELEASE -= -O1
QMAKE_CXXFLAGS_RELEASE -= -O2

QMAKE_CXXFLAGS_RELEASE *= -Ofast
QMAKE_CXXFLAGS_RELEASE *= -march=native
QMAKE_CXXFLAGS_RELEASE *= -flto -fwhole-program

QMAKE_LFLAGS_RELEASE *= -flto -fwhole-program
