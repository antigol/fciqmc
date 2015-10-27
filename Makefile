CC=mpic++
CXXFLAGS=-m64 -pipe -Ofast -march=native -flto -fwhole-program -std=c++0x -Wall -W


all: hubbard

hubbard: hubbard.cc fciqmc.hh mpi_data.hh
	$(CC) -DUSEMPI $(CXXFLAGS) $< -o $@

run: hubbard
	mpirun -np 4 hubbard
