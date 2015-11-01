CC=mpic++
CXXFLAGS=-m64 -pipe -Ofast -march=native -flto -fwhole-program -std=c++0x -Wall -W


all: fciqmc

hubbard: fciqmc.cc fciqmc.hh mpi_data.hh
	$(CC) -DUSEMPI -DHUBBARD $(CXXFLAGS) $< -o $@

run: hubbard
	mpirun -np 4 fciqmc
