#include <map>
#include <string>
#include <iostream>

using namespace std;

#include "mpi_data.hh"

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	/*
	constexpr int n = 10;

	MPI_Datatype mpi_state_type;
	MPI_Type_contiguous(n, MPI_UINT8_T, &mpi_state_type);
	MPI_Type_commit(&mpi_state_type);


	MPI_Allgather(&beg, 1, mp_type, m_begins.data(), 1, mp_type, MPI_COMM_WORLD);


	MPI_Finalize();
	return 0;


	int a = mpi_rank;
	int b[mpi_size];

	MPI_Allgather(&a, 1, MPI_INT, b, 1, MPI_INT, MPI_COMM_WORLD);

	cout << mpi_rank << ": ";
	for (int i = 0; i < mpi_size; ++i) cout << b[i] << ' ';
	cout << endl;

	MPI_Finalize();
	return 0;*/

	mpi_data<int> ys(mpi_rank, mpi_size, MPI_INT);

	map<int, int> xs;
	for (int n = 0; n < 12; ++n) {
		xs.clear();
		xs[rand()%100] = mpi_rank;
		xs[rand()%100] = mpi_rank;
		xs[rand()%100] = mpi_rank;
		xs[rand()%100] = mpi_rank;

		ys.sync(xs);
	}

	cout << mpi_rank << ": " << ys.begin()->first << " -> " << (--ys.end())->first << " (" << ys.size() << ")"<<  endl;

	MPI_Finalize();

	return 0;
}
