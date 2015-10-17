#include <map>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

string serialize(const pair<int,int>& kv)
{
	ostringstream oss;
	oss << kv.first << ' ' << kv.second;
	return oss.str();
}

pair<int,int> unserialize(const string& str)
{
	istringstream iss(str);
	pair<int,int> kv;
	iss >> kv.first >> kv.second;
	return kv;
}

#include "mpi_map.hh"

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	/*int a = mpi_rank;
	int b[mpi_size];

	MPI_Allgather(&a, 1, MPI_INT, b, 1, MPI_INT, MPI_COMM_WORLD);

	cout << mpi_rank << ": ";
	for (int i = 0; i < mpi_size; ++i) cout << b[i] << ' ';
	cout << endl;

	MPI_Finalize();
	return 0;*/

	mpi_map<int> ys(mpi_rank, mpi_size);

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
