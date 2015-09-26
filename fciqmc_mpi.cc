#include <unordered_map>
#include <array>
#include <random>
#include <limits>
#include <vector>
#include <sstream>
#include <thread>
#include <iostream>
#include "fciqmc.hh"

#undef TIMING
//#define TIMING
#undef USEMPI
//#define USEMPI

#ifdef TIMING
#include <chrono>
#endif

#ifdef USEMPI
#include <mpi/mpi.h>
#endif

using namespace std;

// Hubbard model
// H = -t \sum_{<i,j>,s} s^dag_i s_j + s^dag_j s_i + U \sum_i nup_i ndown_j

#define U 1
#define t 1
#define n (16*2)
#define GETSPIN(s, k) (s[(k>>4)] >> (k&0xf << 1))
#define SWAPSPIN(s, k, v) s[(k>>4)] ^= (v << (k&0xf << 1))
typedef array<uint32_t, n/16> state_type;
// 0 00b  nothing
// 1 01b  up
// 2 10b  down
// 3 11b  up & down

struct KeyHasher {
	size_t operator()(const state_type& state) const
	{
		if (state.size() > 1) return state[0] | ((size_t)state[1] << 32);
		else return state[0];
	}
};

#ifdef USEMPI
string mpi_serialize(unordered_map<state_type, int>::const_iterator begin, unordered_map<state_type, int>::const_iterator end)
{
	ostringstream oss;

	while (begin != end) {
		const state_type& state = begin->first;
		int psip = begin->second;

		for (size_t k = 0; k < state.size(); ++k) oss << state[k];
		oss << ' ' << psip << ' ';

		begin++;
	}
	return oss.str();
}

void mpi_unserialize(string str, unordered_map<state_type, int>& map)
{
	istringstream iss(str);
	while (true) {
		state_type state;
		int psip;
		char ch;
		for (size_t k = 0; k < state.size(); ++k) {
			iss.get(ch);
			state[k] = ch-'0';
		}
		iss >> psip;
		if (iss.eof()) break;

		map[state] += psip;
	}
}

constexpr int tag_amount = 0;
constexpr int tag_length = 1;
constexpr int tag_data = 2;
constexpr int tag_energyshift = 3;

void mpi_recv_walkers(int source, unordered_map<state_type, int>& map)
{
	int amount;
	MPI_Recv(&amount, 1, MPI_INT, source, tag_amount, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	for (int i = 0; i < amount; ++i) {
		int len;
		MPI_Recv(&len, 1, MPI_INT, source, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		char data[len];
		MPI_Recv(data, len, MPI_BYTE, source, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		mpi_unserialize(string(data, len), map);
	}
}

unordered_map<state_type, int>::iterator
mpi_send_walkers(int dest, unordered_map<state_type, int>::iterator begin, int amount)
{
	constexpr int max = 2*1024*1024/N;
	const int n = (amount + max - 1) / max;
	MPI_Send((void *)&n, 1, MPI_INT, dest, tag_amount, MPI_COMM_WORLD);
	if (n == 0) return begin;

	const int a = amount / n;
	const int r = amount % n;
	unordered_map<state_type, int>::iterator end = begin;

	for (int i = 0; i < n; ++i) {
		advance(end, a); if (i < r) end++;
		string serial_str = mpi_serialize(begin, end);
		begin = end;

		int len = serial_str.size();
		MPI_Send(&len, 1, MPI_INT, dest, tag_length, MPI_COMM_WORLD);
		MPI_Send((void *)serial_str.data(), len, MPI_BYTE, dest, tag_data, MPI_COMM_WORLD);
	}
	return begin;
}

void mpi_split_send_walkers(int size, unordered_map<state_type, int>& map)
{
	const int states_per_rank = map.size() / size;
	const int states_rest     = map.size() % size;

	auto i = map.begin();
	for (int r = 1; r < size; ++r) {
		int amount = states_per_rank;
		if (r <= states_rest) amount++;

		i = mpi_send_walkers(r, i, amount);
	}
	map.erase(map.begin(), i);
}
#endif

int main(int argc, char* argv[])
{
	using namespace std::chrono;
#ifdef USEMPI
	MPI_Init(&argc, &argv);

	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
	int mpi_rank = 0;
	(void)argc;
	(void)argv;
#endif

	int thr_size = thread::hardware_concurrency();
	cout << "Number of thread on this socket " << thr_size << endl;
	if (thr_size == 0) thr_size = 1;

	double delta_time = 0.002;
	double energyshift = 50.0;
	double damping = 0.10;

	unordered_map<state_type, int, KeyHasher> walkers;
	typedef unordered_map<state_type, int, KeyHasher>::iterator type_it;

	walkers.reserve(10000);

	// Initial population
	if (mpi_rank == 0) {
		state_type psi;
		for (size_t k = 0; k < psi.size(); ++k) {
			psi[k] = 0x99999999; // up, down, up, down...
		}
		walkers[psi] = 1;

		cout << mpi_rank << ": " << walkers.size() << " states initialy created" << endl;
	}

	vector<vector<pair<state_type, int>>> changes(thr_size);

	// Main loop
	for (int iter = 0; iter < 1000; ++iter) {

#ifdef USEMPI
		if (mpi_rank == 0) {
			mpi_split_send_walkers(mpi_size, walkers);
		} else {
			mpi_recv_walkers(0, walkers);
		}
#endif

#ifdef TIMING
		auto t1 = high_resolution_clock::now();
#endif

		const int thr_amt = walkers.size() / thr_size;
		const int thr_rst = walkers.size() % thr_size;

		for (auto& x : changes) x.clear();

		cout << "run threades... " << flush;
		vector<thread> thr_list;
		type_it begin = walkers.begin();
		for (int thr_i = 0; thr_i < thr_size; ++thr_i) {
			type_it end = begin;
			advance(end, thr_amt);
			if (thr_i < thr_rst) end++;

			auto lambda = [&, thr_i, begin, end]() {
				for (auto i = begin; i != end; ++i) {
					const state_type& statei = i->first;
					const int psipi = i->second;

					// (1) Spawning
					for (size_t k0 = 0; k0 < n; ++k0) {
						const size_t k1 = (k0+1)%n;
						// 0 00b  nothing
						// 1 01b  up
						// 2 10b  down
						// 3 11b  up & down
						if ((GETSPIN(statei, k0) & 0x1) ^ (GETSPIN(statei,k1) & 0x1)) {
							state_type statej = statei;
							SWAPSPIN(statej, k0, 0x1);
							SWAPSPIN(statej, k1, 0x1);
							const double H = -t;
							const double T = -H;
							changes[thr_i].push_back(make_pair(statej, binomial_throw(psipi, T * delta_time)));
						}
						if ((GETSPIN(statei, k0) & 0x2) ^ (GETSPIN(statei,k1) & 0x2)) {
							state_type statej = statei;
							SWAPSPIN(statej, k0, 0x2);
							SWAPSPIN(statej, k1, 0x2);
							const double H = -t;
							const double T = -H;
							changes[thr_i].push_back(make_pair(statej, binomial_throw(psipi, T * delta_time)));
						}
					}


					// (2) Diagonal
					double H = 0.0;
					for (size_t k = 0; k < n; ++k) {
						if ((GETSPIN(statei, k)&0x3) == 3) H += U;
					}
					const double T = -(H - energyshift);
					changes[thr_i].push_back(make_pair(statei, binomial_throw(psipi, clamp(-1.0, T * delta_time, 1.0))));
				}
			};
			begin = end;

			//thr_list.push_back(thread(lambda));
			lambda();
		}

		cout << "wait... " << flush;
		for (auto &x : thr_list) x.join();
		cout << "done" << endl;

		// (3) Annihilation
		for (const auto &xx : changes) {
			for (const auto &x : xx) {
				walkers[x.first] += x.second;
			}
		}

#ifdef TIMING
		auto t2 = high_resolution_clock::now();
		cout << mpi_rank << "@" << iter << ": physics " << 1000.0*duration_cast<duration<double>>(t2 - t1).count() << " ms" << endl;
#endif

		size_t count_total_walkers = 0;

		if (mpi_rank == 0) {
#ifdef USEMPI
			for (int r = 1; r < mpi_size; ++r) {
				mpi_recv_walkers(r, walkers);
			}
#endif

			int min_per_state = 0;
			int max_per_state = 0;
			for (auto i = walkers.begin(); i != walkers.end(); ) {
				count_total_walkers += abs(i->second);
				min_per_state = min(min_per_state, i->second);
				max_per_state = max(max_per_state, i->second);

				if (i->second == 0) {
					i = walkers.erase(i);
				} else {
					++i;
				}
			}
			cout << mpi_rank << "@" << iter << ": " << count_total_walkers << " walkers in " << walkers.size() << " states. pips from " <<
							min_per_state << " to " << max_per_state << endl;
		} else {
#ifdef USEMPI
			mpi_send_walkers(0, walkers.begin(), walkers.size());
			walkers.clear();
#endif
		}



		constexpr int A = 5;
		if (iter > 50 && iter%A == 0) {
			if (mpi_rank == 0) {
				static double last_count_total_walkers = count_total_walkers;
				energyshift -= damping / (A * delta_time) * log(count_total_walkers / last_count_total_walkers);
				cout << mpi_rank << "@" << iter << ": energyshift = " << energyshift << endl;
				last_count_total_walkers = count_total_walkers;
#ifdef USEMPI
				for (int r = 1; r < mpi_size; ++r)
					MPI_Send(&energyshift, 1, MPI_DOUBLE, r, tag_energyshift, MPI_COMM_WORLD);
				// todo compute the energy
			} else {
				MPI_Recv(&energyshift, 1, MPI_DOUBLE, 0, tag_energyshift, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
			}
		}

#ifdef TIMING
		auto t3 = high_resolution_clock::now();
		cout << mpi_rank << "@" << iter << ": mpi " << 1000.0*duration_cast<duration<double>>(t3 - t2).count() << " ms" << endl;
#endif
	}

#ifdef USEMPI
	MPI_Finalize();
#endif

	return 0;
}
