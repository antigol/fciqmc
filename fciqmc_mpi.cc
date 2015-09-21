#include <unordered_map>
#include <bitset>
#include <random>
#include <limits>
#include <vector>
#include <sstream>
#include <thread>
#include <iostream>

#undef TIMING
//#define TIMING
#undef USEMPI
#define USEMPI

#ifdef TIMING
#include <chrono>
#endif

#ifdef USEMPI
#include <mpi/mpi.h>
#endif

using namespace std;

inline std::default_random_engine& global_random_engine()
{
	static std::random_device rdev;
	static std::default_random_engine eng(rdev());
	return eng;
}

inline double canonical()
{
	return std::generate_canonical<double, std::numeric_limits<double>::digits>(global_random_engine());
}

inline int random_round(double x)
{
	int a = floor(x);
	if (canonical() < (x-a)) ++a;
	return a;
}

int binomial_throw(int n, double p)
{
	if (n < 0) {
		n = -n;
		p = -p;
	}
	int a = floor(p);
	binomial_distribution<int> distribution(n, p-a);
	return n * a + distribution(global_random_engine());
}

template<class T>
T clamp(T a, T x, T b)
{
	if (x < a) return a;
	if (x > b) return b;
	return x;
}

#define N 50
typedef bitset<N> state_type;

double hamiltonian_ii(const state_type& i)
{
	double E = 0.0;
	for (int k0 = 0; k0 < N; ++k0) {
		int k1 = (k0+1)%N;
		if (i[k0] == i[k1]) E += 0.25;
		else E -= 0.25;
	}
	return E;
}

double hamiltonian_ij(const state_type& i, const state_type& j)
{
	if (i == j) return hamiltonian_ii(i);

	int c0 = 0;
	int c1 = 0;
	for (int k = 0; k < N; ++k) if (i[k] != j[k]) {
		c0 = k;
		break;
	}
	for (int k = c0+1; k < N; ++k) if (i[k] != j[k]) {
		c1 = k;
		break;
	}
	for (int k = c1+1; k < N; ++k) if (i[k] != j[k]) return 0.0;

	if (c0+1 == c1) return 0.5;
	if (c0 == 0 && c1 == N-1) return 0.5;
	return 0.0;
}

#ifdef USEMPI
string mpi_serialize(unordered_map<state_type, int>::const_iterator begin, unordered_map<state_type, int>::const_iterator end)
{
	ostringstream oss;

	while (begin != end) {
		const state_type& state = begin->first;
		int psip = begin->second;

		oss << state.to_string() << ' ' << psip << ' ';

		begin++;
	}
	return oss.str();
}

void mpi_unserialize(string str, unordered_map<state_type, int>& map)
{
	istringstream iss(str);
	while (true) {
		string state_str;
		int psip;

		iss >> state_str >> psip;
		if (iss.eof()) break;

		map[state_type(state_str)] += psip;
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
	double energyshift = 15.0;
	double damping = 0.15;

	unordered_map<state_type, int> walkers;
	typedef unordered_map<state_type, int>::iterator type_it;

	// Initial population
	if (mpi_rank == 0) {
		state_type psi;
		for (int n = 0; n <= N; ++n) {
			for (int k = 0; k < N; ++k) psi[k] = (k < n);
			walkers[psi] = 5;
		}

		cout << mpi_rank << ": " << walkers.size() << " states initialy created" << endl;
	}

	vector<vector<pair<state_type, int>>> changes(thr_size);

	// Main loop
	for (int iter = 0; iter < 2000; ++iter) {

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

		vector<thread> thr_list;
		type_it it = walkers.begin();
		for (int thr_i = 0; thr_i < thr_size; ++thr_i) {
			type_it it_end = it;
			advance(it_end, thr_amt);
			if (thr_i < thr_rst) it_end++;

			thr_list.push_back(thread([&, thr_i](type_it begin, type_it end) {
				for (auto i = begin; i != end; ++i) {
					const state_type& statei = i->first;
					const int psipi = i->second;

					// (1) Spawning
					for (int k0 = 0; k0 < N; ++k0) {
						const int k1 = (k0+1)%N;
						if (statei[k0] == statei[k1]) continue; // otherwise H = 0
						state_type statej = statei;
						statej[k0] = statei[k1];
						statej[k1] = statei[k0];
						const double H = 0.5;
						const double T = -H;
						changes[thr_i].push_back(make_pair(statej, binomial_throw(psipi, T * delta_time)));
					}


					// (2) Diagonal
					const double H = hamiltonian_ii(statei);
					const double T = -(H - energyshift);
					changes[thr_i].push_back(make_pair(statei, binomial_throw(psipi, clamp(-1.0, T * delta_time, 1.0))));
				}
			}, it, it_end));
			it = it_end;
		}

		for (auto &x : thr_list) x.join();

		// (3) Annihilation
		for (auto &xx : changes) {
			for (auto &x : xx) {
				walkers[x.first] += x.second;
			}
		}

#ifdef TIMING
		auto t2 = high_resolution_clock::now();
		cout << mpi_rank << "@" << (iter+1) << ": physics " << 1000.0*duration_cast<duration<double>>(t2 - t1).count() << " ms" << endl;
#endif

		size_t count_total_walkers = 0;

		if (mpi_rank == 0) {
#ifdef USEMPI
			for (int r = 1; r < mpi_size; ++r) {
				mpi_recv_walkers(r, walkers);
			}
#endif

			for (auto i = walkers.begin(); i != walkers.end(); ) {
				count_total_walkers += abs(i->second);

				if (i->second == 0) {
					i = walkers.erase(i);
				} else {
					++i;
				}
			}
			cout << mpi_rank << "@" << (iter+1) << ": " << count_total_walkers << " walkers" << endl;
		} else {
#ifdef USEMPI
			mpi_send_walkers(0, walkers.begin(), walkers.size());
			walkers.clear();
#endif
		}



		constexpr int A = 10;
		if (iter > 100 && iter%A == 0) {
			if (mpi_rank == 0) {
				static double last_count_total_walkers = count_total_walkers;
				energyshift -= damping / (A * delta_time) * log(count_total_walkers / last_count_total_walkers);
				cout << mpi_rank << "@" << (iter+1) << ": energyshift = " << energyshift << endl;
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
		cout << mpi_rank << "@" << (iter+1) << ": mpi " << 1000.0*duration_cast<duration<double>>(t3 - t2).count() << " ms" << endl;
#endif
	}

#ifdef USEMPI
	MPI_Finalize();
#endif

	return 0;
}
