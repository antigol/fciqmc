#include <unordered_map>
#include <bitset>
#include <random>
#include <limits>
#include <chrono>
#include <vector>
#include <sstream>
#include <mpi/mpi.h>

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

#define N 100
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

	int c0, c1;
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

int random_round(double x)
{
	int a = x;
	if (canonical() < (x-a)) ++a;
	return a;
}

string mpi_serialize(unordered_map<state_type, pair<int, int>>::const_iterator begin, unordered_map<state_type, pair<int, int>>::const_iterator end)
{
	ostringstream oss;

	while (begin != end) {
		const state_type& state = begin->first;
		const pair<int, int>& psip = begin->second;

		oss << state.to_string() << ' ' << psip.first << ' ' << psip.second << ' ';

		begin++;
	}
	return oss.str();
}

void mpi_unserialize(string str, unordered_map<state_type, pair<int, int>>& map)
{
	istringstream iss(str);
	while (true) {
		string state_str;
		int first, second;

		iss >> state_str >> first >> second;
		if (iss.eof()) break;

		pair<int, int> &psip = map[state_type(state_str)];
		psip.first  += first;
		psip.second += second;
	}
}

constexpr int tag_length = 0;
constexpr int tag_data = 1;

void mpi_recv_walkers(int source, unordered_map<state_type, pair<int, int>>& map)
{
	int len;
	MPI_Recv(&len, 1, MPI_INT, source, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	char data[len];
	MPI_Recv(data, len, MPI_BYTE, source, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	mpi_unserialize(string(data, len), map);
}

void mpi_send_walkers(int dest, unordered_map<state_type, pair<int, int>>::const_iterator begin, unordered_map<state_type, pair<int, int>>::const_iterator end)
{
	string serial_str = mpi_serialize(begin, end);
	int len = serial_str.size();
	int s1, s2;
	MPI_Pack_size(1,   MPI_INT,  MPI_COMM_WORLD, &s1);
	MPI_Pack_size(len, MPI_BYTE, MPI_COMM_WORLD, &s2);
	int size = 2 * MPI_BSEND_OVERHEAD + s1 + s2;
	cout << size/1024.0/1024.0 << " Mb data" << endl;
	//void *buffer = malloc(size);
	//MPI_Buffer_attach(buffer, size);
	MPI_Send(&len, 1, MPI_INT, dest, tag_length, MPI_COMM_WORLD);
	MPI_Send((void *)serial_str.data(), len, MPI_BYTE, dest, tag_data, MPI_COMM_WORLD);
	//MPI_Buffer_detach(&buffer, &size);
	//free(buffer);
}

void mpi_split_send_walkers(int size, unordered_map<state_type, pair<int, int>>& map)
{
	int states_per_rank = map.size() / size;
	int states_rest     = map.size() % size;

	auto i = map.begin();
	for (int r = 1; r < size; ++r) {
		int amount = states_per_rank;
		if (r <= states_rest) amount++;

		auto iend = i;
		advance(iend, amount);
		mpi_send_walkers(r, i, iend);
		i = iend;
	}
	map.erase(map.begin(), i);
}

int main(int argc, char* argv[])
{
	using namespace std::chrono;
	int rc;
	rc = MPI_Init(&argc, &argv);

	int mpi_size;
	rc = MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int mpi_rank;
	rc = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	double delta_time = 0.2 / N;
	double shift = 20.5;

	unordered_map<state_type, pair<int, int>> walkers;

	// Initial population
	if (mpi_rank == 0) {
		state_type psi;
		for (int n = 0; n <= N; ++n) {
			for (int k = 0; k < N; ++k) psi[k] = (k < n);
			walkers[psi].first = 5;
		}

		cout << mpi_rank << ": " << walkers.size() << " states initialy created" << endl;

		mpi_split_send_walkers(mpi_size, walkers);
	} else {
		mpi_recv_walkers(0, walkers);
	}

	cout << mpi_rank << ": I have " << walkers.size() << " states" << endl;

	// Main loop
	for (int iter = 0; iter < 10000; ++iter) {
		auto t1 = high_resolution_clock::now();

		// (1) Spawning
		vector<pair<state_type, pair<int,int>>> spawns_list;
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& statei = i->first;
			const pair<int,int>& psipi = i->second;
			for (int k0 = 0; k0 < N; ++k0) {
				int k1 = (k0+1)%N;
				if (statei[k0] == statei[k1]) continue;
				state_type statej = statei;
				statej[k0] = statei[k1];
				statej[k1] = statei[k0];
				const double H = 0.5;
				const double T = -H;
				pair<int,int> psipj = make_pair(0, 0);
				if (T > 0.0) {
					// spawn with same charge
					const double proba = T * delta_time;
					if (psipi.first > psipi.second) { // annihilation was done
						psipj.first  += random_round(proba * psipi.first);
					} else {
						psipj.second += random_round(proba * psipi.second);
					}
				} else {
					const double proba = -T * delta_time;
					if (psipi.first > psipi.second) {
						psipj.second += random_round(proba * psipi.first);
					} else {
						psipj.first  += random_round(proba * psipi.second);
					}
				}
				spawns_list.push_back(make_pair(statej, psipj));
			}
		}

		// (2) Diagonal
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			pair<int,int>& psip = i->second;
			const double H = hamiltonian_ii(i->first);
			const double T = -(H - shift);
			if (T > 0.0) {
				// cloning
				const double proba = T * delta_time;
				if (psip.first > psip.second) {
					psip.first  += random_round(proba * psip.first);
				} else {
					psip.second += random_round(proba * psip.second);
				}
			} else {
				const double proba = -T * delta_time;
				if (psip.first > psip.second) {
					psip.first  = max(0, psip.first  - random_round(proba * psip.first));
				} else {
					psip.second = max(0, psip.second - random_round(proba * psip.second));
				}
			}
		}

		// Insert spawns
		for (const pair<state_type, pair<int,int>>& x : spawns_list) {
			pair<int,int>& psip = walkers[x.first];
			psip.first  += x.second.first;
			psip.second += x.second.second;
		}

		auto t2 = high_resolution_clock::now();
		cout << mpi_rank << ": iteration " << (iter+1) << " execution time " << 1000.0*duration_cast<duration<double>>(t2 - t1).count() << " ms" << endl;


		if (mpi_rank == 0) {
			// recv walkers
			for (int r = 1; r < mpi_size; ++r) {
				cout << mpi_rank << ": wait message from " << r << "..." << flush;
				mpi_recv_walkers(r, walkers);
				cout << " received" << endl;
			}

			// (3) Annihilation
			for (auto i = walkers.begin(); i != walkers.end(); ) {
				pair<int,int>& psip = i->second;
				if (psip.first > psip.second) {
					psip.first -= psip.second;
					psip.second = 0;
					++i;
				} else if (psip.second > psip.first) {
					psip.second -= psip.first;
					psip.first = 0;
					++i;
				} else {
					i = walkers.erase(i);
				}
			}

			size_t count_total_walkers = 0;
			for (auto i = walkers.begin(); i != walkers.end(); ++i) {
				count_total_walkers += i->second.first + i->second.second;
			}

			cout << mpi_rank << ": iteration " << (iter+1) << " " << count_total_walkers << " walkers" << endl;

			mpi_split_send_walkers(mpi_size, walkers);
		} else {
			mpi_send_walkers(0, walkers.begin(), walkers.end());
			mpi_recv_walkers(0, walkers);
		}
	}

	MPI_Finalize();

	return 0;
}
