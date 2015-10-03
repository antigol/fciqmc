#include <vector>
#include <array>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <csignal>
#include <chrono>

#include "fciqmc.hh"

using namespace std;

// H = -t \sum_{<i,j>,s} s^dag_i s_j + s^dag_j s_i + U \sum_i nup_i ndown_j
constexpr double t = 1.0;
constexpr double U = 1.0;
constexpr size_t n = 16;
typedef array<uint8_t, n> state_type;

struct KeyHasher {
	size_t operator()(const state_type& state) const
	{
		size_t r = 0;
		for (size_t k = 0; k < n; ++k) {
			r = (r << 4);
			r ^= state[k];
		}
		return r;
	}
};

sig_atomic_t stop = 0;
void int_handler(int)
{
	stop = 1;
}

int main()
{
	signal(SIGINT, int_handler);
	ofstream ofs("data");

	double time1, time2;

	unordered_map<state_type, int, KeyHasher> walkers;

	state_type start;
	for (size_t k = 0; k < n; ++k) {
		start[k] = 1;
	}
	walkers[start] = 1;

	double energyshift = 20.0;
	constexpr double dt = 0.01;

	uniform_int_distribution<> dis(0, n-1);
	vector<pair<state_type, int>> changes;
	changes.reserve(128);

	for (size_t iter = 0; !stop; ++iter) {
		auto t1 = chrono::high_resolution_clock::now();
		changes.clear();

		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& st_i = i->first;
			int w_i = i->second;

			// spawn
			for (int s = abs(w_i); s > 0; --s) {
				size_t k0 = dis(global_random_engine());
				size_t k1 = (k0 + 2) % n;
				if (st_i[k0] > 0) {
					state_type st_j(st_i);
					st_j[k0]--;
					st_j[k1]++;
					// E = -t
					if (canonical() < t * dt) {
						// qj = -sign(E) qi => qj = qi
						changes.emplace_back(st_j, w_i>0?1:-1);
					}
				}
			}

			// diag
			double E = 0.0;
			for (size_t k = 0; k < n; k += 2) {
				E += U * st_i[k] * st_i[k+1];
			}
			E -= energyshift;
			// if E < 0 => clone
			changes.emplace_back(st_i, binomial_throw(w_i, clamp(-1.0, -E * dt, 1.0)));
		}
		auto t2 = chrono::high_resolution_clock::now();

		for (auto& x : changes) walkers[x.first] += x.second;

		// annihilation
		double count_total_walkers = 0.0;
		for (auto i = walkers.begin(); i != walkers.end();) {
			count_total_walkers += abs(i->second);
			if (i->second == 0) {
				i = walkers.erase(i);
			} else {
				++i;
			}
		}

		constexpr int A = 5;
		if (iter > 50 && iter%A == 0) {
			static double last_count_total_walkers = count_total_walkers;
			constexpr double damping = 0.2;

			energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
			last_count_total_walkers = count_total_walkers;

			cout << "@" << iter << ": " << count_total_walkers << "/" << walkers.size() << " es=" << energyshift << endl;

			ofs << iter <<' '<< energyshift <<' '<< count_total_walkers <<' '<< walkers.size() << endl;
		}
		auto t3 = chrono::high_resolution_clock::now();

		time1 = (time1 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t2 - t1).count()) / (iter + 1);
		time2 = (time2 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t3 - t2).count()) / (iter + 1);
	}

	cout << "chrono : " << time1 <<' '<< time2 << endl;
	ofs.close();
	return 0;
}
