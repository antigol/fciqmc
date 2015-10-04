#include <vector>
#include <array>
//#include <unordered_map>
#include <map>
#include <fstream>
#include <iostream>
#include <csignal>
#include <chrono>

#include "fciqmc.hh"

using namespace std;

// H = -t \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1) - mu \sum_i n_i
constexpr double t = 1.0;
constexpr double U = 0.8;
constexpr double mu = 0.6;
constexpr size_t n = 10*10;
typedef array<uint8_t, n> state_type;

//struct KeyHasher {
//	size_t operator()(const state_type& state) const
//	{
//		constexpr size_t n64 = 64 < n ? 64 : n;
//		size_t r = 0;
//		for (size_t k = 0; k < n64; ++k) {
//			r ^= (state[k] << k);
//		}
//		return r;
//	}
//};
struct KeyCompare {
	bool operator()(const state_type& lhs, const state_type& rhs) const
	{
		for (size_t k = 0; k < n; ++k) {
			if (lhs[k] < rhs[k]) return true;
			if (lhs[k] > rhs[k]) return false;
		}
		return false;
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

	double time1 = 0.0, time2 = 0.0, time3 = 0.0, time4 = 0.0;

	vector<size_t> ngh[n];
	for (size_t k = 0; k < n; ++k) {
		//nhg[k] = {(k-1+n)%n, (k+1)%n};
		size_t w = 10;
		size_t h = 10;
		size_t x = k % w;
		size_t y = k / w;
		ngh[k] = {
			w*y           + ((x+1)%w),
			w*y           + ((x-1+w)%w),
			w*((y+1)%h)   + x,
			w*((y-1+h)%h) + x
		};
	}

	//unordered_map<state_type, int, KeyHasher> walkers;
	map<state_type, int, KeyCompare> walkers;

	state_type start;
	for (size_t k = 0; k < n; ++k) {
		start[k] = 1;
	}
	walkers[start] = 1;

	double energyshift = -25.0;
	constexpr double dt = 0.005;

	uniform_int_distribution<> dist_n(0, n-1);
	vector<pair<state_type, int>> changes;

	for (size_t iter = 0; !stop; ++iter) {
		auto t1 = chrono::high_resolution_clock::now();
		changes.clear();

		// H = -t \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1) - mu \sum_i n_i
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& ste_i = i->first;
			int w_i = i->second;
			int s_i = signbit(w_i) ? -1 : 1;

			binomial_distribution<> d(abs(w_i), t * dt);
			int c_i = d(global_random_engine());

			// spawn
			while (c_i) {
				size_t k = dist_n(global_random_engine());
				if (ste_i[k] > 0) {
					state_type ste_j(ste_i);
					ste_j[k]--;

					uniform_int_distribution<> d(0, ngh[k].size()-1);
					size_t l = ngh[k][d(global_random_engine())];
					ste_j[l]++;
					// E = -t
					// qj = -sign(E) qi => qj = qi
					changes.emplace_back(ste_j, s_i);
					c_i--;
				}
			}
		}

		auto t2 = chrono::high_resolution_clock::now();

		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& ste_i = i->first;

			// diag
			double E = 0.0;
			for (size_t k = 0; k < n; ++k) {
				E += ste_i[k] * (U*(ste_i[k] - 1)/2.0 - mu);
			}
			E -= energyshift;
			// if E < 0 => clone
			i->second += binomial_throw(i->second, clamp(-1.0, -E * dt, 1.0));
		}
		auto t3 = chrono::high_resolution_clock::now();


		for (auto& x : changes) walkers[x.first] += x.second;

		auto t4 = chrono::high_resolution_clock::now();

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
		if (iter > 20 && iter%A == 0) {
			static double last_count_total_walkers = count_total_walkers;
			constexpr double damping = 0.05;

			energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
			last_count_total_walkers = count_total_walkers;
		}
		if (iter % A == 0) {
			cout << "@" << iter << ": " << count_total_walkers << "/" << walkers.size() << " es=" << energyshift << endl;
			ofs << iter <<' '<< energyshift <<' '<< count_total_walkers <<' '<< walkers.size() << endl;
		}
		auto t5 = chrono::high_resolution_clock::now();


		time1 = (time1 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t2 - t1).count()) / (iter + 1);
		time2 = (time2 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t3 - t2).count()) / (iter + 1);
		time3 = (time3 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t4 - t3).count()) / (iter + 1);
		time4 = (time4 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t5 - t4).count()) / (iter + 1);
	}

	cout << "chrono : spw"<<round(time1)<<" +dg"<<round(time2)<<" +cgs"<<round(time3)<<" +ann"<<round(time4)
			 <<"= "<<round(time1+time2+time3+time4)<< " (ms in average)" << endl;
	ofs.close();
	return 0;
}
