#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <csignal>
#include <chrono>

#include "fciqmc.hh"

using namespace std;

// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
constexpr double U = 10.0;
constexpr size_t n = 8;
typedef array<uint8_t, n> state_type;

struct KeyCompare {
	bool operator()(const state_type& lhs, const state_type& rhs) const
	{
		for (size_t k = 0; k < n; ++k) {
			if (lhs[k] < rhs[k]) return true;
			if (lhs[k] > rhs[k]) return false;
		}
		return false;
	}
} keyCompare;

double scalar_product(const map<state_type, double, KeyCompare>& a,
											const map<state_type, int, KeyCompare>& b)
{
	double r = 0.0;
	auto i = a.begin();
	auto j = b.begin();

	while (i != a.end() && j != b.end()) {
		if (keyCompare(i->first, j->first)) ++i;
		else if (keyCompare(j->first, i->first)) ++j;
		else {
			r += i->second * j->second;
			++i;
			++j;
		}
	}

	return r;
}

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

	constexpr int ngh_size = 2;
	size_t ngh[n][ngh_size];
	for (size_t k = 0; k < n; ++k) {
		ngh[k][0] = (k-1+n)%n;
		ngh[k][1] = (k+1)%n;
	}

	map<state_type, int, KeyCompare> walkers;

	state_type start;
	for (size_t k = 0; k < n; ++k) {
		start[k] = 1;
	}
	walkers[start] = 1;

	map<state_type, double, KeyCompare> try_state;
	try_state[start] = 1;
	map<state_type, double, KeyCompare> try_state_h;
	for (auto i = try_state.begin(); i != try_state.end(); ++i) {
		const state_type& ste_i = i->first;
		double diag = 0.0;
		for (size_t k = 0; k < n; ++k) diag += ste_i[k] * (ste_i[k] - 1);
		try_state_h[ste_i] += U/2.0 * diag;
		for (size_t k = 0; k < n; ++k) {
			if (ste_i[k] == 0) continue;
			for (size_t l : ngh[k]) {
				state_type ste_j = ste_i;
				ste_j[k]--;
				ste_j[l]++;

				try_state_h[ste_j] += -sqrt(ste_i[k] * (ste_i[l]+1.0));
			}
		}
	}

	double energyshift = 5;
	constexpr double dt = 0.02;

	uniform_int_distribution<> dist_n(0, n-1);
	uniform_int_distribution<> dist_ngh(0, ngh_size-1);

	map<state_type, int> tmp_map;

	for (size_t iter = 0; !stop; ++iter) {
		auto t1 = chrono::high_resolution_clock::now();
		tmp_map.clear();


		// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& ste_i = i->first;
			int w_i = i->second;
			int s_i = (w_i < 0.0) ? -1 : 1;

			for (int c = abs(w_i); c > 0; --c) {
				size_t k;
				size_t l;
				do {
					k = dist_n(global_random_engine());
					l = ngh[k][dist_ngh(global_random_engine())];
				} while (ste_i[k] == 0);


				state_type ste_j(ste_i);
				ste_j[k]--;
				ste_j[l]++;

				double H = -sqrt(ste_i[k] * (ste_i[l]+1.0));
				if (canonical() < abs(H * dt)) {
					tmp_map[ste_j] += s_i; // because H si always negative
				}
			}
		}


		auto t2 = chrono::high_resolution_clock::now();

		// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& ste_i = i->first;
			int w_i = i->second;
//			int s_i = (w_i < 0.0) ? -1 : 1;

			// diag
			double E = 0.0;
			for (size_t k = 0; k < n; ++k) {
				E += ste_i[k] * (ste_i[k] - 1);
			}
			E *= U / 2.0;
			E -= energyshift;
			// if E < 0 => clone
			i->second += binomial_throw(w_i, clamp(-1.0, -E * dt, 1.0));
			/*for (int c = abs(w_i); c > 0; --c) {
				if (canonical() < abs(E * dt)) {
					if (E < 0) i->second += s_i;
					else if (i->second != 0) i->second -= s_i;
				}
			}*/
		}
		auto t3 = chrono::high_resolution_clock::now();

		for (auto i = tmp_map.begin(); i != tmp_map.end(); ++i) {
			walkers[i->first] += i->second;
		}

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
			constexpr double damping = 0.07;

			energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
			last_count_total_walkers = count_total_walkers;
		}
		if (iter % 1 == 0) {
			double energy = scalar_product(try_state_h, walkers) / scalar_product(try_state, walkers);

			cout << "@" << iter << ": " << count_total_walkers << "/" << walkers.size() << " es=" << energyshift << " en=" << energy << endl;
			ofs<<iter<<count_total_walkers <<' '<<walkers.size()<<' '<<energyshift<<' '<<energy<<' '<<endl;
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
