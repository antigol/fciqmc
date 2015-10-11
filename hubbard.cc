#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <csignal>
#include <chrono>

#include "fciqmc.hh"

using namespace std;

// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
constexpr double U = 10.0;
constexpr size_t n = 8;
typedef array<uint8_t, n> state_type;


bool operator<(const state_type& lhs, const state_type& rhs)
{
	for (size_t k = 0; k < n; ++k) {
		if (lhs[k] < rhs[k]) return true;
		if (lhs[k] > rhs[k]) return false;
	}
	return false;
}

double scalar_product(const map<state_type, double>& a,
					  const map<state_type, int>& b)
{
	double r = 0.0;
	auto i = a.begin();
	auto j = b.begin();

	while (i != a.end() && j != b.end()) {
		if (i->first < j->first) ++i;
		else if (j->first < i->first) ++j;
		else {
			r += i->second * (double)j->second;
			++i;
			++j;
		}
	}

	return r;
}

double scalar_product(const map<state_type, int>& a,
					  const map<state_type, int>& b)
{
	double r = 0.0;
	auto i = a.begin();
	auto j = b.begin();

	while (i != a.end() && j != b.end()) {
		if (i->first < j->first) ++i;
		else if (j->first < i->first) ++j;
		else {
			r += (double)i->second * (double)j->second;
			++i;
			++j;
		}
	}

	return r;
}

const map<state_type, double> hamiltonian(const map<state_type, int>& ket, vector<size_t> ngh[])
{
	map<state_type, double> hket;
	for (auto i = ket.begin(); i != ket.end(); ++i) {
		const state_type& ste_i = i->first;

		double repulsion = 0.0;
		for (size_t k = 0; k < n; ++k) repulsion += ste_i[k] * (ste_i[k] - 1);
		hket[ste_i] += U/2.0 * repulsion * i->second;

		for (size_t k = 0; k < n; ++k) {
			if (ste_i[k] == 0) continue;
			for (size_t l : ngh[k]) {
				state_type ste_j(ste_i);
				ste_j[k]--;
				ste_j[l]++;

				hket[ste_j] += -sqrt(ste_i[k] * ste_j[l]) * i->second;
			}
		}
	}
	return hket;
}

sig_atomic_t state = 0;
void int_handler(int)
{
	state++;
}

int main()
{
	signal(SIGINT, int_handler);
	ofstream ofs("data");
	ofs<<setprecision(15);

	double time1 = 0.0, time2 = 0.0, time3 = 0.0, time4 = 0.0;

	vector<size_t> ngh[n];
	for (size_t k = 0; k < n; ++k) {
		ngh[k] = {(k-1+n)%n, (k+1)%n};
	}

	map<state_type, int> walkers;

	state_type start;
	for (size_t k = 0; k < n; ++k) {
		start[k] = 1;
	}
	walkers[start] = 100;

	map<state_type, int> ket;
	ket[start] = 1;

	map<state_type, double> hket = hamiltonian(ket, ngh);

	double energyshift = 30;
	constexpr double dt = 0.01;


	map<state_type, int> tmp_map;
	vector<size_t> ks;
	ks.reserve(n);

	auto t_begin = chrono::high_resolution_clock::now();

	for (size_t iter = 0; state < 3; ++iter) {

		auto t1 = chrono::high_resolution_clock::now();
		tmp_map.clear();


		// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& ste_i = i->first;
			int w_i = i->second;
			int s_i = (w_i < 0.0) ? -1 : 1;

			ks.clear();
			for (size_t k = 0; k < n; ++k) {
				if (ste_i[k] != 0) ks.push_back(k);
			}


#define BINO
#ifdef BINO
			int c = abs(w_i);
			for (size_t ki = 0; ki < ks.size(); ++ki) {
				if (c == 0) break;

				size_t k = ks[ki];

				int ci = c;
				if (ki < ks.size() - 1) {
					binomial_distribution<> dist_k(c, 1.0 / (double)(ks.size() - ki));
					ci = dist_k(global_random_engine());
					c -= ci;
				}

				for (size_t li = 0; li < ngh[k].size(); ++li) {
					if (ci == 0) break;

					size_t l = ngh[k][li];

					int di = ci;
					if (li < ngh[k].size() - 1) {
						binomial_distribution<> dist_l(ci, 1.0 / (double)(ngh[k].size() - li));
						di = dist_l(global_random_engine());
						ci -= di;
					}

					// di walkers sont tombés sur la connexion orientée <kl> avec probabilité de 1/ks.size * 1/ngh[k].size
					state_type ste_j(ste_i);
					ste_j[k]--;
					ste_j[l]++;

					// <i|H|j>  < 0 !
					// p = - <i|H|j> * dt / P(<ij>)
					double p = dt * sqrt(ste_i[k] * ste_j[l]) * ks.size() * ngh[k].size();
					tmp_map[ste_j] += s_i * binomial_throw(di, p);
				}
			}

#else
			uniform_int_distribution<> dist_ks(0, ks.size()-1);

			for (int c = abs(w_i); c > 0; --c) {
				size_t k = ks[dist_ks(global_random_engine())];

				uniform_int_distribution<> dist_ls(0, ngh[k].size()-1);
				size_t l = ngh[k][dist_ls(global_random_engine())];

				state_type ste_j(ste_i);
				ste_j[k]--;
				ste_j[l]++;

				double K = -sqrt(ste_i[k] * (ste_i[l]+1.0)) * ks.size() * ngh[k].size();
				if (canonical() < abs(K * dt)) {
					tmp_map[ste_j] += s_i; // because H si always negative
				}
			}
#endif
		}
		auto t2 = chrono::high_resolution_clock::now();



		// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
		for (auto i = walkers.begin(); i != walkers.end(); ++i) {
			const state_type& ste_i = i->first;
			int w_i = i->second;

			// diag
			double E = 0.0;
			for (size_t k = 0; k < n; ++k) {
				E += ste_i[k] * (ste_i[k] - 1);
			}
			E *= U / 2.0;
			E -= energyshift;
			// if the energy is negative, then we must clone the walkers.

			double p = clamp(-1.0, -E * dt, 1.0); // probability to clone(positive value) / kill(negative value)
			if (w_i < 0) {
				w_i = -w_i;
				p = -p;
			}
			i->second += binomial_throw(w_i, p);
		}

		auto t3 = chrono::high_resolution_clock::now();
#define ITER
#ifdef ITER
		{
			auto i = walkers.begin();
			auto j = tmp_map.begin();
			while (i != walkers.end() && j != tmp_map.end()) {
				if (i->first < j->first) {
					++i;
				} else if (j->first < i->first) {
					walkers.insert(i, *j);
					++j;
				} else {
					i->second += j->second;
					++i;
					++j;
				}
			}
			walkers.insert(j, tmp_map.end());
		}
#else
		for (auto i = tmp_map.begin(); i != tmp_map.end(); ++i) {
			walkers[i->first] += i->second;
		}
#endif
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

		constexpr int A = 3;
		if (iter > 10 && iter%A == 0) {
			static double last_count_total_walkers = count_total_walkers;
			constexpr double damping = 0.10;

			energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
			last_count_total_walkers = count_total_walkers;
		}

		auto t5 = chrono::high_resolution_clock::now();

		if (state == 1) {
			state = 2;
			cout << "compute hamiltonian" << endl;
			ket = walkers;
			hket = hamiltonian(ket, ngh);
		}

		if (iter%2 == 0) {
			double energy = scalar_product(hket, walkers) / scalar_product(ket, walkers);

			cout << "@" << iter << ": " << count_total_walkers << "/" << walkers.size() << " es=" << energyshift << " en=" << energy << endl;
			ofs<<iter<<' '<<count_total_walkers <<' '<<walkers.size()<<' '<<energyshift<<' '<<energy<<' '<<endl;
		}



		time1 = (time1 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t2 - t1).count()) / (iter + 1);
		time2 = (time2 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t3 - t2).count()) / (iter + 1);
		time3 = (time3 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t4 - t3).count()) / (iter + 1);
		time4 = (time4 * iter + 1000.0*chrono::duration_cast<chrono::duration<double>>(t5 - t4).count()) / (iter + 1);
	}

	cout << "chrono : spw"<<round(time1)<<" +dg"<<round(time2)<<" +cgs"<<round(time3)<<" +ann"<<round(time4)
		 <<"= "<<round(time1+time2+time3+time4)<< " (ms in average)" << endl;

	auto t_end = chrono::high_resolution_clock::now();
	double t_total = chrono::duration_cast<chrono::duration<double>>(t_end - t_begin).count();
	cout << "Total time : " << t_total << " seconds." << endl;

	ofs.close();
	return 0;
}
