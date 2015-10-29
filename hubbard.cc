#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <csignal>
#include <chrono>
#include <algorithm>

#include "fciqmc.hh"

//#define USEMPI

#ifdef USEMPI
#include "mpi_data.hh"
#endif

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

const map<state_type, double> hamiltonian(const map<state_type, double>& ket, vector<size_t> ngh[])
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

int main(int argc, char* argv[])
{
	signal(SIGINT, int_handler);
	ofstream ofs("data");
	ofs<<setprecision(15);

	double time_spwdg = 0.0, time_sync = 0.0, time_oth = 0.0;
	double avg_time_spwdg = 0.0, avg_time_sync = 0.0, avg_time_oth = 0.0;
	double menergy = 0.0;

	vector<size_t> ngh[n];
	for (size_t k = 0; k < n; ++k) {
		ngh[k] = {(k-1+n)%n, (k+1)%n};
	}

	state_type start;
	for (size_t k = 0; k < n; ++k) {
		start[k] = 1;
	}
	map<state_type, int> tmp_map;

#ifdef USEMPI
	MPI_Init(&argc, &argv);
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	MPI_Datatype mpi_state_type;
	MPI_Type_contiguous(n, MPI_UINT8_T, &mpi_state_type);
	MPI_Type_commit(&mpi_state_type);

	mpi_data<state_type> walkers(mpi_rank, mpi_size, mpi_state_type);
	tmp_map[start] = 5;
	walkers.sync(tmp_map);
	tmp_map.clear();
#else
	(void)argc;
	(void)argv;
	map<state_type, int> walkers;
	walkers[start] = 100;
#endif


	map<state_type, double> ket;
	ket[start] = 1.0;
	map<state_type, double> hket = hamiltonian(ket, ngh);

#ifdef USEMPI
	if (mpi_rank != 0) {
		ket.clear();
		hket.clear();
	}
	walkers.sync_bis(ket, hket);
#else
#endif

	double energyshift = 20;
	constexpr double dt = 0.01;


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
#if defined(BINO) && defined(USEMPI)
			vector<pair<state_type,double>>
					dst_in,  // list of connected states which belong to the local data
					dst_out; //  "        "        "           "      to other threads
			for (size_t k : ks) {
				for (size_t l : ngh[k]) {
					state_type ste_j(ste_i);
					ste_j[k]--;
					ste_j[l]++;
					if (walkers.islocal(ste_j)) {
						dst_in.push_back(make_pair(ste_j, sqrt(ste_i[k] * ste_j[l])));
					} else {
						dst_out.push_back(make_pair(ste_j, sqrt(ste_i[k] * ste_j[l])));
					}
				}
			}

			if (dst_in.size() > 0) {
				int nrm = floor(0.0 * dst_out.size());
				for (int i = 0; i < nrm; ++i) {
					dst_out.pop_back();
				}
			}

			double advantage = 10.0;
			int c = abs(w_i);

			double p = dt * (advantage * dst_in.size() + dst_out.size()) / advantage;
			for (size_t j = 0; j < dst_in.size(); ++j) {
				if (c == 0) break;
				binomial_distribution<> dist(c, advantage / (advantage * (dst_in.size() - j) + dst_out.size()));
				int ckl = dist(global_random_engine());
				c -= ckl;

				tmp_map[dst_in[j].first] += s_i * binomial_throw(ckl, p * dst_in[j].second);
			}

			p = dt * (advantage * dst_in.size() + dst_out.size());
			for (size_t j = 0; j < dst_out.size(); ++j) {
				if (c == 0) break;
				binomial_distribution<> dist(c, 1.0 / double(dst_out.size() - j));
				int ckl = dist(global_random_engine());
				c -= ckl;

				tmp_map[dst_out[j].first] += s_i * binomial_throw(ckl, p * dst_out[j].second);
			}
#elif defined(BINO)

			int c = abs(w_i);
			double p_done = 0.0;

			for (size_t k : ks) {
				for (size_t l : ngh[k]) {
					if (c == 0) break;

					state_type ste_j(ste_i);
					ste_j[k]--;
					ste_j[l]++;
					double p = dt * sqrt(ste_i[k] * ste_j[l]);

					if (p + p_done > 1.0)
						cerr << "probability greater than 1" << endl;

					binomial_distribution<> dist(c, p / (1.0 - p_done));
					int ckl = dist(global_random_engine());
					c -= ckl;
					p_done += p;

					tmp_map[ste_j] += s_i * ckl;
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

			// H = - \sum_{<i,j>} b^dag_i b_j + U/2 \sum_i n_i (n_i - 1)
			// diagonal part
			{
				double E = 0.0;
				for (size_t k = 0; k < n; ++k) {
					E += ste_i[k] * (ste_i[k] - 1);
				}
				E *= U / 2.0;
				E -= energyshift;
				// if the energy is negative, then we must clone the walkers.

				double p = clamp(-1.0, -E * dt, 1.0); // probability to clone(positive value) / kill(negative value)
				if (w_i < 0)
					i->second -= binomial_throw(-w_i, p);
				else
					i->second += binomial_throw(w_i, p);
			}
		}
		auto t2 = chrono::high_resolution_clock::now();

#ifdef USEMPI
		walkers.sync(tmp_map);
#else

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
#endif

		auto t3 = chrono::high_resolution_clock::now();

		if (iter == 100) state = 1;
		if (iter >= 130) state = 3;

#ifdef USEMPI
		if (state == 1) {
			state = 2;
			cout << "compute hamiltonian" << endl;
			ket.clear();
			for (const pair<state_type,int>& x : walkers) {
				ket.insert(make_pair(x.first, double(x.second)));
			}
			hket = hamiltonian(ket, ngh);
			walkers.sync_bis(ket, hket);
		}

		double energy = walkers.energy();
		menergy = 0.9 * menergy + 0.1 * energy;

		double count_total_walkers = 0.0;
		{
			double count = walkers.count();
			MPI_Reduce(&count, &count_total_walkers, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		}
		if (mpi_rank == 0) {
#endif
			constexpr int A = 3;
			if (iter > 10 && iter%A == 0) {
				static double last_count_total_walkers = count_total_walkers;
				constexpr double damping = 0.10;

				energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
				last_count_total_walkers = count_total_walkers;
			}
#ifdef USEMPI

			if (iter%2 == 0) {
				cout<<"@"<<iter<<": "<<count_total_walkers<<"/"<<walkers.total_size()
					 <<" es="<<energyshift<< " en=" << energy<< endl;
				ofs<<iter<<' '<<count_total_walkers <<' '<<walkers.total_size()<<' '<<energyshift<<' '<<energy<<endl;
			}
		}
		constexpr int B = 10;
		if (iter%B == 0) {
			double l, r, s;
			MPI_Reduce(&walkers.sent_left,  &l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&walkers.sent_right, &r, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&walkers.sent_map,   &s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

			if (mpi_rank == 0) {
				l /= B;
				r /= B;
				s /= B;
				cout << "chrono : spwdg"<<round(time_spwdg)<<" +sync"<<round(time_sync)<<" +other"<<round(time_oth)
						 <<"= "<<round(time_spwdg+time_sync+time_oth)<< "ms " <<round(l)<<"/"<<round(r)<<"/"<<round(s)<<" l/r/s"<< endl;
			}

			walkers.sent_left  = 0.0;
			walkers.sent_right = 0.0;
			walkers.sent_map   = 0.0;
		}
		MPI_Bcast(&energyshift, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#else
			if (state == 1) {
				state = 2;
				cout << "compute hamiltonian" << endl;
				ket.clear();
				for (const pair<state_type,int>& x : walkers) {
					ket.insert(make_pair(x.first, double(x.second)));
				}
				hket = hamiltonian(ket, ngh);
			}

			double energy = scalar_product(hket, walkers) / scalar_product(ket, walkers);
			menergy = 0.75 * menergy + 0.25 * energy;

			if (iter%2 == 0) {
				cout << "@" << iter << ": " << count_total_walkers << "/" << walkers.size()
						 << " es=" << energyshift << " en=" << energy << endl;
				ofs<<iter<<' '<<count_total_walkers <<' '<<walkers.size()<<' '<<energyshift<<' '<<energy<<' '<<endl;
			}
#endif

			auto t4 = chrono::high_resolution_clock::now();

			time_spwdg = 1000.0*chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
			time_sync  = 1000.0*chrono::duration_cast<chrono::duration<double>>(t3 - t2).count();
			time_oth   = 1000.0*chrono::duration_cast<chrono::duration<double>>(t4 - t3).count();

			avg_time_spwdg = (avg_time_spwdg * iter + time_spwdg) / (iter + 1);
			avg_time_sync  = (avg_time_sync  * iter + time_sync ) / (iter + 1);
			avg_time_oth   = (avg_time_oth   * iter + time_oth  ) / (iter + 1);
		}

		cout << "chrono : spwdg"<<round(avg_time_spwdg)<<" +sync"<<round(avg_time_sync)<<" +other"<<round(avg_time_oth)
				 <<"= "<<round(avg_time_spwdg+avg_time_sync+avg_time_oth)<< " (ms in average)" << endl;

		auto t_end = chrono::high_resolution_clock::now();
		double t_total = chrono::duration_cast<chrono::duration<double>>(t_end - t_begin).count();
		cout << "Total time : " << t_total << " seconds. Mean energy : " << setprecision(15)<< menergy << endl;

		ofs.close();

#ifdef USEMPI
		//MPI_Type_free(&mpi_state_type);
		MPI_Finalize();
#endif
		return 0;
	}
