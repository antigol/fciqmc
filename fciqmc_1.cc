#include <iostream>
#include <unordered_map>
#include <bitset>
#include <array>
#include <random>
#include <limits>
#include <chrono>
#include <fstream>

using namespace std;

// retourne un générateur de nombres aléatoires global
inline std::default_random_engine& global_random_engine()
{
	// crée un generateur de graines en statique
	static std::random_device rdev;
	// crée un générateur de nombres aléatoires en statique
	static std::default_random_engine eng(rdev());
	return eng;
}

// retourne un nombre entre 0 et 1 (non compris)
inline double canonical()
{
	return std::generate_canonical<double, std::numeric_limits<double>::digits>(global_random_engine());
}

#define N 100
#define alpha (0.0)
typedef array<bool, N> TS;
/*
 * H = \sum_k \vec S_k \cdot \vec S_{k+1} + alpha S^z_k
 *
 * = \sum_k ( 1/2 S^+_k S^-_{k+1} + 1/2 S^-_k S^+_{k+1} + S^z_k S^z_{k+1} )
 *
 * = \sum_k ( 1/2 "swap k and k+1 if different otherwise 0" + "s_k == s_{k+1} ? + : -" 1/4 "identity" )
 *                    only for non-diagonal terms           |      only for diagonal terms
 *                    only if they differ by only 1 swap
 **/

double hamiltonian_ii(const TS& i)
{
	double E = 0.0;
	for (int k0 = 0; k0 < N; ++k0) {
		int k1 = (k0+1)%N;
		if (i[k0]) E += 0.5 * alpha;
		else E -= 0.5 * alpha;

		if (i[k0] == i[k1]) E += 0.25;
		else E -= 0.25;
	}
	return E;
}

double hamiltonian_ij(const TS& i, const TS& j)
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

int main()
{
	using namespace std::chrono;

	double delta_time = 0.2 / N;
	double shift = 20.5;

	auto hash_function = [](const TS& x) {
		size_t ret = 0;
		for (size_t i = 0; i < x.size(); ++i) {
			ret *= 2;
			ret += x[i];
		}
		return ret;
	};

	unordered_map<TS, pair<int, int>, decltype(hash_function)> psips(10, hash_function);

	TS psit;
	for (int n = 0; n <= N; ++n) {
		for (int k = 0; k < N; ++k) psit[k] = (k < n);
		psips[psit].first = 5;
	}

	int last_total_walkers = 1;
	int A = 25;

	high_resolution_clock::time_point t1, t2;
	ofstream statfile("stats.txt");

	for (int iter = 1; iter <= 10000; ++iter) {
		cout << "iteration " << iter << endl;
		/*for (auto i = psips.begin(); i != psips.end(); ++i) {
			cout << i->first.to_string() << " -> " << i->second.first << "," << i->second.second << endl;
		}*/
		//cout << "Spawning..." << endl;

		// Spawning
		t1 = high_resolution_clock::now();
		vector<pair<TS, pair<int,int>>> spawns;
		for (auto i = psips.begin(); i != psips.end(); ++i) {

			const TS& statei = i->first;
			for (int k0 = 0; k0 < N; ++k0) {
				int k1 = (k0+1)%N;
				// connected sites
				TS statej = statei;
				statej[k0] = !statej[k0];
				statej[k1] = !statej[k1];

				// compute H(i,j)
				double H = (statei[k0] == statei[k1]) ? 0.0 : 0.5;

				pair<int,int> psipj = make_pair(0, 0);

				double T = -H;
				if (T == 0.0) continue;
				double proba = fabs(T)*delta_time;
				//cout << "proba=" << proba << endl;

				if (T > 0.0) {
					if (proba >= 1.0) {
						double sure = floor(proba);
						psipj.first += sure * i->second.first;
						psipj.second += sure * i->second.second;
						proba -= sure;
					}

					for (int positive = 0; positive < i->second.first; ++positive) {
						if (canonical() < proba) psipj.first++; // new charge (+)
					}
					for (int negative = 0; negative < i->second.first; ++negative) {
						if (canonical() < proba) psipj.second++; // new charge (-)
					}
				} else {
					if (proba >= 1.0) {
						double sure = floor(proba);
						psipj.second += sure * i->second.first;
						psipj.first += sure * i->second.second;
						proba -= sure;
					}

					for (int positive = 0; positive < i->second.first; ++positive) {
						if (canonical() < proba) psipj.second++; // new charge (-)
					}
					for (int negative = 0; negative < i->second.first; ++negative) {
						if (canonical() < proba) psipj.first++; // new charge (+)
					}
				}

				if (psipj.first == 0 && psipj.second == 0) continue;
				//cout << "push psipj" << endl;
				spawns.push_back(make_pair(statej, psipj));
			}
		}
		t2 = high_resolution_clock::now();
		cout << "spawning " << duration_cast<duration<double>>(t2 - t1).count() << endl;

		// Diagonal death/cloning
		t1 = high_resolution_clock::now();
		int died_walkers = 0;
		for (auto i = psips.begin(); i != psips.end(); ++i) {
			pair<int,int>& psip = i->second;

			// compute H(i,i)
			double H = hamiltonian_ii(i->first);

			double T = -(H - shift);
			if (T > 0.0) {
				double proba = T * delta_time;
				for (int positive = psip.first; positive > 0; --positive) {
					if (canonical() < proba) psip.first++; // clone
				}
				for (int negative = psip.second; negative > 0; --negative) {
					if (canonical() < proba) psip.second++; // clone
				}
			} else {
				double proba = -T * delta_time;
				for (int positive = psip.first; positive > 0; --positive) {
					if (canonical() < proba) {
						psip.first--; // die
						died_walkers++;
					}
				}
				for (int negative = psip.second; negative > 0; --negative) {
					if (canonical() < proba) {
						psip.second--; // die
						died_walkers++;
					}
				}
			}
		}
		t2 = high_resolution_clock::now();
		cout << "diagonal " << duration_cast<duration<double>>(t2 - t1).count() << endl;

		//cout << spawns.size() << " inserts" << endl;
		t1 = high_resolution_clock::now();
		int created_walkers = 0;
		for (const pair<TS, pair<int,int>>& x : spawns) {
			pair<int,int>& psip = psips[x.first];
			created_walkers += x.second.first + x.second.second;
			psip.first += x.second.first;
			psip.second += x.second.second;
		}
		t2 = high_resolution_clock::now();
		cout << "insetion " << duration_cast<duration<double>>(t2 - t1).count() << endl;

		/*
		for (auto i = psips.begin(); i != psips.end(); ++i) {
			cout << i->first.to_string() << " -> " << i->second.first << "," << i->second.second << endl;
		}*/
		//cout << "Diagonal..." << endl;





		//cout << "Annihilation..." << endl;

		// Annihilation
		t1 = high_resolution_clock::now();
		int total_walkers = 0;
		int anihilated_walkers = 0;
		int state_max = 0;
		TS state;
		for (auto i = psips.begin(); i != psips.end(); ) {
			pair<int,int>& psip = i->second;

			if (psip.first > psip.second) {
				psip.first -= psip.second;
				anihilated_walkers += 2*psip.second;
				psip.second = 0;
				total_walkers += psip.first;
				if (psip.first > state_max) {
					state_max = psip.first;
					state = i->first;
				}
				++i;
			} else if (psip.second > psip.first) {
				psip.second -= psip.first;
				anihilated_walkers += 2*psip.first;
				psip.first = 0;
				total_walkers += psip.second;
				if (psip.second > state_max) {
					state_max = psip.second;
					state = i->first;
				}
				++i;
			} else {
				i = psips.erase(i);
			}
		}
		t2 = high_resolution_clock::now();
		cout << "annihila " << duration_cast<duration<double>>(t2 - t1).count() << endl;

		// Modify the shift
		if (iter % A == 0 && iter > 2000) {
			t1 = high_resolution_clock::now();
			double E_up = 0.0;
			double E_down = 0.0;
			for (auto i = psips.begin(); i != psips.end(); ++i) {
				double pi = i->second.first > i->second.second ? i->second.first : -i->second.second;
				for (auto j = psips.begin(); j != psips.end(); ++j) {
					double pj = j->second.first > j->second.second ? j->second.first : -j->second.second;
					E_up += pi * hamiltonian_ij(i->first, j->first) * pj;
				}
				E_down += pi * pi;
			}
			double E = E_up / E_down;
			t2 = high_resolution_clock::now();
			cout << "energy   " << duration_cast<duration<double>>(t2 - t1).count() << "\t is " << E << endl;

			shift -= 0.05 / A / delta_time * log(total_walkers / (double)last_total_walkers);
			last_total_walkers = total_walkers;
		}

		if (iter == 1) last_total_walkers = total_walkers;

		statfile << total_walkers << '\t' << created_walkers << '\t' << died_walkers << '\t' << anihilated_walkers << endl;

		cout << "amount of psips is " << total_walkers << " shift is " << shift << endl;
		for (int k = 0; k < N; ++k) cout << state[k];
		cout << endl;
	}

	statfile.close();

	return 0;
}

