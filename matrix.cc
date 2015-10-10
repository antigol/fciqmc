#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <csignal>
#include <chrono>

#include "fciqmc.hh"

using namespace std;

int main()
{
	constexpr int n = 3;
	vector<double> matrix = {
		//		10.00000,   -2.82843,   0.00000,
		//		-2.82843,    0.00000,  -2.82843,
		//		0.00000,    -2.82843,   10.00000
		2, 3, 0,
		3, -1, 3,
		0, 3, -2
	};

	int walkers[n];
	for (int i = 0; i < n; ++i)
		walkers[i] = 0;

	walkers[0] = 1;

	double energyshift = 10.0;
	constexpr double dt = 0.006;

	int changes[n];

	for (size_t iter = 0; iter < 5000; ++iter) {
		for (int i = 0; i < n; ++i)
			changes[i] = 0;

		for (int i = 0; i < n; ++i)
		{
			vector<int> sac;
			for (int j = 0; j < n; ++j) {
				if (j != i) {
					if (matrix[i*n+j] != 0.0)
						sac.push_back(j);
				}
			}
			uniform_int_distribution<> d(0, sac.size()-1);
			for (int c = abs(walkers[i]); c > 0; --c) {
				int j = sac[d(global_random_engine())];

				//for (int j : sac) {
					double K = matrix[i*n+j];
					double p = abs(K) * dt * sac.size();
					if (canonical() < p) {
						if (K < 0.0) {
							if (walkers[i] > 0) changes[j]++;
							else changes[j]--;
						} else {
							if (walkers[i] > 0) changes[j]--;
							else changes[j]++;
						}
					}
				//}
			}
		}

		for (int i = 0; i < n; ++i) {
			walkers[i] += binomial_throw(walkers[i], clamp(-1.0, -(matrix[i*n+i] - energyshift) * dt, 1.0));
		}

		for (int i = 0; i < n; ++i)
			walkers[i] += changes[i];

		double count_total_walkers = 0.0;
		for (int i = 0; i < n; ++i)
			count_total_walkers += abs(walkers[i]);

		constexpr int A = 5;
		if (iter > 20 && iter%A == 0) {
			static double last_count_total_walkers = count_total_walkers;
			constexpr double damping = 0.05;

			energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
			last_count_total_walkers = count_total_walkers;
		}
		cout << "@" << iter << ": " << count_total_walkers << " es=" << energyshift << endl;
	}

	return 0;
}
