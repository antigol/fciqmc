#include <vector>
#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <csignal>
#include <chrono>

#include "fciqmc.hh"

using namespace std;

// H = (a, b; b, c)
// (a+c)/2  \pm sqrt((a-c)²/4 + b²)
constexpr double a = 3;
constexpr double b = 2;
constexpr double c = 4;

int main()
{
	int walkers[2];

	walkers[0] = 1;
	walkers[1] = 0;

	double energyshift = 10.0;
	constexpr double dt = 0.01;

	int changes[2] = {0, 0};

	for (size_t iter = 0; iter < 2000; ++iter) {
		changes[0] = changes[1] = 0;

		// H = (a, b, b, c)
		{
			binomial_distribution<> d(abs(walkers[0]), abs(b) * dt);
			changes[1] += (b * walkers[0] < 0.0 ? 1.0 : -1.0) * d(global_random_engine());
		}
		{
			binomial_distribution<> d(abs(walkers[1]), abs(b) * dt);
			changes[0] += (b * walkers[1] < 0.0 ? 1.0 : -1.0) * d(global_random_engine());
		}

		walkers[0] += binomial_throw(walkers[0], clamp(-1.0, -(a - energyshift) * dt, 1.0));
		walkers[1] += binomial_throw(walkers[1], clamp(-1.0, -(c - energyshift) * dt, 1.0));

		walkers[0] += changes[0];
		walkers[1] += changes[1];

		double count_total_walkers = abs(walkers[0]) + abs(walkers[1]);
		constexpr int A = 5;
		if (iter > 20 && iter%A == 0) {
			static double last_count_total_walkers = count_total_walkers;
			constexpr double damping = 0.05;

			energyshift -= damping / (A * dt) * log(count_total_walkers / last_count_total_walkers);
			last_count_total_walkers = count_total_walkers;
		}
		cout << "@" << iter << ": " << count_total_walkers << " es=" << energyshift << endl;
	}

	double mu = sqrt((a-c)*(a-c)/4.0 + b*b);
	cout << walkers[1] / (double)walkers[0] << " =?= " << -(mu+(a-c)/2.0)/b << endl;

	// (a+c)/2  \pm sqrt((a-c)²/4 + b²)
	cout << ((a+c)/2.0 + mu) << " or " << ((a+c)/2.0 - mu) << endl;

	return 0;
}
