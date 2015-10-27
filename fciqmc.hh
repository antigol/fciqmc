#ifndef FCIQMC_HH
#define FCIQMC_HH

#include <random>
#include <cmath>
#include <map>

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
	int a = std::floor(x);
	if (canonical() < (x-a)) ++a;
	return a;
}

// n must be positive, but p can take any value
int binomial_throw(int n, double p)
{
	int a = std::floor(p);
	std::binomial_distribution<> distribution(n, p-a);
	return n * a + distribution(global_random_engine());
}

template<class T>
inline T clamp(T a, T x, T b)
{
	if (x < a) return a;
	if (x > b) return b;
	return x;
}

template<class K>
double scalar_product(const std::map<K, double>& a,
											const std::map<K, int>& b)
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

#endif // FCIQMC_HH
