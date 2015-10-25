#include <iostream>
#include <array>
#include <vector>
#include <set>

using namespace std;

/*
 graph graphname {
		 a -- b -- c;
		 b -- d;
 }
 */

constexpr size_t n = 3;
typedef array<uint8_t, n> state_type;

bool operator<(const state_type& lhs, const state_type& rhs)
{
	for (size_t k = 0; k < n; ++k) {
		if (lhs[k] < rhs[k]) return true;
		if (lhs[k] > rhs[k]) return false;
	}
	return false;
}

ostream& operator <<(ostream& out, const state_type& x)
{
	for (size_t k = 0; k < n; ++k) {
		out << "x";
		out << int(x[k]);
	}
	return out;
}

int main()
{
	state_type start;
	for (size_t k = 0; k < n; ++k) start[k] = 1;

	vector<size_t> ngh[n];
	for (size_t k = 0; k < n; ++k) {
		ngh[k] = {(k-1+n)%n, (k+1)%n};
	}

	set<state_type> open, close;
	open.insert(start);

	cout << "graph {" << endl;

	while (!open.empty()) {
		state_type x = *(open.begin());
		open.erase(open.begin());


		for (size_t k = 0; k < n; ++k) {
			if (x[k] > 0) {
				for (size_t l : ngh[k]) {
					state_type y = x;
					y[k]--;
					y[l]++;

					if (close.count(y) == 0) {
						cout << x << " -- " << y << ";" << endl;
						open.insert(y);
					}
				}
			}
		}

		close.insert(x);
	}

	cout << "}" << endl;

	return 0;
}
