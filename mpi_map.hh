#ifndef MPI_MAP_HH
#define MPI_MAP_HH

#include <map>
#include <vector>
#include <string>
#include <mpi/mpi.h>

template<class K, class T>
class mpi_map
{
public:
	mpi_map(int rank, int size) :
		mp_rank(rank), mp_size(size)
	{
		m_sizes.resize(mp_size, 0);
		m_begins.resize(mp_size);
	}

	void add(const K& k, const T& v)
	{
		// find where with m_begins
		int dst = 0;
		for (int i = 1; i < mp_size; ++i) {
			if (m_sizes[i] == 0 || m_begins[i] < k || m_begins[i] == k) {
				dst = i;
			} else {
				break;
			}
		}

		// send message
		std::string message = serialize(k, v);
		MPI_Send((void*)message.data(), message.size(), MPI_BYTE, dst, 0, MPI_COMM_WORLD);
	}

	void mpi_sumup(const std::map<K,T>& map)
	{
		// add + synchronize
	}

	void synchronize()
	{
		// recv messages and add

		// send size
		// recv sizes

		// send two neighbour to equilibrate
		// recv from neighbour

		// send begin + size
		// recv begins + sizes
	}

	std::map<K,T>::iterator begin() { return m_map.begin(); }
	std::map<K,T>::iterator end() { return m_map.end(); }

private:
	int mp_rank;
	int mp_size;

	std::vector<size_t> m_sizes;
	std::vector<K> m_begins;
	std::map<K,T> m_map;
};

#endif // MPI_MAP_HH
