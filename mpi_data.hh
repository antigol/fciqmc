#ifndef MPI_DATA_HH
#define MPI_DATA_HH

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <mpi/mpi.h>

template<class K>
class mpi_data
{
public:
	mpi_data(int rank, int size, MPI_Datatype type) :
		mp_rank(rank), mp_size(size), mp_type(type)
	{
		m_sizes.resize(mp_size, 0);
		m_begins.resize(mp_size);
	}

	void sync(const std::map<K,int>& map)
	{
		enum {
			tag_size = 0,
			tag_key,
			tag_value
		};
		int value;
		K key;

		{ // SEND MAP
			int dst = 0;
			for (auto i = map.begin(); i != map.end(); ++i) {
				if (i->second == 0) continue;

				// find rank to send data to
				while (dst+1 < mp_size && m_sizes[dst+1]!=0 && !(i->first<m_begins[dst+1])) ++dst;

				if (dst == mp_rank) {
					m_local[i->first] += i->second;
				} else {
					// send to rank
					MPI_Send((void*)&(i->second), 1, MPI_INT, dst, tag_value, MPI_COMM_WORLD);
					MPI_Send((void*)&(i->first), 1, mp_type, dst, tag_key, MPI_COMM_WORLD);
				}
			}
			value = 0;
			for (dst = 0; dst < mp_size; ++dst) {
				if (dst != mp_rank) {
					MPI_Send(&value, 1, MPI_INT, dst, tag_value, MPI_COMM_WORLD);
				}
			}
		}


		{ // RECV MAP
			for (int src = 0; src < mp_size; ++src) {
				if (src != mp_rank) {
					while (true) {
						MPI_Recv(&value, 1, MPI_INT, src, tag_value, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						if (value == 0) break;


						MPI_Recv(&key, 1, mp_type, src, tag_key, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						m_local[key] += value;
					}
				}
			}
		}

		{ // AHNNIHILATION
			for (auto i = m_local.begin(); i != m_local.end(); ) {
				if (i->second == 0) {
					i = m_local.erase(i);
				} else {
					++i;
				}
			}
		}

		{ // SEND/RECV SIZES
			int size = m_local.size();
			MPI_Allgather(&size, 1, MPI_INT, m_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
		}

		{ // SEND TO NEIGHBOUR
			int total = 0;
			for (auto s : m_sizes) total += s;
			int q = total / mp_size;
			int r = total % mp_size;

			int left = 0;
			for (int i = 0; i < mp_rank; ++i) left += m_sizes[i];
			int right = total - m_sizes[mp_rank] - left;

			int tleft = q * mp_rank + std::min(mp_rank, r);
			int tright = total - tleft - q;
			if (mp_rank < r) tright -= 1;

			// [q+1] [q+1] ... [q+1] [q] [q] ... [q]
			// \------ r times ----/

			// SEND LEFT <-- x
			int dst = mp_rank - 1;
			if (dst >= 0) {
				if (left < tleft) {
					// send (tleft - left) to the left
					int sleft = std::min<int>(m_local.size(), tleft - left);
					// begin by send map.begin
					auto i = m_local.begin();
					for (int n = 0; n < sleft; ++n) {
						MPI_Send((void*)&(i->second), 1, MPI_INT, dst, tag_value, MPI_COMM_WORLD);
						MPI_Send((void*)&(i->first), 1, mp_type, dst, tag_key, MPI_COMM_WORLD);

						++i;
					}
					m_local.erase(m_local.begin(), i);
				}
				value = 0;
				MPI_Send(&value, 1, MPI_INT, dst, tag_value, MPI_COMM_WORLD);
			}

			// RECV FROM RIGHT x <--
			int src = mp_rank + 1;
			if (src < mp_size) {
				while (true) {
					MPI_Recv(&value, 1, MPI_INT, src, tag_value, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (value == 0) break;

					MPI_Recv(&key, 1, mp_type, src, tag_key, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_local.insert(m_local.end(), std::make_pair(key, value));
				}
			}

			// SEND RIGHT x -->
			dst = mp_rank + 1;
			if (dst < mp_size) {
				if (right < tright) {
					int sright = std::min<int>(m_local.size(), tright - right);
					// begin by send end
					auto i = m_local.end();
					for (int n = 0; n < sright; ++n) {
						--i;

						MPI_Send((void*)&(i->second), 1, MPI_INT, dst, tag_value, MPI_COMM_WORLD);
						MPI_Send((void*)&(i->first), 1, mp_type, dst, tag_key, MPI_COMM_WORLD);
					}
					m_local.erase(i, m_local.end());
				}
				value = 0;
				MPI_Send(&value, 1, MPI_INT, dst, tag_value, MPI_COMM_WORLD);
			}

			// RECV FROM LEFT --> x
			src = mp_rank - 1;
			if (src >= 0) {
				while (true) {
					MPI_Recv(&value, 1, MPI_INT, src, tag_value, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (value == 0) break;

					MPI_Recv(&key, 1, mp_type, src, tag_key, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_local.insert(m_local.begin(), std::make_pair(key, value));
				}

			}
		}

		{ // SEND/RECV BEGINS
			K beg;
			if (m_local.size() > 0) beg = m_local.begin()->first;
			MPI_Allgather(&beg, 1, mp_type, m_begins.data(), 1, mp_type, MPI_COMM_WORLD);
		}

		{ // SEND/RECV SIZES
			int size = m_local.size();
			MPI_Allgather(&size, 1, MPI_INT, m_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
		}
	}

	typename std::map<K,int>::iterator begin() { return m_local.begin(); }
	typename std::map<K,int>::iterator end()   { return m_local.end();   }
	size_t size() const { return m_local.size(); }
	size_t total_size() const {
		size_t n = 0;
		for (auto x : m_sizes) n += x;
		return n;
	}
	int count() const {
		int n = 0;
		for (auto i = m_local.begin(); i != m_local.end(); ++i) n += abs(i->second);
		return n;
	}

private:
	int mp_rank;
	int mp_size;
	MPI_Datatype mp_type;

	std::vector<int> m_sizes;
	std::vector<K> m_begins;
	std::map<K,int> m_local;
};

#endif // MPI_DATA_HH
