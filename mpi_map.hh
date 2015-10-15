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

	void mpi_sumup(const std::map<K,T>& map)
	{
		enum {
			tag_length = 0,
			tag_data,
			tag_size
		};
		int len;

		{ // SEND MAP
			int dst = 0;
			for (auto i = map.begin(); i != map.end(); ++i) {
				// find rank to send data to
				while (dst+1 < mp_size && m_sizes[dst+1]!=0 && !(i->first<m_begins[dst+1])) ++dst;

				if (dst == mp_rank) {
					m_map.insert(*i);
				} else {
					// send to rank
					std::string message = serialize(*i);
					len = message.size();
					MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
					MPI_Send((void*)message.data(), len, MPI_BYTE, dst, tag_data, MPI_COMM_WORLD);
				}
			}
			len = 0;
			for (dst = 0; dst < mp_size; ++dst) {
				if (dst != mp_rank) {
					MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
				}
			}
		}

		{ // RECV MAP
			for (int src = 0; src < mp_size; ++src) {
				if (src != mp_rank) {
					len = 1;
					std::string message;
					while (len > 0) {
						MPI_Recv(&len, 1, MPI_INT, src, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						if (len == 0) break;
						message.resize(len);
						MPI_Recv((void*)message.data(), len, MPI_BYTE, src, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						m_map.insert(unserialize(message));
					}
				}
			}
		}

		{ // SEND/RECV SIZES
			int size = m_map.size();
			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Send(&size, 1, MPI_INT, r, tag_size, MPI_COMM_WORLD);
				}
			}
			for (int r = 0; r < mp_size; ++r) {
				if (r == mp_rank) {
					m_sizes[mp_rank] = m_map.size();
				} else {
					MPI_Recv(&size, 1, MPI_INT, r, tag_size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_sizes[r] = size;
				}
			}
		}

		size_t total = 0;
		for (size_t s : m_sizes) total += s;
		size_t q = total / mp_size;
		size_t r = total % mp_size;

		size_t left = 0;
		for (int i = 0; i < mp_rank; ++i) left += m_sizes[i];
		size_t right = total - m_sizes[mp_rank] - left;

		size_t tleft = q * mp_rank + std::min(mp_rank, r);
		size_t tright = total - tleft - q;
		if (mp_rank < r) tright -= 1;

		// [q+1] [q+1] ... [q+1] [q] [q] ... [q]
		// \------ r times ----/

		// send/recv to neighbours
		if (left < tleft) {
			// send (tleft - left) to the left
			int sleft = std::min(m_map.size(), tleft - left);
			// begin by send map.begin
		}
		if (right < tright) {
			int sright = std::min(m_map.size(), tright - right);
			// begin by send end

		}

		// recv
		if (mp_rank > 0) {
			int len = 1;
			while (len > 0) {
				MPI_Recv(&len, 1, MPI_INT, mp_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (len > 0) {
					MPI_Recv(&data, len, MPI_BYTE, mp_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_map.insert(m_map.begin(), pair);
				}
			}
		}
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
