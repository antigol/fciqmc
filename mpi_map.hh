#ifndef MPI_MAP_HH
#define MPI_MAP_HH

#include <map>
#include <vector>
#include <string>
#include <iostream>
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

		//std::cout << mp_rank << ": SEND MAP" << endl;
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


		//std::cout << mp_rank << ": RECV MAP" << endl;
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

		//std::cout << mp_rank << ": SIZES" << endl;
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

		int total = 0;
		for (size_t s : m_sizes) total += s;
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

		//std::cout << mp_rank << ": SEND NEIGH" << endl;
		{ // SEND TO NEIGHBOUR
			int dst = mp_rank - 1;
			if (dst >= 0) {
				if (left < tleft) {
					// send (tleft - left) to the left
					int sleft = std::min<int>(m_map.size(), tleft - left);
					// begin by send map.begin
					auto i = m_map.begin();
					for (int n = 0; n < sleft; ++n) {
						std::string message = serialize(*i);
						++i;
						len = message.size();
						MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
						MPI_Send((void*)message.data(), len, MPI_BYTE, dst, tag_data, MPI_COMM_WORLD);
					}
					m_map.erase(m_map.begin(), i);
				}
				len = 0;
				MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
			}
		}

		{
			int dst = mp_rank + 1;
			if (dst < mp_size) {
				if (right < tright) {
					int sright = std::min<int>(m_map.size(), tright - right);
					// begin by send end
					auto i = m_map.end();
					for (int n = 0; n < sright; ++n) {
						--i;
						std::string message = serialize(*i);
						len = message.size();
						MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
						MPI_Send((void*)message.data(), len, MPI_BYTE, dst, tag_data, MPI_COMM_WORLD);
					}
					m_map.erase(i, m_map.end());
				}
				len = 0;
				MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
			}
		}

		//std::cout << mp_rank << ": RECV NEIGH" << endl;
		// RECV FROM NEIGHBOUR
		if (mp_rank > 0) {
			int len = 1;
			std::string message;
			while (len > 0) {
				MPI_Recv(&len, 1, MPI_INT, mp_rank-1, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (len > 0) {
					message.resize(len);
					MPI_Recv((void*)message.data(), len, MPI_BYTE, mp_rank-1, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_map.insert(m_map.begin(), unserialize(message));
				}
			}
		}

		if (mp_rank < mp_size - 1) {
			int len = 1;
			std::string message;
			while (len > 0) {
				MPI_Recv(&len, 1, MPI_INT, mp_rank+1, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (len > 0) {
					message.resize(len);
					MPI_Recv((void*)message.data(), len, MPI_BYTE, mp_rank+1, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_map.insert(m_map.end(), unserialize(message));
				}
			}
		}

		// SEND/RECV BEGINS
		{
			//std::cout << mp_rank << ": SEND BEGINS" << endl;
			std::string message;
			if (m_map.size() > 0)
				message = serialize(*(m_map.begin()));
			len = message.size();
			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Send(&len, 1, MPI_INT, r, tag_length, MPI_COMM_WORLD);
					if (len > 0) MPI_Send((void*)message.data(), len, MPI_BYTE, r, tag_data, MPI_COMM_WORLD);
				}
			}

			//std::cout << mp_rank << ": RECV BEGINS" << endl;
			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Recv(&len, 1, MPI_INT, r, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (len > 0) {
						message.resize(len);
						MPI_Recv((void*)message.data(), len, MPI_BYTE, r, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						m_begins[r] = unserialize(message).first;
					}
				} else {
					if (m_map.size() > 0) m_begins[r] = m_map.begin()->first;
				}
			}
		}

		//std::cout << mp_rank << ": SIZES" << endl;
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
	}

	typename std::map<K,T>::iterator begin() { return m_map.begin(); }
	typename std::map<K,T>::iterator end()   { return m_map.end();   }

private:
	int mp_rank;
	int mp_size;

	std::vector<size_t> m_sizes;
	std::vector<K> m_begins;
	std::map<K,T> m_map;
};

#endif // MPI_MAP_HH
