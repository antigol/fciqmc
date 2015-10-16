#ifndef MPI_MAP_HH
#define MPI_MAP_HH

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <mpi/mpi.h>

template<class K>
class mpi_map
{
public:
	mpi_map(int rank, int size/*, MPI_Datatype type*/) :
		mp_rank(rank), mp_size(size)//, mp_type(type)
	{
		m_sizes.resize(mp_size, 0);
		m_begins.resize(mp_size);
	}

	void sync(const std::map<K,int>& map)
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
					m_local[i->first] += i->second;
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
						std::pair<K,int> kv = unserialize(message);
						m_local[kv.first] += kv.second;
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
			MPI_Allgather(&size, 1, MPI_INT, m_sizes.data(), mp_size, MPI_INT, MPI_COMM_WORLD);
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

			int dst = mp_rank - 1;
			if (dst >= 0) {
				if (left < tleft) {
					// send (tleft - left) to the left
					int sleft = std::min<int>(m_local.size(), tleft - left);
					// begin by send map.begin
					auto i = m_local.begin();
					for (int n = 0; n < sleft; ++n) {
						std::string message = serialize(*i);
						++i;
						len = message.size();
						MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
						MPI_Send((void*)message.data(), len, MPI_BYTE, dst, tag_data, MPI_COMM_WORLD);
					}
					m_local.erase(m_local.begin(), i);
				}
				len = 0;
				MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
			}

			dst = mp_rank + 1;
			if (dst < mp_size) {
				if (right < tright) {
					int sright = std::min<int>(m_local.size(), tright - right);
					// begin by send end
					auto i = m_local.end();
					for (int n = 0; n < sright; ++n) {
						--i;
						std::string message = serialize(*i);
						len = message.size();
						MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
						MPI_Send((void*)message.data(), len, MPI_BYTE, dst, tag_data, MPI_COMM_WORLD);
					}
					m_local.erase(i, m_local.end());
				}
				len = 0;
				MPI_Send(&len, 1, MPI_INT, dst, tag_length, MPI_COMM_WORLD);
			}
		}

		// RECV FROM NEIGHBOUR
		if (mp_rank > 0) {
			int len = 1;
			std::string message;
			while (len > 0) {
				MPI_Recv(&len, 1, MPI_INT, mp_rank-1, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (len > 0) {
					message.resize(len);
					MPI_Recv((void*)message.data(), len, MPI_BYTE, mp_rank-1, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_local.insert(m_local.begin(), unserialize(message));
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
					m_local.insert(m_local.end(), unserialize(message));
				}
			}
		}

		// SEND/RECV BEGINS
		{
			std::string message;
			if (m_local.size() > 0)
				message = serialize(*(m_local.begin()));
			len = message.size();
			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Send(&len, 1, MPI_INT, r, tag_length, MPI_COMM_WORLD);
					if (len > 0) MPI_Send((void*)message.data(), len, MPI_BYTE, r, tag_data, MPI_COMM_WORLD);
				}
			}

			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Recv(&len, 1, MPI_INT, r, tag_length, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (len > 0) {
						message.resize(len);
						MPI_Recv((void*)message.data(), len, MPI_BYTE, r, tag_data, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						m_begins[r] = unserialize(message).first;
					}
				} else {
					if (m_local.size() > 0) m_begins[r] = m_local.begin()->first;
				}
			}
		}
		/*
		if (0) {
			K beg;
			if (m_local.size() > 0)
				beg = m_local.begin()->first;

			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Send(&beg, 1, mp_type, r, 15, MPI_COMM_WORLD);
				}
			}

			for (int r = 0; r < mp_size; ++r) {
				if (r != mp_rank) {
					MPI_Recv(&beg, 1, mp_type, r, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					m_begins[r] = beg;
				} else {
					if (m_local.size() > 0) m_begins[r] = m_local.begin()->first;
				}
			}
		}
		if (0) {
			K beg;
			if (m_local.size() > 0) beg = m_local.begin()->first;
			MPI_Allgather(&beg, 1, mp_type, m_begins.data(), mp_size, mp_type, MPI_COMM_WORLD);
		}*/

		{ // SEND/RECV SIZES
			int size = m_local.size();
			MPI_Allgather(&size, 1, MPI_INT, m_sizes.data(), mp_size, MPI_INT, MPI_COMM_WORLD);
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
	//MPI_Datatype mp_type;

	std::vector<int> m_sizes;
	std::vector<K> m_begins;
	std::map<K,int> m_local;
};

#endif // MPI_MAP_HH
