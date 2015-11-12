#ifndef YOUNGTABLEAU
#define YOUNGTABLEAU

#include <tuple>
#include <cstdint>
#include <array>

constexpr size_t N = 4;   // SU(N)
constexpr size_t n = N*5; // n sites
typedef std::array<uint8_t, n> youngtableau; // N lignes de n/N valeurs

// FACTS
// si (i,i+1) ne sont pas dans la même ligne/colone alors si on les swap ça reste un tableau standard

#endif // YOUNGTABLEAU

