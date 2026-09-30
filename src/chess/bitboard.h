// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "types.h"

namespace pointchess {

// Population count
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

// Index of least significant bit, b must be non-zero
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }

// Index of most significant bit, b must be non-zero
inline int msb(Bitboard b) { return 63 ^ __builtin_clzll(b); }

// Extract and clear least significant bit
inline Square pop_lsb(Bitboard& b) {
    Square s = Square(lsb(b));
    b &= b - 1;
    return s;
}

// Directions for king/rook/bishop attacks
constexpr int NORTH = 8;
constexpr int SOUTH = -8;
constexpr int EAST = 1;
constexpr int WEST = -1;

// Pawn attack offsets per color
inline int pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

// Magic bitboard for sliding pieces
struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    unsigned shift;

    Bitboard index(Bitboard occupied) const {
#if defined(__SIZEOF_INT128__)
        return (occupied * magic) >> shift;
#else
        unsigned lo = unsigned(occupied & 0xFFFFFFFFULL);
        unsigned hi = unsigned(occupied >> 32);
        unsigned mlo = unsigned(magic & 0xFFFFFFFFULL);
        unsigned mhi = unsigned(magic >> 32);
        return Bitboard((Bitboard(lo) * mlo) ^ (Bitboard(hi) * mhi)) >> shift;
#endif
    }
};

extern Magic RookMagics[64];
extern Magic BishopMagics[64];

void init_magics();

// Attack lookups
Bitboard rook_attacks(Square sq, Bitboard occupied);
Bitboard bishop_attacks(Square sq, Bitboard occupied);
Bitboard queen_attacks(Square sq, Bitboard occupied);

// Non-sliding attacks
Bitboard knight_attacks(Square sq);
Bitboard king_attacks(Square sq);
Bitboard pawn_attacks(Color c, Square sq);

// Line/ray helpers
Bitboard between_bb(Square s1, Square s2);
Bitboard line_bb(Square s1, Square s2);
bool aligned(Square s1, Square s2, Square s3);

// Distance tables
extern int SquareDistance[64][64];
void init_distance();

// File/rank helpers
inline Bitboard files_adjacent(File f) {
    Bitboard b = 0;
    if (f > FILE_A) b |= file_bb(File(f - 1));
    if (f < FILE_H) b |= file_bb(File(f + 1));
    return b;
}

} // namespace pointchess
