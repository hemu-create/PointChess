// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bitboard.h"

#include <cstring>
#include <random>

namespace pointchess {

Magic RookMagics[64];
Magic BishopMagics[64];
int SquareDistance[64][64];

// Precomputed attack tables for non-sliding pieces
static Bitboard KnightAttacks[64];
static Bitboard KingAttacks[64];
static Bitboard PawnAttacks[COLOR_NB][64];

// Rook and bishop attack tables (indexed via magics)
static Bitboard RookTable[64][4096];
static Bitboard BishopTable[64][512];

static Bitboard sliding_attacks(Square sq, Bitboard occupied, const int dirs[4][2]) {
    Bitboard attacks = 0;
    int r = int(rank_of(sq));
    int f = int(file_of(sq));
    for (int i = 0; i < 4; ++i) {
        int rr = r + dirs[i][0];
        int ff = f + dirs[i][1];
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
            attacks |= 1ULL << (rr * 8 + ff);
            if (occupied & (1ULL << (rr * 8 + ff))) break;
            rr += dirs[i][0];
            ff += dirs[i][1];
        }
    }
    return attacks;
}

static Bitboard rook_mask(Square sq) {
    Bitboard attacks = 0;
    int r = int(rank_of(sq));
    int f = int(file_of(sq));
    for (int i = 0; i < 4; ++i) {
        int rr = r + (i == 0) - (i == 1);
        int ff = f + (i == 2) - (i == 3);
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
            if ((i == 0 && rr == 7) || (i == 1 && rr == 0) ||
                (i == 2 && ff == 7) || (i == 3 && ff == 0)) {
                attacks |= 1ULL << (rr * 8 + ff);
                break;
            }
            attacks |= 1ULL << (rr * 8 + ff);
            rr += (i == 0) - (i == 1);
            ff += (i == 2) - (i == 3);
        }
    }
    return attacks;
}

static Bitboard bishop_mask(Square sq) {
    Bitboard attacks = 0;
    int r = int(rank_of(sq));
    int f = int(file_of(sq));
    for (int i = 0; i < 4; ++i) {
        int rr = r + (i == 0 || i == 1 ? 1 : -1);
        int ff = f + (i == 0 || i == 2 ? 1 : -1);
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
            if ((rr == 7 || rr == 0) && (ff == 7 || ff == 0)) {
                attacks |= 1ULL << (rr * 8 + ff);
                break;
            }
            attacks |= 1ULL << (rr * 8 + ff);
            rr += (i == 0 || i == 1 ? 1 : -1);
            ff += (i == 0 || i == 2 ? 1 : -1);
        }
    }
    return attacks;
}

static Bitboard find_magic(Square sq, bool is_rook, std::mt19937_64& rng) {
    Bitboard mask = is_rook ? rook_mask(sq) : bishop_mask(sq);
    int bits = popcount(mask);
    int table_size = 1 << bits;
    Bitboard table[4096];
    Bitboard occupied[4096];
    Bitboard ref[4096];

    // Enumerate all occupancies of the mask
    for (int i = 0; i < table_size; ++i) {
        Bitboard occ = 0;
        Bitboard m = mask;
        int j = 0;
        while (m) {
            Bitboard b = pop_lsb(m);
            if (i & (1 << j)) occ |= b;
            ++j;
        }
        occupied[i] = occ;
        ref[i] = sliding_attacks(sq, occ, is_rook
            ? (const int[4][2]){{1,0},{-1,0},{0,1},{0,-1}}
            : (const int[4][2]){{1,1},{1,-1},{-1,1},{-1,-1}});
    }

    for (int tries = 0; tries < 10000000; ++tries) {
        Bitboard magic = rng() & rng() & rng();
        if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;

        bool fail = false;
        std::memset(table, 0, sizeof(table));
        for (int i = 0; i < table_size && !fail; ++i) {
            unsigned idx = unsigned(((occupied[i] * magic) >> (64 - bits)));
            if (table[idx] == 0) {
                table[idx] = ref[i];
            } else if (table[idx] != ref[i]) {
                fail = true;
            }
        }
        if (!fail) return magic;
    }
    return 0; // Should never happen
}

void init_magics() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    std::mt19937_64 rng(0x123456789ABCDEFULL);

    const int rook_dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    const int bishop_dirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};

    for (int sq = 0; sq < 64; ++sq) {
        RookMagics[sq].mask = rook_mask(Square(sq));
        RookMagics[sq].shift = 64 - popcount(RookMagics[sq].mask);
        RookMagics[sq].magic = find_magic(Square(sq), true, rng);
        RookMagics[sq].attacks = RookTable[sq];

        BishopMagics[sq].mask = bishop_mask(Square(sq));
        BishopMagics[sq].shift = 64 - popcount(BishopMagics[sq].mask);
        BishopMagics[sq].magic = find_magic(Square(sq), false, rng);
        BishopMagics[sq].attacks = BishopTable[sq];

        // Fill attack tables
        int rbits = popcount(RookMagics[sq].mask);
        int bbits = popcount(BishopMagics[sq].mask);
        for (int i = 0; i < (1 << rbits); ++i) {
            Bitboard occ = 0;
            Bitboard m = RookMagics[sq].mask;
            int j = 0;
            while (m) {
                Bitboard b = pop_lsb(m);
                if (i & (1 << j)) occ |= b;
                ++j;
            }
            RookTable[sq][(occ * RookMagics[sq].magic) >> RookMagics[sq].shift] =
                sliding_attacks(Square(sq), occ, rook_dirs);
        }
        for (int i = 0; i < (1 << bbits); ++i) {
            Bitboard occ = 0;
            Bitboard m = BishopMagics[sq].mask;
            int j = 0;
            while (m) {
                Bitboard b = pop_lsb(m);
                if (i & (1 << j)) occ |= b;
                ++j;
            }
            BishopTable[sq][(occ * BishopMagics[sq].magic) >> BishopMagics[sq].shift] =
                sliding_attacks(Square(sq), occ, bishop_dirs);
        }
    }

    // Knight attacks
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard b = 0;
        int r = int(rank_of(Square(sq)));
        int f = int(file_of(Square(sq)));
        const int dr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        const int df[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        for (int i = 0; i < 8; ++i) {
            int rr = r + dr[i];
            int ff = f + df[i];
            if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
                b |= 1ULL << (rr * 8 + ff);
        }
        KnightAttacks[sq] = b;
    }

    // King attacks
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard b = 0;
        int r = int(rank_of(Square(sq)));
        int f = int(file_of(Square(sq)));
        for (int dr = -1; dr <= 1; ++dr)
            for (int df = -1; df <= 1; ++df) {
                if (dr == 0 && df == 0) continue;
                int rr = r + dr;
                int ff = f + df;
                if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
                    b |= 1ULL << (rr * 8 + ff);
            }
        KingAttacks[sq] = b;
    }

    // Pawn attacks
    for (int c = 0; c < 2; ++c) {
        for (int sq = 0; sq < 64; ++sq) {
            Bitboard b = 0;
            int r = int(rank_of(Square(sq)));
            int f = int(file_of(Square(sq)));
            int dir = (c == WHITE) ? 1 : -1;
            if (r + dir >= 0 && r + dir < 8) {
                if (f > 0) b |= 1ULL << ((r + dir) * 8 + f - 1);
                if (f < 7) b |= 1ULL << ((r + dir) * 8 + f + 1);
            }
            PawnAttacks[c][sq] = b;
        }
    }

    // Distance table
    for (int a = 0; a < 64; ++a)
        for (int b = 0; b < 64; ++b) {
            int dr = std::abs(int(rank_of(Square(a))) - int(rank_of(Square(b))));
            int df = std::abs(int(file_of(Square(a))) - int(file_of(Square(b))));
            SquareDistance[a][b] = (dr > df) ? dr : df;
        }
}

Bitboard rook_attacks(Square sq, Bitboard occupied) {
    return RookTable[sq][(occupied * RookMagics[sq].magic) >> RookMagics[sq].shift];
}

Bitboard bishop_attacks(Square sq, Bitboard occupied) {
    return BishopTable[sq][(occupied * BishopMagics[sq].magic) >> BishopMagics[sq].shift];
}

Bitboard queen_attacks(Square sq, Bitboard occupied) {
    return rook_attacks(sq, occupied) | bishop_attacks(sq, occupied);
}

Bitboard knight_attacks(Square sq) { return KnightAttacks[sq]; }
Bitboard king_attacks(Square sq) { return KingAttacks[sq]; }
Bitboard pawn_attacks(Color c, Square sq) { return PawnAttacks[c][sq]; }

Bitboard between_bb(Square s1, Square s2) {
    // Precomputed between table would be faster; compute on the fly for now
    Bitboard between = 0;
    int r1 = int(rank_of(s1)), f1 = int(file_of(s1));
    int r2 = int(rank_of(s2)), f2 = int(file_of(s2));
    if (r1 == r2 && f1 == f2) return 0;
    if (r1 == r2) {
        int lo = std::min(f1, f2), hi = std::max(f1, f2);
        for (int f = lo + 1; f < hi; ++f) between |= 1ULL << (r1 * 8 + f);
    } else if (f1 == f2) {
        int lo = std::min(r1, r2), hi = std::max(r1, r2);
        for (int r = lo + 1; r < hi; ++r) between |= 1ULL << (r * 8 + f1);
    } else if (std::abs(r1 - r2) == std::abs(f1 - f2)) {
        int dr = (r2 > r1) ? 1 : -1;
        int df = (f2 > f1) ? 1 : -1;
        int r = r1 + dr, f = f1 + df;
        while (r != r2 && f != f2) {
            between |= 1ULL << (r * 8 + f);
            r += dr; f += df;
        }
    }
    return between;
}

Bitboard line_bb(Square s1, Square s2) {
    Bitboard line = 0;
    int r1 = int(rank_of(s1)), f1 = int(file_of(s1));
    int r2 = int(rank_of(s2)), f2 = int(file_of(s2));
    if (r1 == r2) {
        for (int f = 0; f < 8; ++f) line |= 1ULL << (r1 * 8 + f);
    } else if (f1 == f2) {
        for (int r = 0; r < 8; ++r) line |= 1ULL << (r * 8 + f1);
    } else if (std::abs(r1 - r2) == std::abs(f1 - f2)) {
        int dr = (r2 > r1) ? 1 : -1;
        int df = (f2 > f1) ? 1 : -1;
        int r = r1, f = f1;
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            line |= 1ULL << (r * 8 + f);
            r += dr; f += df;
        }
    }
    return line;
}

bool aligned(Square s1, Square s2, Square s3) {
    return (between_bb(s1, s3) | between_bb(s2, s3) | between_bb(s1, s2)) & (1ULL << s2);
}

} // namespace pointchess
