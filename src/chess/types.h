// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from the Glaurung codebase lineage (Tord Romstad, GPL-2.0-or-later).

#pragma once

#include <cstdint>
#include <string>

namespace pointchess {

using Bitboard = std::uint64_t;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };

enum PieceType : int {
    NO_PIECE_TYPE = 0,
    PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
    ALL_PIECES = 0
};

enum Piece : int {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 9, B_KNIGHT = 10, B_BISHOP = 11, B_ROOK = 12, B_QUEEN = 13, B_KING = 14
};

enum Square : int {
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2 = 8, B2, C2, D2, E2, F2, G2, H2,
    A3 = 16, B3, C3, D3, E3, F3, G3, H3,
    A4 = 24, B4, C4, D4, E4, F4, G4, H4,
    A5 = 32, B5, C5, D5, E5, F5, G5, H5,
    A6 = 40, B6, C6, D6, E6, F6, G6, H6,
    A7 = 48, B7, C7, D7, E7, F7, G7, H7,
    A8 = 56, B8, C8, D8, E8, F8, G8, H8,
    SQ_NONE = 64
};

enum File : int { FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };
enum Rank : int { RANK_1 = 0, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };

enum CastlingRight : int {
    NO_CASTLING = 0,
    WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8,
    KING_SIDE = WHITE_OO | BLACK_OO,
    QUEEN_SIDE = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING = WHITE_CASTLING | BLACK_CASTLING
};

enum MoveType : int {
    NORMAL = 0, PROMOTION = 1 << 14, EN_PASSANT = 2 << 14, CASTLING = 3 << 14
};

constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY = 246;
constexpr int VALUE_NONE = 32002;
constexpr int VALUE_ZERO = 0;
constexpr int VALUE_DRAW = 0;
constexpr int VALUE_MATE = 32000;
constexpr int VALUE_INFINITE = 32001;
constexpr int VALUE_TB_WIN = VALUE_MATE - MAX_PLY;

inline constexpr Color operator~(Color c) { return Color(c ^ BLACK); }
inline constexpr PieceType type_of(Piece pc) { return PieceType(pc & 7); }
inline constexpr Color color_of(Piece pc) { return Color(pc >> 3); }
inline constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) | pt); }
inline constexpr Square make_square(File f, Rank r) { return Square((r << 3) | f); }
inline constexpr File file_of(Square s) { return File(s & 7); }
inline constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
inline constexpr int square_distance(Square a, Square b) {
    int df = int(file_of(a)) - int(file_of(b));
    int dr = int(rank_of(a)) - int(rank_of(b));
    return (df > dr) ? df : dr;
}
inline constexpr Square relative_square(Color c, Square s) {
    return Square(s ^ (c ? 56 : 0));
}
inline constexpr Rank relative_rank(Color c, Rank r) { return Rank(r ^ (c ? RANK_8 : RANK_1)); }
inline constexpr bool is_ok(Square s) { return s >= A1 && s <= H8; }

inline constexpr Bitboard file_bb(File f) { return 0x0101010101010101ULL << f; }
inline constexpr Bitboard rank_bb(Rank r) { return 0xFFULL << (r * 8); }

inline constexpr Bitboard operator&(Bitboard b, Square s) { return b & (1ULL << s); }
inline constexpr Bitboard operator|(Bitboard b, Square s) { return b | (1ULL << s); }
inline constexpr Bitboard operator^(Bitboard b, Square s) { return b ^ (1ULL << s); }
inline constexpr bool more_than_one(Bitboard b) { return b & (b - 1); }

// Move encoding: bits 0-5 from, 6-11 to, 12-13 promotion piece type, 14-15 flags
enum Move : int { MOVE_NONE = 0, MOVE_NULL = 65 };

inline constexpr Square move_from(Move m) { return Square(m & 0x3F); }
inline constexpr Square move_to(Move m) { return Square((m >> 6) & 0x3F); }
inline constexpr int move_flags(Move m) { return m & 0xC000; }
inline constexpr PieceType promotion_type(Move m) { return PieceType(((m >> 12) & 3) + KNIGHT); }
inline constexpr Move make_move(Square from, Square to) { return Move(from | (to << 6)); }
inline constexpr Move make_move(Square from, Square to, MoveType mt) {
    return Move(from | (to << 6) | mt);
}
inline constexpr Move make_promotion(Square from, Square to, PieceType pt) {
    return Move(from | (to << 6) | PROMOTION | ((pt - KNIGHT) << 12));
}
inline constexpr bool is_ok_move(Move m) { return m != MOVE_NONE && m != MOVE_NULL; }

const char* const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::string square_to_string(Square s);
std::string move_to_string(Move m);

} // namespace pointchess
