// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "types.h"
#include "bitboard.h"
#include <vector>

namespace pointchess {

// Zobrist keys
struct Zobrist {
    Bitboard psq[COLOR_NB][7][64];   // [color][piece type][square]
    Bitboard enpassant[64];
    Bitboard castling[16];
    Bitboard side;
    Bitboard no_pawns;
};

extern Zobrist Zob;

void init_zobrist();

// State info for make/unmake
struct StateInfo {
    int castlingRights;
    int rule50;
    int pliesFromNull;
    Square epSquare;
    Bitboard checkersBB;
    Piece captured;
    Bitboard key;
    Bitboard pawnKey;
    Bitboard materialKey;
    Move lastMove;
    int repetition;

    StateInfo() : castlingRights(NO_CASTLING), rule50(0), pliesFromNull(0),
                  epSquare(SQ_NONE), checkersBB(0), captured(NO_PIECE),
                  key(0), pawnKey(0), materialKey(0), lastMove(MOVE_NONE),
                  repetition(0) {}
};

class Board {
public:
    Board();

    void set_fen(const std::string& fen);
    std::string fen() const;

    // Piece placement
    Piece piece_on(Square s) const;
    Bitboard pieces(PieceType pt) const;
    Bitboard pieces(Color c) const;
    Bitboard pieces() const;
    Bitboard pieces(Color c, PieceType pt) const;
    Square king_square(Color c) const;

    // Side to move
    Color side_to_move() const;

    // Castling
    int can_castle(Color c) const;
    int can_castle(CastlingRight cr) const;
    bool castling_impeded(CastlingRight cr) const;
    Square castling_rook_square(CastlingRight cr) const;

    // En passant
    Square ep_square() const;

    // Check
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    bool in_check() const;

    // Attack detection
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    bool legal(Move m) const;
    bool pseudo_legal(Move m) const;
    bool gives_check(Move m) const;

    // SEE
    int see(Move m) const;
    int see_ge(Move m, int threshold) const;

    // Move making
    void do_move(Move m, StateInfo& newSt);
    void do_move(Move m, StateInfo& newSt, bool givesCheck);
    void undo_move(Move m);
    void do_null_move(StateInfo& newSt);
    void undo_null_move();

    // Repetition
    int is_repetition(int ply) const;
    bool is_draw(int ply) const;

    // Insufficient material
    bool insufficient_material() const;

    // Hashing
    Bitboard key() const;
    Bitboard compute_key() const;

    // History access
    const StateInfo& state() const { return st; }
    StateInfo& state_ref() { return st; }
    const std::vector<StateInfo>& history() const { return history_; }

    // Move history
    Move last_move() const;

    // Piece count
    int count(PieceType pt, Color c) const;

    // Debug
    std::string pretty() const;

private:
    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);

    Bitboard pieces_bb[16];  // indexed by Piece
    Bitboard byColor[COLOR_NB];
    Bitboard byColorType[COLOR_NB][7];
    Piece board[64];
    int pieceCount[16];
    int castlingRights;
    Square epSquare;
    Color side;
    int rule50;
    int pliesFromNull;
    Bitboard checkersBB;
    Bitboard key_;
    Bitboard pawnKey_;
    Bitboard materialKey_;
    StateInfo st;
    std::vector<StateInfo> history_;
    Move lastMove_;
};

// Static exchange evaluation values
extern int PieceValue[7];

} // namespace pointchess
