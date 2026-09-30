// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// Modern HalfKAv2 Feature Extractor

#pragma once

#include "../glaurung/position.h"
#include "../nnue/nnue_arch.h"

namespace pointchess {
namespace nnue {

// Maps a Glaurung Piece to HalfKAv2 piece type:
// 0..5: Friendly P, N, B, R, Q, K
// 6..10: Enemy P, N, B, R, Q
inline int piece_to_halfka_type(Piece p, Color perspective) {
    Color pc_color = color_of_piece(p);
    PieceType pt = type_of_piece(p);
    if (pt < PAWN || pt > KING) return -1;

    if (pc_color == perspective) {
        return int(pt) - 1; // 0..5 (P, N, B, R, Q, K)
    } else {
        if (pt == KING) return -1; // Enemy king is tracked via king-square coordinate
        return (int(pt) - 1) + 6; // 6..10 (Enemy P, N, B, R, Q)
    }
}

// Evaluate a position using Modern HalfKAv2 NNUE
inline Value evaluate_nnue(const Position& pos) {
    Square w_ksq = pos.king_square(WHITE);
    Square b_ksq = pos.king_square(BLACK);

    // Black perspective flips squares along vertical axis (sq ^ 56)
    int w_k = int(w_ksq);
    int b_k = int(b_ksq) ^ 56;

    int w_features[34];
    int b_features[34];
    int num_w = 0;
    int num_b = 0;

    // Scan board squares
    for (int s = 0; s < 64; ++s) {
        Square sq = Square(s);
        Piece p = pos.piece_on(sq);
        if (p == NO_PIECE || p == EMPTY || p == OUTSIDE) continue;

        // White perspective
        int w_pt = piece_to_halfka_type(p, WHITE);
        if (w_pt >= 0 && num_w < 34) {
            w_features[num_w++] = NNUEEvaluation::halfka_index(w_k, w_pt, s);
        }

        // Black perspective
        int b_pt = piece_to_halfka_type(p, BLACK);
        if (b_pt >= 0 && num_b < 34) {
            int b_sq = s ^ 56;
            b_features[num_b++] = NNUEEvaluation::halfka_index(b_k, b_pt, b_sq);
        }
    }

    int stm = (pos.side_to_move() == WHITE) ? 0 : 1;
    int cp_score = GlobalNNUE.evaluate(w_k, b_k,
                                       w_features, num_w,
                                       b_features, num_b,
                                       stm);

    return value_from_centipawns(cp_score);
}

} // namespace nnue
} // namespace pointchess
