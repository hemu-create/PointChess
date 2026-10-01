// PointBeta Feature Extractor from Glaurung Position
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../glaurung/position.h"
#include "../nnue/pointbeta.h"

namespace pointchess {
namespace pointbeta {

// Piece mapping for dynamic non-pawn subspace
inline int piece_to_pointbeta_dynamic(Piece p, Color perspective) {
    Color pc = color_of_piece(p);
    PieceType pt = type_of_piece(p);
    if (pt <= PAWN || pt > KING) return -1;

    if (pc == perspective) {
        if (pt == KING) return -1; // Friendly king is tracked by coordinate
        return int(pt) - 2; // 0: N, 1: B, 2: R, 3: Q
    } else {
        if (pt == KING) return 8; // Enemy king type
        return (int(pt) - 2) + 4; // 4: n, 5: b, 6: r, 7: q
    }
}

inline Value evaluate_pointbeta(const Position& pos) {
    Square w_ksq = pos.king_square(WHITE);
    Square b_ksq = pos.king_square(BLACK);

    int w_k = int(w_ksq);
    int b_k = int(b_ksq) ^ 56;

    int wk[32], bk[32]; int nwk = 0, nbk = 0;
    int wp[16], bp[16]; int nwp = 0, nbp = 0;
    int wr[8],  br[8];  int nwr = 0, nbr = 0;
    int ws[32], bs[32]; int nws = 0, nbs = 0;

    int w_kr = w_k / 8, w_kf = w_k % 8;
    int b_kr = b_k / 8, b_kf = b_k % 8;

    for (int s = 0; s < 64; ++s) {
        Square sq = Square(s);
        Piece p = pos.piece_on(sq);
        if (p == NO_PIECE || p == EMPTY || p == OUTSIDE) continue;

        PieceType pt = type_of_piece(p);
        Color pc = color_of_piece(p);

        // 1. Dynamic King-Relative non-pawns
        int dyn_w = piece_to_pointbeta_dynamic(p, WHITE);
        if (dyn_w >= 0 && nwk < 32) wk[nwk++] = w_k * 576 + dyn_w * 64 + s;

        int dyn_b = piece_to_pointbeta_dynamic(p, BLACK);
        if (dyn_b >= 0 && nbk < 32) bk[nbk++] = b_k * 576 + dyn_b * 64 + (s ^ 56);

        // 2. King-Independent Pawns
        if (pt == PAWN) {
            if (pc == WHITE) {
                if (nwp < 16) wp[nwp++] = s;
                if (nbp < 16) bp[nbp++] = 64 + (s ^ 56);
            } else {
                if (nbp < 16) bp[nbp++] = s ^ 56;
                if (nwp < 16) wp[nwp++] = 64 + s;
            }
        }

        // 3. 5x5 King Zone Safety
        int r = s / 8, f = s % 8;
        int dr_w = r - w_kr, df_w = f - w_kf;
        if (dr_w >= -2 && dr_w <= 2 && df_w >= -2 && df_w <= 2 && nws < 32) {
            int off_w = (dr_w + 2) * 5 + (df_w + 2);
            int pt_w = (dyn_w >= 0 ? dyn_w : 0);
            ws[nws++] = w_k * 200 + (pt_w % 8) * 25 + off_w;
        }

        int r_b = (s ^ 56) / 8, f_b = f;
        int dr_b = r_b - b_kr, df_b = f_b - b_kf;
        if (dr_b >= -2 && dr_b <= 2 && df_b >= -2 && df_b <= 2 && nbs < 32) {
            int off_b = (dr_b + 2) * 5 + (df_b + 2);
            int pt_b = (dyn_b >= 0 ? dyn_b : 0);
            bs[nbs++] = b_k * 200 + (pt_b % 8) * 25 + off_b;
        }
    }

    if (nwr == 0) wr[nwr++] = 0;
    if (nbr == 0) br[nbr++] = 0;
    if (nws == 0) ws[nws++] = 0;
    if (nbs == 0) bs[nbs++] = 0;

    int stm = (pos.side_to_move() == WHITE) ? 0 : 1;
    int cp = GlobalPointBeta.evaluate(wk, nwk, bk, nbk,
                                      wp, nwp, bp, nbp,
                                      wr, nwr, br, nbr,
                                      ws, nws, bs, nbs,
                                      stm);

    return value_from_centipawns(cp);
}

} // namespace pointbeta
} // namespace pointchess
