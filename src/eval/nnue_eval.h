// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// Modern Incremental HalfKAv2 NNUE Engine with SIMD Vectorization

#pragma once

#include "../glaurung/position.h"
#include "../nnue/nnue_arch.h"
#include "pointbeta_eval.h"

namespace pointchess {
namespace nnue {

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

// Small direct-mapped eval cache
struct NNUEEvalCache {
    static const int SIZE = 16384;
    struct Entry { uint64_t key = 0; int32_t value = 0; };
    Entry table[SIZE];
    int hits = 0, lookups = 0;
    Value lookup(uint64_t key, bool& found) {
        lookups++;
        Entry& e = table[key & (SIZE - 1)];
        if (e.key == key) { hits++; found = true; return Value(e.value); }
        found = false; return Value(0);
    }
    void store(uint64_t key, Value v) {
        Entry& e = table[key & (SIZE - 1)];
        e.key = key; e.value = int(v);
    }
};
inline NNUEEvalCache& nnue_cache() { static NNUEEvalCache c; return c; }

// Incremental Accumulator Stack (White + Black per ply)
struct AccumulatorEntry {
    BigAccumulator acc[2]; // 0: White, 1: Black
    bool computed[2];
};

struct ThreadAccumulatorStack {
    AccumulatorEntry stack[104];

    void reset() {
        for (int i = 0; i < 104; ++i) {
            stack[i].computed[0] = false;
            stack[i].computed[1] = false;
        }
    }
};

inline ThreadAccumulatorStack& get_acc_stack(int thread_id = 0) {
    (void)thread_id;
    static thread_local ThreadAccumulatorStack stacks;
    return stacks;
}

inline void refresh_accumulator(const Position& pos, BigAccumulator& acc, Color c) {
    acc.clear(GlobalNNUE.params().feature_biases);
    Square ksq = pos.king_square(c);
    int k = (c == WHITE) ? int(ksq) : (int(ksq) ^ 56);
    const int16_t* weights = GlobalNNUE.params().feature_weights;

    Bitboard occ = pos.occupied_squares();
    while (occ) {
        Square sq = pop_1st_bit(&occ);
        Piece p = pos.piece_on(sq);
        int pt = piece_to_halfka_type(p, c);
        if (pt >= 0) {
            int s = (c == WHITE) ? int(sq) : (int(sq) ^ 56);
            int feat = NNUEEvaluation::halfka_index(k, pt, s);
            acc.add_feature(feat, weights);
        }
    }
}

// Incremental move updater in O(1) time
inline void update_accumulator_move(const Position& pos, Move m, int ply, int thread_id = 0) {
    if (ply < 0 || ply >= 100) return;
    auto& acc_stack = get_acc_stack(thread_id);
    AccumulatorEntry& curr = acc_stack.stack[ply];
    AccumulatorEntry& next = acc_stack.stack[ply + 1];

    Square from = move_from(m);
    Square to = move_to(m);
    Piece p = pos.piece_on(from);
    PieceType pt = type_of_piece(p);
    Color us = color_of_piece(p);

    const int16_t* weights = GlobalNNUE.params().feature_weights;
    Square w_ksq = pos.king_square(WHITE);
    Square b_ksq = pos.king_square(BLACK);
    int w_k = int(w_ksq);
    int b_k = int(b_ksq) ^ 56;

    next.acc[WHITE] = curr.acc[WHITE];
    next.acc[BLACK] = curr.acc[BLACK];
    next.computed[WHITE] = curr.computed[WHITE];
    next.computed[BLACK] = curr.computed[BLACK];

    if (pt == KING) {
        if (us == WHITE) next.computed[WHITE] = false;
        else next.computed[BLACK] = false;

        Color them = opposite_color(us);
        int k_them = (them == WHITE) ? w_k : b_k;
        int pt_them_from = piece_to_halfka_type(p, them);
        if (pt_them_from >= 0 && next.computed[them]) {
            int s_from = (them == WHITE) ? int(from) : (int(from) ^ 56);
            int s_to   = (them == WHITE) ? int(to)   : (int(to) ^ 56);
            next.acc[them].sub_feature(NNUEEvaluation::halfka_index(k_them, pt_them_from, s_from), weights);
            next.acc[them].add_feature(NNUEEvaluation::halfka_index(k_them, pt_them_from, s_to), weights);
        }
        return;
    }

    int w_pt = piece_to_halfka_type(p, WHITE);
    int b_pt = piece_to_halfka_type(p, BLACK);

    if (w_pt >= 0 && next.computed[WHITE]) {
        next.acc[WHITE].sub_feature(NNUEEvaluation::halfka_index(w_k, w_pt, int(from)), weights);
        next.acc[WHITE].add_feature(NNUEEvaluation::halfka_index(w_k, w_pt, int(to)), weights);
    }
    if (b_pt >= 0 && next.computed[BLACK]) {
        next.acc[BLACK].sub_feature(NNUEEvaluation::halfka_index(b_k, b_pt, int(from) ^ 56), weights);
        next.acc[BLACK].add_feature(NNUEEvaluation::halfka_index(b_k, b_pt, int(to) ^ 56), weights);
    }

    if (pos.move_is_capture(m)) {
        Square cap_sq = to;
        if (move_is_ep(m)) {
            cap_sq = make_square(square_file(to), square_rank(from));
        }
        Piece cap_p = pos.piece_on(cap_sq);
        if (cap_p != NO_PIECE && cap_p != EMPTY) {
            int w_cap = piece_to_halfka_type(cap_p, WHITE);
            int b_cap = piece_to_halfka_type(cap_p, BLACK);
            if (w_cap >= 0 && next.computed[WHITE]) {
                next.acc[WHITE].sub_feature(NNUEEvaluation::halfka_index(w_k, w_cap, int(cap_sq)), weights);
            }
            if (b_cap >= 0 && next.computed[BLACK]) {
                next.acc[BLACK].sub_feature(NNUEEvaluation::halfka_index(b_k, b_cap, int(cap_sq) ^ 56), weights);
            }
        }
    }

    if (move_is_castle(m)) {
        Square r_from, r_to;
        if (to > from) {
            r_from = make_square(FILE_H, square_rank(from));
            r_to   = make_square(FILE_F, square_rank(from));
        } else {
            r_from = make_square(FILE_A, square_rank(from));
            r_to   = make_square(FILE_D, square_rank(from));
        }
        Piece rook = pos.piece_on(r_from);
        int w_r = piece_to_halfka_type(rook, WHITE);
        int b_r = piece_to_halfka_type(rook, BLACK);
        if (w_r >= 0 && next.computed[WHITE]) {
            next.acc[WHITE].sub_feature(NNUEEvaluation::halfka_index(w_k, w_r, int(r_from)), weights);
            next.acc[WHITE].add_feature(NNUEEvaluation::halfka_index(w_k, w_r, int(r_to)), weights);
        }
        if (b_r >= 0 && next.computed[BLACK]) {
            next.acc[BLACK].sub_feature(NNUEEvaluation::halfka_index(b_k, b_r, int(r_from) ^ 56), weights);
            next.acc[BLACK].add_feature(NNUEEvaluation::halfka_index(b_k, b_r, int(r_to) ^ 56), weights);
        }
    }

    if (move_promotion(m)) {
        Piece prom_p = piece_of_color_and_type(us, move_promotion(m));
        int w_prom = piece_to_halfka_type(prom_p, WHITE);
        int b_prom = piece_to_halfka_type(prom_p, BLACK);
        if (w_pt >= 0 && w_prom >= 0 && next.computed[WHITE]) {
            next.acc[WHITE].sub_feature(NNUEEvaluation::halfka_index(w_k, w_pt, int(to)), weights);
            next.acc[WHITE].add_feature(NNUEEvaluation::halfka_index(w_k, w_prom, int(to)), weights);
        }
        if (b_pt >= 0 && b_prom >= 0 && next.computed[BLACK]) {
            next.acc[BLACK].sub_feature(NNUEEvaluation::halfka_index(b_k, b_pt, int(to) ^ 56), weights);
            next.acc[BLACK].add_feature(NNUEEvaluation::halfka_index(b_k, b_prom, int(to) ^ 56), weights);
        }
    }
}

// Evaluate a position using Modern Incremental HalfKAv2 or PointBeta NNUE
inline Value evaluate_nnue(const Position& pos, int ply = 0, int thread_id = 0) {
    uint64_t key = uint64_t(pos.get_key()) ^ (pos.side_to_move() == WHITE ? 0x9e3779b97f4a7c15ULL : 0);
    bool found = false;
    Value cached = nnue_cache().lookup(key, found);
    if (found) return cached;

    if (pointbeta::GlobalPointBeta.is_loaded()) {
        Value pb_val = pointbeta::evaluate_pointbeta(pos);
        nnue_cache().store(key, pb_val);
        return pb_val;
    }

    auto& acc_stack = get_acc_stack(thread_id);
    int safe_ply = (ply >= 0 && ply < 100) ? ply : 0;
    AccumulatorEntry& entry = acc_stack.stack[safe_ply];

    if (!entry.computed[WHITE]) {
        refresh_accumulator(pos, entry.acc[WHITE], WHITE);
        entry.computed[WHITE] = true;
    }
    if (!entry.computed[BLACK]) {
        refresh_accumulator(pos, entry.acc[BLACK], BLACK);
        entry.computed[BLACK] = true;
    }

    int cp_score;
    if (pos.side_to_move() == WHITE) {
        cp_score = GlobalNNUE.evaluate_accumulators(entry.acc[WHITE], entry.acc[BLACK]);
    } else {
        cp_score = GlobalNNUE.evaluate_accumulators(entry.acc[BLACK], entry.acc[WHITE]);
    }

    Value v = value_from_centipawns(cp_score);
    nnue_cache().store(key, v);
    return v;
}

} // namespace nnue
} // namespace pointchess
