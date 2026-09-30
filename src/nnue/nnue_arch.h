// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// Modern HalfKAv2 NNUE Neural Network Architecture with SCReL

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace pointchess {
namespace nnue {

// Modern HalfKAv2 Feature Dimensions (King + All Pieces including Kings)
// 11 Piece Types from friendly perspective:
// 0: Friendly Pawn, 1: Friendly Knight, 2: Friendly Bishop, 3: Friendly Rook, 4: Friendly Queen, 5: Friendly King
// 6: Enemy Pawn, 7: Enemy Knight, 8: Enemy Bishop, 9: Enemy Rook, 10: Enemy Queen
constexpr int NUM_PIECE_TYPES = 11;
constexpr int NUM_PIECE_SQUARES = 64;
constexpr int HALF_KA_PIECE_FEATURES = NUM_PIECE_TYPES * NUM_PIECE_SQUARES; // 704
constexpr int HALF_KA_FEATURES = 64 * HALF_KA_PIECE_FEATURES; // 45,056
constexpr int ACCUMULATOR_SIZE = 256; // 256 per perspective -> 512 total
constexpr int L1_SIZE = 32;
constexpr int L2_SIZE = 32;

// Quantization scales
constexpr int WEIGHT_SCALE_L0 = 256;
constexpr int WEIGHT_SCALE_L1 = 64;
constexpr int WEIGHT_SCALE_L2 = 64;
constexpr int WEIGHT_SCALE_OUT = 16;
constexpr int OUTPUT_SCALE = 16;

// Custom PointChess format magic
constexpr const char* PCHESS_MAGIC = "POINTCHESS_V2_HALFKAV2";

// Accumulator for one perspective (White or Black king)
struct alignas(64) Accumulator {
    int16_t values[ACCUMULATOR_SIZE];

    void clear(const int16_t* biases) {
        std::memcpy(values, biases, sizeof(values));
    }

    void add_feature(int feature_idx, const int16_t* weights) {
        const int16_t* w = weights + feature_idx * ACCUMULATOR_SIZE;
        for (int i = 0; i < ACCUMULATOR_SIZE; ++i) {
            values[i] += w[i];
        }
    }

    void sub_feature(int feature_idx, const int16_t* weights) {
        const int16_t* w = weights + feature_idx * ACCUMULATOR_SIZE;
        for (int i = 0; i < ACCUMULATOR_SIZE; ++i) {
            values[i] -= w[i];
        }
    }
};

// Network Weights structure
struct alignas(64) NetworkParameters {
    // Feature transformer: 45056 -> 256
    int16_t feature_weights[HALF_KA_FEATURES * ACCUMULATOR_SIZE];
    int16_t feature_biases[ACCUMULATOR_SIZE];

    // Layer 1: 512 (256 W + 256 B) -> 32
    int8_t l1_weights[ACCUMULATOR_SIZE * 2 * L1_SIZE];
    int32_t l1_biases[L1_SIZE];

    // Layer 2: 32 -> 32
    int8_t l2_weights[L1_SIZE * L2_SIZE];
    int32_t l2_biases[L2_SIZE];

    // Output layer: 32 -> 1
    int8_t out_weights[L2_SIZE];
    int32_t out_bias;

    NetworkParameters();
    void init_default_weights();
};

class NNUEEvaluation {
public:
    NNUEEvaluation();
    ~NNUEEvaluation() = default;

    // Custom .pchess format loader
    bool load_pchess_file(const std::string& filepath);
    bool load_network_file(const std::string& filepath);
    bool load_pnet_file(const std::string& filepath);
    bool is_loaded() const { return network_loaded; }

    // Evaluation for side to move
    int evaluate(int white_king_sq, int black_king_sq,
                 const int* white_features, int num_white_features,
                 const int* black_features, int num_black_features,
                 int side_to_move);

    // Fast evaluation from pre-computed accumulators
    int evaluate_accumulators(const Accumulator& us_acc, const Accumulator& them_acc);

    const NetworkParameters& params() const { return net; }
    NetworkParameters& params_mut() { return net; }

    // Modern HalfKAv2 Feature index helper:
    // King sq (0..63), piece_type (0..10), piece_sq (0..63)
    static inline int halfka_index(int king_sq, int piece_type, int sq) {
        return king_sq * HALF_KA_PIECE_FEATURES + piece_type * 64 + sq;
    }

private:
    NetworkParameters net;
    bool network_loaded;
    std::string current_file;
};

extern NNUEEvaluation GlobalNNUE;

} // namespace nnue
} // namespace pointchess
