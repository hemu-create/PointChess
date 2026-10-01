// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// PointChess Dual-Network NNUE (Big Net ~11.5M Parameters + Fast Small Net)

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

// Modern HalfKAv2 Feature Dimensions (King + All 11 Piece Types)
constexpr int NUM_PIECE_TYPES = 11; // 6 Friendly (P, N, B, R, Q, K), 5 Enemy (P, N, B, R, Q)
constexpr int NUM_PIECE_SQUARES = 64;
constexpr int HALF_KA_PIECE_FEATURES = NUM_PIECE_TYPES * NUM_PIECE_SQUARES; // 704
constexpr int HALF_KA_FEATURES = 64 * HALF_KA_PIECE_FEATURES; // 45,056

// Big Network Dimensions (11.5 Million Parameters total)
constexpr int BIG_ACCUMULATOR_SIZE = 256; // 256 per perspective -> 512 total
constexpr int BIG_L1_SIZE = 32;
constexpr int BIG_L2_SIZE = 32;

// Fast Small Network Dimensions (for quick cutoff evaluation)
constexpr int SMALL_ACCUMULATOR_SIZE = 32;

// Quantization scales
constexpr int WEIGHT_SCALE_L0 = 256;
constexpr int WEIGHT_SCALE_L1 = 64;
constexpr int WEIGHT_SCALE_L2 = 64;
constexpr int WEIGHT_SCALE_OUT = 16;
constexpr int OUTPUT_SCALE = 16;

// PointChess Dual-Net Custom Format Magic
constexpr const char* PCHESS_MAGIC = "POINTCHESS_DUAL_NET_V2";

// Accumulator for one perspective
struct alignas(64) BigAccumulator {
    int16_t values[BIG_ACCUMULATOR_SIZE];

    void clear(const int16_t* biases) {
        std::memcpy(values, biases, sizeof(values));
    }

    void add_feature(int feature_idx, const int16_t* weights) {
        const int16_t* w = weights + feature_idx * BIG_ACCUMULATOR_SIZE;
        for (int i = 0; i < BIG_ACCUMULATOR_SIZE; ++i) {
            values[i] += w[i];
        }
    }

    void sub_feature(int feature_idx, const int16_t* weights) {
        const int16_t* w = weights + feature_idx * BIG_ACCUMULATOR_SIZE;
        for (int i = 0; i < BIG_ACCUMULATOR_SIZE; ++i) {
            values[i] -= w[i];
        }
    }
};

// 11.5-Million Parameter Big Network Weights structure
struct alignas(64) BigNetworkParameters {
    // Feature Transformer: 45,056 x 256 = 11,534,336 parameters (11.5 Million)
    int16_t feature_weights[HALF_KA_FEATURES * BIG_ACCUMULATOR_SIZE];
    int16_t feature_biases[BIG_ACCUMULATOR_SIZE];

    // Layer 1: 512 -> 32
    int8_t l1_weights[BIG_ACCUMULATOR_SIZE * 2 * BIG_L1_SIZE];
    int32_t l1_biases[BIG_L1_SIZE];

    // Layer 2: 32 -> 32
    int8_t l2_weights[BIG_L1_SIZE * BIG_L2_SIZE];
    int32_t l2_biases[BIG_L2_SIZE];

    // Output Layer: 32 -> 1
    int8_t out_weights[BIG_L2_SIZE];
    int32_t out_bias;

    BigNetworkParameters();
    void init_default_weights();
};

class NNUEEvaluation {
public:
    NNUEEvaluation();
    ~NNUEEvaluation() = default;

    bool load_pchess_file(const std::string& filepath);
    bool load_network_file(const std::string& filepath);
    bool load_pnet_file(const std::string& filepath);
    bool is_loaded() const { return network_loaded; }

    // Evaluation using 11.5M Parameter Big Network with SCReL
    int evaluate(int white_king_sq, int black_king_sq,
                 const int* white_features, int num_white_features,
                 const int* black_features, int num_black_features,
                 int side_to_move);

    int evaluate_accumulators(const BigAccumulator& us_acc, const BigAccumulator& them_acc);

    const BigNetworkParameters& params() const { return net; }
    BigNetworkParameters& params_mut() { return net; }

    // HalfKAv2 Feature index helper
    static inline int halfka_index(int king_sq, int piece_type, int sq) {
        return king_sq * HALF_KA_PIECE_FEATURES + piece_type * 64 + sq;
    }

private:
    BigNetworkParameters net;
    bool network_loaded;
    std::string current_file;
};

extern NNUEEvaluation GlobalNNUE;

} // namespace nnue
} // namespace pointchess
