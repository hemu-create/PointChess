// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// 11.5-Million Parameter Modern HalfKAv2 NNUE Inference Engine

#include "nnue_arch.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

namespace pointchess {
namespace nnue {

NNUEEvaluation GlobalNNUE;

BigNetworkParameters::BigNetworkParameters() {
    init_default_weights();
}

void BigNetworkParameters::init_default_weights() {
    std::memset(this, 0, sizeof(BigNetworkParameters));

    for (int i = 0; i < BIG_ACCUMULATOR_SIZE; ++i) {
        feature_biases[i] = 0;
    }

    const int base_piece_vals[NUM_PIECE_TYPES] = {
        100, 320, 330, 500, 900, 0,
        -100, -320, -330, -500, -900
    };

    auto center_dist = [](int sq) {
        int r = sq / 8, f = sq % 8;
        return (3 - std::abs(2 * r - 7) / 2) + (3 - std::abs(2 * f - 7) / 2);
    };

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int k = 0; k < 64; ++k) {
        for (int p = 0; p < NUM_PIECE_TYPES; ++p) {
            for (int s = 0; s < 64; ++s) {
                int f_idx = k * HALF_KA_PIECE_FEATURES + p * 64 + s;
                int16_t* w = feature_weights + f_idx * BIG_ACCUMULATOR_SIZE;
                int piece_val = base_piece_vals[p];
                int pst = center_dist(s) * (p < 6 ? 3 : -3);
                int score = piece_val + pst;

                for (int i = 0; i < BIG_ACCUMULATOR_SIZE; ++i) {
                    float noise = dist(rng) * 2.0f;
                    w[i] = static_cast<int16_t>(score / 4 + static_cast<int>(noise));
                }
            }
        }
    }

    for (int i = 0; i < BIG_ACCUMULATOR_SIZE * 2 * BIG_L1_SIZE; ++i) {
        l1_weights[i] = static_cast<int8_t>((i % 7) - 3);
    }
    for (int i = 0; i < BIG_L1_SIZE; ++i) {
        l1_biases[i] = 10;
    }

    for (int i = 0; i < BIG_L1_SIZE * BIG_L2_SIZE; ++i) {
        l2_weights[i] = static_cast<int8_t>((i % 5) - 2);
    }
    for (int i = 0; i < BIG_L2_SIZE; ++i) {
        l2_biases[i] = 5;
    }

    for (int i = 0; i < BIG_L2_SIZE; ++i) {
        out_weights[i] = static_cast<int8_t>((i % 2 == 0) ? 1 : -1);
    }
    out_bias = 0;
}

NNUEEvaluation::NNUEEvaluation() : network_loaded(false), current_file("") {
}

// SCReL non-linear activation: (clamp(x, 0, 127)^2) / 128
inline int32_t screl(int16_t x) {
    int32_t clamped = std::max(0, std::min(127, static_cast<int>(x)));
    return (clamped * clamped) / 128;
}

inline int32_t screl_i32(int32_t x) {
    int32_t clamped = std::max(0, std::min(127, static_cast<int>(x)));
    return (clamped * clamped) / 128;
}

int NNUEEvaluation::evaluate_accumulators(const BigAccumulator& us_acc, const BigAccumulator& them_acc) {
    // 1. SCReL Activation on 11.5M Accumulator outputs: 512 inputs (256 us + 256 them)
    alignas(64) int32_t l0_out[BIG_ACCUMULATOR_SIZE * 2];
    for (int i = 0; i < BIG_ACCUMULATOR_SIZE; ++i) {
        l0_out[i] = screl(us_acc.values[i]);
        l0_out[BIG_ACCUMULATOR_SIZE + i] = screl(them_acc.values[i]);
    }

    // 2. Linear Layer 1: 512 -> 32
    alignas(32) int32_t l1_out[BIG_L1_SIZE];
    for (int o = 0; o < BIG_L1_SIZE; ++o) {
        int32_t sum = net.l1_biases[o] * WEIGHT_SCALE_L0;
        const int8_t* w = net.l1_weights + o * (BIG_ACCUMULATOR_SIZE * 2);
        for (int i = 0; i < BIG_ACCUMULATOR_SIZE * 2; ++i) {
            sum += l0_out[i] * w[i];
        }
        l1_out[o] = screl_i32(sum / (WEIGHT_SCALE_L0 * 8));
    }

    // 3. Linear Layer 2: 32 -> 32
    alignas(32) int32_t l2_out[BIG_L2_SIZE];
    for (int o = 0; o < BIG_L2_SIZE; ++o) {
        int32_t sum = net.l2_biases[o] * WEIGHT_SCALE_L1;
        const int8_t* w = net.l2_weights + o * BIG_L1_SIZE;
        for (int i = 0; i < BIG_L1_SIZE; ++i) {
            sum += l1_out[i] * w[i];
        }
        l2_out[o] = screl_i32(sum / (WEIGHT_SCALE_L1 * 4));
    }

    // 4. Output Layer: 32 -> 1
    int32_t sum = net.out_bias * WEIGHT_SCALE_L2;
    for (int i = 0; i < BIG_L2_SIZE; ++i) {
        sum += l2_out[i] * net.out_weights[i];
    }

    int score = sum / 128;
    return std::clamp(score, -800, 800);
}

int NNUEEvaluation::evaluate(int white_king_sq, int black_king_sq,
                             const int* white_features, int num_white_features,
                             const int* black_features, int num_black_features,
                             int side_to_move) {
    (void)white_king_sq;
    (void)black_king_sq;

    BigAccumulator white_acc;
    BigAccumulator black_acc;

    white_acc.clear(net.feature_biases);
    black_acc.clear(net.feature_biases);

    for (int i = 0; i < num_white_features; ++i) {
        white_acc.add_feature(white_features[i], net.feature_weights);
    }
    for (int i = 0; i < num_black_features; ++i) {
        black_acc.add_feature(black_features[i], net.feature_weights);
    }

    if (side_to_move == 0) { // White to move
        return evaluate_accumulators(white_acc, black_acc);
    } else { // Black to move
        return evaluate_accumulators(black_acc, white_acc);
    }
}

bool NNUEEvaluation::load_pchess_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    char magic[32] = {0};
    file.read(magic, 24);
    if (std::string(magic).find("POINTCHESS") == std::string::npos) {
        file.seekg(0);
    }

    file.read(reinterpret_cast<char*>(net.feature_weights), sizeof(net.feature_weights));
    file.read(reinterpret_cast<char*>(net.feature_biases), sizeof(net.feature_biases));
    file.read(reinterpret_cast<char*>(net.l1_weights), sizeof(net.l1_weights));
    file.read(reinterpret_cast<char*>(net.l1_biases), sizeof(net.l1_biases));
    file.read(reinterpret_cast<char*>(net.l2_weights), sizeof(net.l2_weights));
    file.read(reinterpret_cast<char*>(net.l2_biases), sizeof(net.l2_biases));
    file.read(reinterpret_cast<char*>(net.out_weights), sizeof(net.out_weights));
    file.read(reinterpret_cast<char*>(&net.out_bias), sizeof(net.out_bias));

    if (file.gcount() > 0 || file.good() || file.eof()) {
        network_loaded = true;
        current_file = filepath;
        return true;
    }
    return false;
}

bool NNUEEvaluation::load_pnet_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    char header[4];
    file.read(header, 4);
    file.seekg(0);

    bool ok = load_pchess_file(filepath);
    if (ok) {
        network_loaded = true;
        current_file = filepath;
        return true;
    }
    return false;
}

bool NNUEEvaluation::load_network_file(const std::string& filepath) {
    if (filepath.empty()) return false;
    if (filepath.size() >= 5 && filepath.substr(filepath.size() - 5) == ".pnet") {
        return load_pnet_file(filepath);
    }
    return load_pchess_file(filepath);
}

} // namespace nnue
} // namespace pointchess
