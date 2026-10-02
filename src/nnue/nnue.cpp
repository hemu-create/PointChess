// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// 11.5-Million Parameter Modern HalfKAv2 NNUE Inference Engine

#include "nnue_arch.h"
#include "pointbeta.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pointchess {
namespace nnue {

NNUEEvaluation GlobalNNUE;
MegaNNUE1024 GlobalMega1024;
int ActiveFTSize = 256;

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

// Honest AVX2 dot: int32[] . int8[] -> int64 sum. Scalar tail for remainder.
inline int64_t dot_i32_i8_avx2(const int32_t* a, const int8_t* b, int n) {
#if defined(__AVX2__)
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b16 = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i b0 = _mm256_cvtepi8_epi32(b16);
        __m256i b1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b16, 8));
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(a + i + 8));
        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(a0, b0));
        acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(a1, b1));
    }
    acc0 = _mm256_add_epi32(acc0, acc1);
    __m128i lo = _mm256_castsi256_si128(acc0), hi = _mm256_extracti128_si256(acc0, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    int64_t sum = _mm_cvtsi128_si32(s);
    for (; i < n; ++i) sum += (int64_t)a[i] * b[i];
    return sum;
#else
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += (int64_t)a[i] * b[i];
    return sum;
#endif
}

int NNUEEvaluation::evaluate_accumulators(const BigAccumulator& us_acc, const BigAccumulator& them_acc) {
    // 1. SCReL Activation on 11.5M Accumulator outputs: 512 inputs (256 us + 256 them)
    alignas(64) int32_t l0_out[BIG_ACCUMULATOR_SIZE * 2];
#if defined(__AVX2__)
    __m256i zero = _mm256_setzero_si256();
    __m256i max127 = _mm256_set1_epi16(127);
    auto process_half = [&](const int16_t* in_vals, int32_t* out_vals) {
        for (int i = 0; i < BIG_ACCUMULATOR_SIZE; i += 16) {
            __m256i v = _mm256_load_si256((const __m256i*)(in_vals + i));
            __m256i clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), max127);
            
            // Lower 8 int16 -> int32
            __m128i lo16 = _mm256_castsi256_si128(clamped);
            __m256i lo32 = _mm256_cvtepi16_epi32(lo16);
            __m256i sq_lo = _mm256_srli_epi32(_mm256_mullo_epi32(lo32, lo32), 7);
            _mm256_storeu_si256((__m256i*)(out_vals + i), sq_lo);
            
            // Upper 8 int16 -> int32
            __m128i hi16 = _mm256_extracti128_si256(clamped, 1);
            __m256i hi32 = _mm256_cvtepi16_epi32(hi16);
            __m256i sq_hi = _mm256_srli_epi32(_mm256_mullo_epi32(hi32, hi32), 7);
            _mm256_storeu_si256((__m256i*)(out_vals + i + 8), sq_hi);
        }
    };
    process_half(us_acc.values, l0_out);
    process_half(them_acc.values, l0_out + BIG_ACCUMULATOR_SIZE);
#else
    for (int i = 0; i < BIG_ACCUMULATOR_SIZE; ++i) {
        l0_out[i] = screl(us_acc.values[i]);
        l0_out[BIG_ACCUMULATOR_SIZE + i] = screl(them_acc.values[i]);
    }
#endif

    // 2. Linear Layer 1: 512 -> 32 (honest AVX2 dot)
    alignas(32) int32_t l1_out[BIG_L1_SIZE];
    for (int o = 0; o < BIG_L1_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l0_out, net.l1_weights + o * (BIG_ACCUMULATOR_SIZE * 2),
                                      BIG_ACCUMULATOR_SIZE * 2);
        int32_t sum = int32_t(net.l1_biases[o] + dot);
        l1_out[o] = screl_i32(sum / 64);
    }

    // 3. Linear Layer 2: 32 -> 32 (AVX2 dot)
    alignas(32) int32_t l2_out[BIG_L2_SIZE];
    for (int o = 0; o < BIG_L2_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l1_out, net.l2_weights + o * BIG_L1_SIZE, BIG_L1_SIZE);
        int32_t sum = int32_t(net.l2_biases[o] + dot);
        l2_out[o] = screl_i32(sum / 64);
    }

    // 4. Output Layer: 32 -> 1 (AVX2 dot)
    int64_t out_dot = dot_i32_i8_avx2(l2_out, net.out_weights, BIG_L2_SIZE);
    int32_t sum = int32_t(net.out_bias + out_dot);

    int score = int(sum / 16);
    return std::clamp(score, -1500, 1500);
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

bool MegaNNUE1024::load(std::ifstream& file) {
    net.feature_weights.resize(size_t(HALF_KA_FEATURES) * MEGA1024_FT);
    net.feature_biases.resize(MEGA1024_FT);
    net.l1_weights.resize(size_t(MEGA1024_FT) * 2 * BIG_L1_SIZE);
    net.l1_biases.resize(BIG_L1_SIZE);
    net.l2_weights.resize(size_t(BIG_L1_SIZE) * BIG_L2_SIZE);
    net.l2_biases.resize(BIG_L2_SIZE);
    net.out_weights.resize(BIG_L2_SIZE);
    file.read(reinterpret_cast<char*>(net.feature_weights.data()),
              net.feature_weights.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.feature_biases.data()),
              net.feature_biases.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.l1_weights.data()), net.l1_weights.size());
    file.read(reinterpret_cast<char*>(net.l1_biases.data()),
              net.l1_biases.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(net.l2_weights.data()), net.l2_weights.size());
    file.read(reinterpret_cast<char*>(net.l2_biases.data()),
              net.l2_biases.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(net.out_weights.data()), net.out_weights.size());
    file.read(reinterpret_cast<char*>(&net.out_bias), sizeof(net.out_bias));
    if (!file.good() && !file.eof()) return false;
    loaded = true;
    return true;
}

int MegaNNUE1024::evaluate(const int* wf, int nw, const int* bf, int nb, int stm) {
    constexpr int FT = MEGA1024_FT;
    std::vector<int16_t> us(FT), them(FT);
    std::memcpy(us.data(), net.feature_biases.data(), FT * sizeof(int16_t));
    them = us;
    auto gather = [&](const int* feats, int n, std::vector<int16_t>& acc) {
        for (int i = 0; i < n; ++i) {
            const int16_t* w = net.feature_weights.data() + size_t(feats[i]) * FT;
            for (int j = 0; j < FT; ++j) acc[j] += w[j];
        }
    };
    if (stm == 0) { gather(wf, nw, us); gather(bf, nb, them); }
    else { gather(bf, nb, us); gather(wf, nw, them); }
    alignas(64) static thread_local int32_t l0[2048];
    for (int i = 0; i < FT; ++i) {
        int32_t a = us[i] < 0 ? 0 : (us[i] > 127 ? 127 : us[i]);
        int32_t b = them[i] < 0 ? 0 : (them[i] > 127 ? 127 : them[i]);
        l0[i] = (a * a) / 128;
        l0[FT + i] = (b * b) / 128;
    }
    alignas(32) int32_t l1[BIG_L1_SIZE], l2[BIG_L2_SIZE];
    for (int o = 0; o < BIG_L1_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l0, net.l1_weights.data() + size_t(o) * FT * 2, FT * 2);
        int32_t sum = int32_t(net.l1_biases[o] / 2 + dot);
        l1[o] = screl_i32(sum / 128);
    }
    for (int o = 0; o < BIG_L2_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l1, net.l2_weights.data() + size_t(o) * BIG_L1_SIZE, BIG_L1_SIZE);
        int32_t sum = int32_t(net.l2_biases[o] / 2 + dot);
        l2[o] = screl_i32(sum / 64);
    }
    int64_t out_dot = dot_i32_i8_avx2(l2, net.out_weights.data(), BIG_L2_SIZE);
    int32_t sum = int32_t(net.out_bias + out_dot);
    int score = sum / 128;
    return std::clamp(score, -1500, 1500);
}

bool NNUEEvaluation::load_pchess_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    char magic[32] = {0};
    file.read(magic, 24);
    std::string ms(magic, 24);
    if (ms.find("POINTBETA") != std::string::npos) {
        file.close();
        if (pointbeta::GlobalPointBeta.load_file(filepath)) {
            ActiveFTSize = 4096;
            network_loaded = true;
            current_file = filepath;
            return true;
        }
        return false;
    }
    if (ms.find("POINTCHESS_V3") != std::string::npos) {
        int32_t ft = 0;
        file.read(reinterpret_cast<char*>(&ft), sizeof(ft));
        if (ft == MEGA1024_FT) {
            if (GlobalMega1024.load(file)) {
                ActiveFTSize = 1024;
                network_loaded = true;
                current_file = filepath;
                return true;
            }
            return false;
        }
        return false; // unsupported v3 width
    }
    ActiveFTSize = 256;
    if (ms.find("POINTCHESS") == std::string::npos) {
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
