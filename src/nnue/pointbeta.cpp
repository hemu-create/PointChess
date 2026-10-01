// PointBeta NNUE: Factorized Quad-Accumulator Architecture Implementation
// SPDX-License-Identifier: GPL-3.0-or-later

#include "pointbeta.h"
#include <fstream>
#include <iostream>

namespace pointchess {
namespace pointbeta {

PointBetaEvaluation GlobalPointBeta;

PointBetaEvaluation::PointBetaEvaluation() : loaded(false), current_file("") {
}

static inline int32_t screl_i32(int32_t x) {
    int32_t clamped = std::max(0, std::min(127, static_cast<int>(x)));
    return (clamped * clamped) / 128;
}

static inline int64_t dot_i32_i8_avx2(const int32_t* a, const int8_t* b, int n) {
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

bool PointBetaEvaluation::load_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    char magic[32] = {0};
    file.read(magic, 24);
    std::string ms(magic, 24);
    if (ms.find("POINTBETA_V1") == std::string::npos) {
        return false;
    }

    net.w_king.resize(size_t(NUM_KING_FEATURES) * FT_KING_SIZE);
    net.b_king.resize(FT_KING_SIZE);
    net.w_pawn.resize(size_t(NUM_PAWN_FEATURES) * FT_PAWN_SIZE);
    net.b_pawn.resize(FT_PAWN_SIZE);
    net.w_pair.resize(size_t(NUM_PAIR_FEATURES) * FT_PAIR_SIZE);
    net.b_pair.resize(FT_PAIR_SIZE);
    net.w_safety.resize(size_t(NUM_SAFETY_FEATURES) * FT_SAFETY_SIZE);
    net.b_safety.resize(FT_SAFETY_SIZE);

    net.w_l1.resize(size_t(L0_TOTAL_SIZE) * L1_SIZE);
    net.b_l1.resize(L1_SIZE);
    net.w_l2.resize(size_t(L1_SIZE) * L2_SIZE);
    net.b_l2.resize(L2_SIZE);
    net.w_l3.resize(size_t(L2_SIZE) * L3_SIZE);
    net.b_l3.resize(L3_SIZE);
    net.w_out.resize(L3_SIZE);

    file.read(reinterpret_cast<char*>(net.w_king.data()), net.w_king.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.b_king.data()), net.b_king.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.w_pawn.data()), net.w_pawn.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.b_pawn.data()), net.b_pawn.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.w_pair.data()), net.w_pair.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.b_pair.data()), net.b_pair.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.w_safety.data()), net.w_safety.size() * sizeof(int16_t));
    file.read(reinterpret_cast<char*>(net.b_safety.data()), net.b_safety.size() * sizeof(int16_t));

    file.read(reinterpret_cast<char*>(net.w_l1.data()), net.w_l1.size());
    file.read(reinterpret_cast<char*>(net.b_l1.data()), net.b_l1.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(net.w_l2.data()), net.w_l2.size());
    file.read(reinterpret_cast<char*>(net.b_l2.data()), net.b_l2.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(net.w_l3.data()), net.w_l3.size());
    file.read(reinterpret_cast<char*>(net.b_l3.data()), net.b_l3.size() * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(net.w_out.data()), net.w_out.size());
    file.read(reinterpret_cast<char*>(&net.b_out), sizeof(net.b_out));

    if (file.good() || file.eof()) {
        loaded = true;
        current_file = filepath;
        return true;
    }
    return false;
}

int PointBetaEvaluation::evaluate(const int* wk, int nwk, const int* bk, int nbk,
                                 const int* wp, int nwp, const int* bp, int nbp,
                                 const int* wr, int nwr, const int* br, int nbr,
                                 const int* ws, int nws, const int* bs, int nbs,
                                 int stm) {
    if (!loaded) return 0;

    // Accumulators for White perspective
    alignas(64) static thread_local int32_t w_rep[L0_HALF_SIZE];
    alignas(64) static thread_local int32_t b_rep[L0_HALF_SIZE];

    auto accumulate = [](const int* feats, int count, int ft_dim,
                         const int16_t* weights, const int16_t* biases,
                         int32_t* out_act) {
        alignas(64) int16_t acc[FT_KING_SIZE];
        std::memcpy(acc, biases, ft_dim * sizeof(int16_t));
        for (int i = 0; i < count; ++i) {
            const int16_t* w = weights + size_t(feats[i]) * ft_dim;
            for (int j = 0; j < ft_dim; ++j) acc[j] += w[j];
        }
        for (int j = 0; j < ft_dim; ++j) {
            int32_t c = std::max(0, std::min(127, static_cast<int>(acc[j])));
            out_act[j] = (c * c) / 128;
        }
    };

    // White Perspective
    accumulate(wk, nwk, FT_KING_SIZE,   net.w_king.data(),   net.b_king.data(),   w_rep);
    accumulate(wp, nwp, FT_PAWN_SIZE,   net.w_pawn.data(),   net.b_pawn.data(),   w_rep + FT_KING_SIZE);
    accumulate(wr, nwr, FT_PAIR_SIZE,   net.w_pair.data(),   net.b_pair.data(),   w_rep + FT_KING_SIZE + FT_PAWN_SIZE);
    accumulate(ws, nws, FT_SAFETY_SIZE, net.w_safety.data(), net.b_safety.data(), w_rep + FT_KING_SIZE + FT_PAWN_SIZE + FT_PAIR_SIZE);

    // Black Perspective
    accumulate(bk, nbk, FT_KING_SIZE,   net.w_king.data(),   net.b_king.data(),   b_rep);
    accumulate(bp, nbp, FT_PAWN_SIZE,   net.w_pawn.data(),   net.b_pawn.data(),   b_rep + FT_KING_SIZE);
    accumulate(br, nbr, FT_PAIR_SIZE,   net.w_pair.data(),   net.b_pair.data(),   b_rep + FT_KING_SIZE + FT_PAWN_SIZE);
    accumulate(bs, nbs, FT_SAFETY_SIZE, net.w_safety.data(), net.b_safety.data(), b_rep + FT_KING_SIZE + FT_PAWN_SIZE + FT_PAIR_SIZE);

    // Combine into 4096-dim L0
    alignas(64) static thread_local int32_t l0[L0_TOTAL_SIZE];
    if (stm == 0) { // White to move
        std::memcpy(l0, w_rep, L0_HALF_SIZE * sizeof(int32_t));
        std::memcpy(l0 + L0_HALF_SIZE, b_rep, L0_HALF_SIZE * sizeof(int32_t));
    } else {        // Black to move
        std::memcpy(l0, b_rep, L0_HALF_SIZE * sizeof(int32_t));
        std::memcpy(l0 + L0_HALF_SIZE, w_rep, L0_HALF_SIZE * sizeof(int32_t));
    }

    // Layer 1: 4096 -> 1024
    alignas(32) static thread_local int32_t l1[L1_SIZE];
    for (int o = 0; o < L1_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l0, net.w_l1.data() + size_t(o) * L0_TOTAL_SIZE, L0_TOTAL_SIZE);
        l1[o] = screl_i32(int32_t(net.b_l1[o] * WEIGHT_SCALE_L0 + dot) / (WEIGHT_SCALE_L0 * 8));
    }

    // Layer 2: 1024 -> 256
    alignas(32) static thread_local int32_t l2[L2_SIZE];
    for (int o = 0; o < L2_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l1, net.w_l2.data() + size_t(o) * L1_SIZE, L1_SIZE);
        l2[o] = screl_i32(int32_t(net.b_l2[o] * WEIGHT_SCALE_L1 + dot) / (WEIGHT_SCALE_L1 * 4));
    }

    // Layer 3: 256 -> 64
    alignas(32) static thread_local int32_t l3[L3_SIZE];
    for (int o = 0; o < L3_SIZE; ++o) {
        int64_t dot = dot_i32_i8_avx2(l2, net.w_l3.data() + size_t(o) * L2_SIZE, L2_SIZE);
        l3[o] = screl_i32(int32_t(net.b_l3[o] * WEIGHT_SCALE_L2 + dot) / (WEIGHT_SCALE_L2 * 4));
    }

    // Output Layer: 64 -> 1
    int64_t out_dot = dot_i32_i8_avx2(l3, net.w_out.data(), L3_SIZE);
    int32_t sum = int32_t(net.b_out * WEIGHT_SCALE_L3 + out_dot);
    return std::clamp(sum / 256, -1500, 1500);
}

} // namespace pointbeta
} // namespace pointchess
