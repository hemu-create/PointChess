// PointBeta NNUE: Factorized Quad-Accumulator Architecture (4096 L0 Dimension)
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pointchess {
namespace pointbeta {

constexpr int NUM_KING_FEATURES   = 64 * 9 * 64;   // 36,864
constexpr int NUM_PAWN_FEATURES   = 384;           // 384
constexpr int NUM_PAIR_FEATURES   = 464;           // 464
constexpr int NUM_SAFETY_FEATURES = 64 * 8 * 25;   // 12,800

constexpr int FT_KING_SIZE   = 1024;
constexpr int FT_PAWN_SIZE   = 512;
constexpr int FT_PAIR_SIZE   = 256;
constexpr int FT_SAFETY_SIZE = 256;

constexpr int L0_HALF_SIZE   = FT_KING_SIZE + FT_PAWN_SIZE + FT_PAIR_SIZE + FT_SAFETY_SIZE; // 2048
constexpr int L0_TOTAL_SIZE  = L0_HALF_SIZE * 2;                                            // 4096

constexpr int L1_SIZE = 1024;
constexpr int L2_SIZE = 256;
constexpr int L3_SIZE = 64;

constexpr int WEIGHT_SCALE_L0  = 256;
constexpr int WEIGHT_SCALE_L1  = 64;
constexpr int WEIGHT_SCALE_L2  = 64;
constexpr int WEIGHT_SCALE_L3  = 64;
constexpr int WEIGHT_SCALE_OUT = 16;

struct alignas(64) PointBetaWeights {
    std::vector<int16_t> w_king;    // 36864 * 1024
    std::vector<int16_t> b_king;    // 1024
    std::vector<int16_t> w_pawn;    // 384 * 512
    std::vector<int16_t> b_pawn;    // 512
    std::vector<int16_t> w_pair;    // 464 * 256
    std::vector<int16_t> b_pair;    // 256
    std::vector<int16_t> w_safety;  // 12800 * 256
    std::vector<int16_t> b_safety;  // 256

    std::vector<int8_t>  w_l1;      // 4096 * 1024
    std::vector<int32_t> b_l1;      // 1024
    std::vector<int8_t>  w_l2;      // 1024 * 256
    std::vector<int32_t> b_l2;      // 256
    std::vector<int8_t>  w_l3;      // 256 * 64
    std::vector<int32_t> b_l3;      // 64
    std::vector<int8_t>  w_out;     // 64
    int32_t b_out = 0;
};

class PointBetaEvaluation {
public:
    PointBetaEvaluation();
    bool load_file(const std::string& filepath);
    bool is_loaded() const { return loaded; }

    int evaluate(const int* wk, int nwk, const int* bk, int nbk,
                 const int* wp, int nwp, const int* bp, int nbp,
                 const int* wr, int nwr, const int* br, int nbr,
                 const int* ws, int nws, const int* bs, int nbs,
                 int stm);

private:
    PointBetaWeights net;
    bool loaded = false;
    std::string current_file;
};

extern PointBetaEvaluation GlobalPointBeta;

} // namespace pointbeta
} // namespace pointchess
