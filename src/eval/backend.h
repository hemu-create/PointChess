// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace pointchess {

enum BackendType {
    BACKEND_CPU_AUTO = 0,
    BACKEND_CPU_NNUE = 1,
    BACKEND_CPU_CLASSICAL = 2,
    BACKEND_GPU_CUDA = 3,
    BACKEND_GPU_OPENCL = 4
};

enum PersonalityType {
    PERSONALITY_DEFAULT = 0,
    PERSONALITY_AGGRESSIVE = 1,
    PERSONALITY_SOLID = 2,
    PERSONALITY_POSITIONAL = 3,
    PERSONALITY_TACTICAL = 4,
    PERSONALITY_GAMBITEER = 5,
    PERSONALITY_DYNAMIC = 6
};

struct BackendConfig {
    BackendType backend = BACKEND_CPU_AUTO;
    bool use_nnue = true;
    std::string eval_file = "pointchess.pchess";
    int gpu_device = 0;
    int gpu_batch_size = 32;
    PersonalityType personality = PERSONALITY_DEFAULT;
    bool gpu_available = false;
};

class EvaluationManager {
public:
    static EvaluationManager& instance() {
        static EvaluationManager mgr;
        return mgr;
    }

    void init();
    void set_backend(const std::string& backend_name);
    void set_personality(const std::string& personality_name);
    void set_eval_file(const std::string& path);
    void set_use_nnue(bool enable);
    void set_gpu_device(int device_id);

    BackendType active_backend() const { return config.backend; }
    PersonalityType active_personality() const { return config.personality; }
    const BackendConfig& get_config() const { return config; }
    bool is_gpu_active() const { return config.backend == BACKEND_GPU_CUDA || config.backend == BACKEND_GPU_OPENCL; }

    // Personality multiplier helpers
    int get_aggressiveness_mult() const;
    int get_cowardice_mult() const;
    int get_mobility_mult() const;
    int get_space_mult() const;
    int get_king_safety_mult() const;
    int get_pawn_structure_mult() const;

private:
    EvaluationManager();
    BackendConfig config;
    void check_gpu_availability();
    void apply_personality_weights();
};

} // namespace pointchess
