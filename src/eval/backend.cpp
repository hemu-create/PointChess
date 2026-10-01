// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "backend.h"
#include "../nnue/nnue_arch.h"
#include <cstdlib>
#include <iostream>
#include <algorithm>

namespace pointchess {

EvaluationManager::EvaluationManager() {
    config.backend = BACKEND_CPU_AUTO;
    config.use_nnue = true;
    config.eval_file = "pointchess.pchess";
    config.gpu_device = 0;
    config.gpu_batch_size = 32;
    config.personality = PERSONALITY_DEFAULT;
    config.gpu_available = false;
}

void EvaluationManager::check_gpu_availability() {
    config.gpu_available = false;

    // 1. Universal Direct Rendering Infrastructure (AMD Radeon, Intel Arc / Xe, NVIDIA, etc.)
    FILE* f_dri = fopen("/dev/dri/renderD128", "r");
    if (!f_dri) f_dri = fopen("/dev/dri/card0", "r");
    if (f_dri) {
        config.gpu_available = true;
        fclose(f_dri);
        return;
    }

    // 2. AMD ROCm / HSA Kernel Fusion Driver
    FILE* f_rocm = fopen("/dev/kfd", "r");
    if (f_rocm) {
        config.gpu_available = true;
        fclose(f_rocm);
        return;
    }

    // 3. NVIDIA Device Nodes
    FILE* f_nv = fopen("/dev/nvidia0", "r");
    if (!f_nv) f_nv = fopen("/dev/nvidiactl", "r");
    if (f_nv) {
        config.gpu_available = true;
        fclose(f_nv);
        return;
    }

    // 4. Universal GPU Environment Variables (NVIDIA CUDA, AMD HIP/ROCm, Intel OneAPI, OpenCL)
    const char* env_vars[] = {
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "GPU_DEVICE_ORDINAL",
        "OCL_ICD_VENDORS"
    };
    for (const char* var : env_vars) {
        const char* val = getenv(var);
        if (val && std::string(val) != "-1" && std::string(val) != "") {
            config.gpu_available = true;
            return;
        }
    }

    // 5. Query system GPU toolchains
    #if !defined(_WIN32)
    if (system("which nvidia-smi >/dev/null 2>&1 || which rocm-smi >/dev/null 2>&1 || which clinfo >/dev/null 2>&1 || which vulkaninfo >/dev/null 2>&1") == 0) {
        config.gpu_available = true;
        return;
    }
    #endif

    // Always permit user to activate GPU mode
    config.gpu_available = true;
}

void EvaluationManager::init() {
    check_gpu_availability();
    // Attempt default net load
    nnue::GlobalNNUE.load_network_file(config.eval_file);
}

void EvaluationManager::set_backend(const std::string& backend_name) {
    std::string b = backend_name;
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);

    if (b == "gpu" || b == "cuda" || b == "rocm" || b == "metal" || b == "vulkan" || b == "direct3d") {
        config.backend = BACKEND_GPU_CUDA;
        std::cout << "info string PointChess backend set to GPU (Universal Hardware / GPU Acceleration enabled)" << std::endl;
    } else if (b == "opencl" || b == "gpu (opencl)") {
        config.backend = BACKEND_GPU_OPENCL;
        std::cout << "info string PointChess backend set to GPU (OpenCL Acceleration)" << std::endl;
    } else if (b == "classical" || b == "cpu classical") {
        config.backend = BACKEND_CPU_CLASSICAL;
        config.use_nnue = false;
        std::cout << "info string PointChess backend set to CPU Classical Evaluation" << std::endl;
    } else {
        config.backend = BACKEND_CPU_NNUE;
        config.use_nnue = true;
        std::cout << "info string PointChess backend set to CPU NNUE" << std::endl;
    }
}

void EvaluationManager::set_personality(const std::string& personality_name) {
    std::string p = personality_name;
    std::transform(p.begin(), p.end(), p.begin(), ::tolower);

    if (p == "aggressive") {
        config.personality = PERSONALITY_AGGRESSIVE;
        std::cout << "info string Personality: Aggressive (High king attack, active piece play, attacking initiative)" << std::endl;
    } else if (p == "solid") {
        config.personality = PERSONALITY_SOLID;
        std::cout << "info string Personality: Solid (Solid pawn structures, king shelter, conservative risk)" << std::endl;
    } else if (p == "positional") {
        config.personality = PERSONALITY_POSITIONAL;
        std::cout << "info string Personality: Positional (Space advantage, bishop pair, central control, outposts)" << std::endl;
    } else if (p == "tactical") {
        config.personality = PERSONALITY_TACTICAL;
        std::cout << "info string Personality: Tactical (Sharp dynamic play, check pressure, tactical extensions)" << std::endl;
    } else if (p == "gambiteer") {
        config.personality = PERSONALITY_GAMBITEER;
        std::cout << "info string Personality: Gambiteer (Dynamic sacrifices for rapid piece development and initiative)" << std::endl;
    } else if (p == "dynamic") {
        config.personality = PERSONALITY_DYNAMIC;
        std::cout << "info string Personality: Dynamic (Unbalanced play, dynamic compensation, piece activity)" << std::endl;
    } else {
        config.personality = PERSONALITY_DEFAULT;
        std::cout << "info string Personality: Default (Universal balanced master playstyle)" << std::endl;
    }
}

void EvaluationManager::set_eval_file(const std::string& path) {
    config.eval_file = path;
    bool ok = nnue::GlobalNNUE.load_network_file(path);
    if (ok) {
        std::cout << "info string Successfully loaded NNUE file: " << path << std::endl;
    } else {
        std::cout << "info string Notice: Could not open " << path << ", using calibrated default weights." << std::endl;
    }
}

void EvaluationManager::set_use_nnue(bool enable) {
    config.use_nnue = enable;
    if (enable && config.backend == BACKEND_CPU_CLASSICAL) {
        config.backend = BACKEND_CPU_NNUE;
    }
}

void EvaluationManager::set_gpu_device(int device_id) {
    config.gpu_device = device_id;
}

int EvaluationManager::get_aggressiveness_mult() const {
    switch (config.personality) {
        case PERSONALITY_AGGRESSIVE: return 160;
        case PERSONALITY_GAMBITEER:  return 190;
        case PERSONALITY_TACTICAL:   return 140;
        case PERSONALITY_DYNAMIC:    return 130;
        case PERSONALITY_SOLID:      return 70;
        case PERSONALITY_POSITIONAL: return 90;
        default: return 100;
    }
}

int EvaluationManager::get_cowardice_mult() const {
    switch (config.personality) {
        case PERSONALITY_AGGRESSIVE: return 60;
        case PERSONALITY_GAMBITEER:  return 40;
        case PERSONALITY_SOLID:      return 150;
        case PERSONALITY_POSITIONAL: return 120;
        case PERSONALITY_TACTICAL:   return 70;
        default: return 100;
    }
}

int EvaluationManager::get_mobility_mult() const {
    switch (config.personality) {
        case PERSONALITY_AGGRESSIVE: return 130;
        case PERSONALITY_TACTICAL:   return 145;
        case PERSONALITY_DYNAMIC:    return 135;
        case PERSONALITY_GAMBITEER:  return 140;
        default: return 100;
    }
}

int EvaluationManager::get_space_mult() const {
    switch (config.personality) {
        case PERSONALITY_POSITIONAL: return 145;
        case PERSONALITY_SOLID:      return 115;
        case PERSONALITY_AGGRESSIVE: return 120;
        default: return 100;
    }
}

int EvaluationManager::get_king_safety_mult() const {
    switch (config.personality) {
        case PERSONALITY_AGGRESSIVE: return 150;
        case PERSONALITY_TACTICAL:   return 140;
        case PERSONALITY_GAMBITEER:  return 160;
        case PERSONALITY_SOLID:      return 130;
        default: return 100;
    }
}

int EvaluationManager::get_pawn_structure_mult() const {
    switch (config.personality) {
        case PERSONALITY_POSITIONAL: return 140;
        case PERSONALITY_SOLID:      return 135;
        case PERSONALITY_GAMBITEER:  return 70;
        default: return 100;
    }
}

} // namespace pointchess
