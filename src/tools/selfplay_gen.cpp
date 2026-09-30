// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "selfplay_gen.h"
#include "../glaurung/position.h"
#include "../glaurung/movegen.h"
#include "../glaurung/evaluate.h"
#include "../glaurung/search.h"
#include "../glaurung/misc.h"
#include "../eval/backend.h"
#include "../nnue/nnue_arch.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <chrono>

namespace pointchess {

struct GameSample {
    std::string fen;
    int16_t eval_cp;
};

static Move pick_best_search_move(Position& pos, int depth, int thread_id, int& out_score) {
    MoveStack mlist[256];
    int n = generate_legal_moves(pos, mlist);
    if (n == 0) return MOVE_NONE;
    if (n == 1) {
        out_score = 0;
        return mlist[0].move;
    }

    UndoInfo u;
    Move best_m = mlist[0].move;
    int best_val = -30000;
    EvalInfo ei;

    for (int i = 0; i < n; ++i) {
        Move m = mlist[i].move;
        pos.do_move(m, u);

        int val = 0;
        if (depth <= 1) {
            val = -value_to_centipawns(evaluate(pos, ei, thread_id));
        } else {
            // Shallow 2-ply evaluation
            MoveStack mlist2[256];
            int n2 = generate_legal_moves(pos, mlist2);
            if (n2 == 0) {
                val = pos.is_check() ? 25000 : 0;
            } else {
                int min_v = 30000;
                for (int j = 0; j < std::min(n2, 8); ++j) {
                    pos.do_move(mlist2[j].move, u);
                    int v2 = value_to_centipawns(evaluate(pos, ei, thread_id));
                    pos.undo_move(mlist2[j].move, u);
                    if (v2 < min_v) min_v = v2;
                }
                val = -min_v;
            }
        }

        pos.undo_move(m, u);

        if (val > best_val) {
            best_val = val;
            best_m = m;
        }
    }

    out_score = best_val;
    return best_m;
}

void run_selfplay_generator(int total_games, int num_threads, int search_depth, const std::string& output_file) {
    std::cout << "============================================================" << std::endl;
    std::cout << " PointChess Million-Game Self-Play Dataset Generator" << std::endl;
    std::cout << " Target Games: " << total_games << " | Threads: " << num_threads << " | Depth: " << search_depth << std::endl;
    std::cout << " Output File:  " << output_file << std::endl;
    std::cout << "============================================================" << std::endl;

    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return;
    }

    std::mutex file_mutex;
    std::atomic<int> games_completed(0);
    std::atomic<uint64_t> total_positions(0);
    auto start_time = std::chrono::steady_clock::now();

    int games_per_thread = (total_games + num_threads - 1) / num_threads;

    auto worker_func = [&](int thread_id, int games_to_run) {
        std::mt19937 rng(1337 + thread_id * 997);
        std::vector<std::string> thread_buffer;
        thread_buffer.reserve(10000);

        for (int g = 0; g < games_to_run; ++g) {
            if (games_completed.load(std::memory_order_relaxed) >= total_games) break;

            Position pos(StartPosition);
            UndoInfo u;
            std::vector<GameSample> game_records;
            game_records.reserve(100);

            // Random opening diversification (2 to 6 random legal plies)
            int random_plies = 2 + (rng() % 5);
            for (int r = 0; r < random_plies; ++r) {
                MoveStack mlist[256];
                int n = generate_legal_moves(pos, mlist);
                if (n == 0) break;
                Move rand_m = mlist[rng() % n].move;
                pos.do_move(rand_m, u);
            }

            // Play out the game
            float result = 0.5f; // Draw by default
            for (int ply = 0; ply < 120; ++ply) {
                MoveStack mlist[256];
                int n = generate_legal_moves(pos, mlist);
                if (n == 0) {
                    if (pos.is_check()) {
                        result = (pos.side_to_move() == WHITE) ? 0.0f : 1.0f;
                    } else {
                        result = 0.5f;
                    }
                    break;
                }

                if (pos.is_draw()) {
                    result = 0.5f;
                    break;
                }

                int score = 0;
                Move best_m = pick_best_search_move(pos, search_depth, thread_id % 8, score);
                if (best_m == MOVE_NONE) break;

                // Record position
                game_records.push_back({pos.to_fen(), static_cast<int16_t>(score)});

                // Adjudication on extreme advantage
                if (score >= 900) {
                    result = (pos.side_to_move() == WHITE) ? 1.0f : 0.0f;
                    break;
                } else if (score <= -900) {
                    result = (pos.side_to_move() == WHITE) ? 0.0f : 1.0f;
                    break;
                }

                pos.do_move(best_m, u);
            }

            // Buffer positions with final game outcome
            for (const auto& sample : game_records) {
                thread_buffer.push_back(sample.fen + " | " + std::to_string(sample.eval_cp) + " | " + std::to_string(result));
            }

            int curr_games = ++games_completed;
            total_positions += game_records.size();

            // Flush thread buffer periodically to reduce lock contention
            if (thread_buffer.size() >= 5000 || g == games_to_run - 1) {
                std::lock_guard<std::mutex> lock(file_mutex);
                for (const auto& line : thread_buffer) {
                    out << line << "\n";
                }
                thread_buffer.clear();
            }

            // Progress reporting
            if (curr_games % 1000 == 0 || curr_games == total_games) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double gps = curr_games / std::max(0.001, elapsed);
                std::cout << "Progress: [" << curr_games << "/" << total_games << " games] | "
                          << "Positions: " << total_positions.load() << " | "
                          << "Speed: " << static_cast<int>(gps) << " games/sec ("
                          << static_cast<int>(total_positions.load() / std::max(0.001, elapsed)) << " pos/sec) | "
                          << "Elapsed: " << static_cast<int>(elapsed) << "s" << std::endl;
            }
        }
    };

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back(worker_func, t, games_per_thread);
    }
    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    out.flush();
    out.close();

    auto total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "\n============================================================" << std::endl;
    std::cout << " Self-Play Complete!" << std::endl;
    std::cout << " Total Games: " << games_completed.load() << std::endl;
    std::cout << " Total Positions: " << total_positions.load() << std::endl;
    std::cout << " Time: " << total_time << "s (" << static_cast<int>(games_completed.load() / total_time) << " games/sec)" << std::endl;
    std::cout << " Saved to: " << output_file << std::endl;
    std::cout << "============================================================" << std::endl;
}

} // namespace pointchess
