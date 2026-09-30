// PointChess - a chess engine in the Glaurung tradition.
// SPDX-License-Identifier: GPL-3.0-or-later
// High-Speed Multi-Threaded Self-Play Generator (Supports 200k - 1M+ Games)

#pragma once

#include <string>

namespace pointchess {

void run_selfplay_generator(int total_games, int num_threads, int search_depth, const std::string& output_file);

} // namespace pointchess
