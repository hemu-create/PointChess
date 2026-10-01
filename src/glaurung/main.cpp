/*
  Glaurung, a UCI chess playing engine.
  Copyright (C) 2004-2008 Tord Romstad

  Glaurung is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  Glaurung is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


////
//// Includes
////

#include <cstdlib>
#include <iostream>

#include "benchmark.h"
#include "bitboard.h"
#include "direction.h"
#include "endgame.h"
#include "evaluate.h"
#include "material.h"
#include "mersenne.h"
#include "misc.h"
#include "movepick.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "uci.h"
#include "ucioption.h"
#include "../eval/backend.h"
#include "../tools/selfplay_gen.h"


//// 
//// Functions
////

int main(int argc, char *argv[]) {

  // Disable IO buffering
  setbuf(stdin, NULL);
  setbuf(stdout, NULL);
  std::cout.rdbuf()->pubsetbuf(NULL, 0);
  std::cin.rdbuf()->pubsetbuf(NULL, 0);

  // Initialization

  init_mersenne();
  init_direction_table();
  init_bitboards();
  init_uci_options();
  Position::init_zobrist();
  Position::init_piece_square_tables();
  MaterialInfo::init();
  MovePicker::init_phase_table();
  init_search_tables();
  init_eval(THREAD_MAX);
  pointchess::EvaluationManager::instance().init();
  init_bitbases();
  init_threads();

  // Make random number generation less deterministic, for book moves
  int i = abs(get_system_time() % 10000);
  for(int j = 0; j < i; j++)
    genrand_int32();

  // Process command line arguments
  if(argc >= 2) {
    if(std::string(argv[1]) == "bench") {
      if(argc != 4) {
        std::cout << "Usage: pointchess bench <hash> <threads>" << std::endl;
        exit(0);
      }
      benchmark(std::string(argv[2]), std::string(argv[3]));
      return 0;
    }
    else if(std::string(argv[1]) == "genselfplay") {
      int games = (argc >= 3) ? atoi(argv[2]) : 200000;
      int threads = (argc >= 4) ? atoi(argv[3]) : 12;
      int depth = (argc >= 5) ? atoi(argv[4]) : 2;
      std::string out_file = (argc >= 6) ? std::string(argv[5]) : "selfplay_dataset.txt";
      pointchess::run_selfplay_generator(games, threads, depth, out_file);
      return 0;
    }
  }

  // Print copyright notice
  std::cout << engine_name() << ".  "
            << "Copyright (C) 2004-2008 Tord Romstad."
            << std::endl
            << "PointChess derived from Glaurung. "
            << "Copyright (C) 2024-2026 hemu-create."
            << std::endl;

  // Enter UCI mode
  uci_main_loop();

  return 0;
}
