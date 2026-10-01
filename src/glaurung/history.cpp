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

#include <cassert>
#include <cstring>

#include "history.h"


////
//// Functions
////

/// Constructor

History::History() {
  this->clear();
}


/// History::clear() clears the history tables.

// Gravity update borrowed from Stockfish history.h StatsEntry::operator<<
// (GPL-3.0, cf. tools/sf_ref/history.h): self-attenuating, bounded in
// [-D, D], no global rescale pass needed.
static const int HistoryGravityD = 16384;

static void gravity_update(int &entry, int bonus) {
  if(bonus > HistoryGravityD) bonus = HistoryGravityD;
  if(bonus < -HistoryGravityD) bonus = -HistoryGravityD;
  int absBonus = bonus >= 0 ? bonus : -bonus;
  entry += bonus - entry * absBonus / HistoryGravityD;
}

void History::clear() {
  memset(history, 0, 2 * 8 * 64 * sizeof(int));
  memset(successCount, 0, 2 * 8 * 64 * sizeof(int));
  memset(failureCount, 0, 2 * 8 * 64 * sizeof(int));
  memset(countermoves, 0, 64 * 64 * sizeof(Move));
  memset(contHistory, 0, 16 * 64 * 16 * 64 * sizeof(int));
  memset(contHistory2Ply, 0, 16 * 64 * 16 * 64 * sizeof(int));
  memset(captureHistory, 0, 16 * 64 * 8 * sizeof(int));
}

void History::update_countermove(Move prevMove, Move refutation) {
  if (prevMove != MOVE_NONE && move_is_ok(prevMove) && refutation != MOVE_NONE && move_is_ok(refutation)) {
    countermoves[move_from(prevMove)][move_to(prevMove)] = refutation;
  }
}

Move History::get_countermove(Move prevMove) const {
  if (prevMove != MOVE_NONE && move_is_ok(prevMove)) {
    return countermoves[move_from(prevMove)][move_to(prevMove)];
  }
  return MOVE_NONE;
}

void History::capture_success(Piece p, Move m, PieceType cap, Depth d) {
  if (piece_is_ok(p) && move_is_ok(m) && cap >= PAWN && cap <= KING) {
    int plies = int(d) / int(OnePly);
    gravity_update(captureHistory[p][move_to(m)][int(cap)], 16 * plies * plies);
  }
}

int History::capture_score(Piece p, Move m, PieceType cap) const {
  if (piece_is_ok(p) && move_is_ok(m) && cap >= PAWN && cap <= KING) {
    return captureHistory[p][move_to(m)][int(cap)];
  }
  return 0;
}


/// History::success() registers a move as being successful.  This is done
/// whenever a non-capturing move causes a beta cutoff in the main search.
/// The three parameters are the moving piece, the move itself, and the
/// search depth.

void History::success(Piece p, Move m, Depth d,
                      Move prevMove1, Piece prevP1,
                      Move prevMove2, Piece prevP2) {
  assert(piece_is_ok(p));
  assert(move_is_ok(m));

  int plies = int(d) / int(OnePly);
  gravity_update(history[p][move_to(m)], 8 * plies * plies);
  successCount[p][move_to(m)]++;

  // 1-Ply Continuation History
  if (prevMove1 != MOVE_NONE && move_is_ok(prevMove1) && prevP1 != NO_PIECE && piece_is_ok(prevP1)) {
    gravity_update(contHistory[prevP1][move_to(prevMove1)][p][move_to(m)], 12 * plies * plies);
  }

  // 2-Ply Continuation History
  if (prevMove2 != MOVE_NONE && move_is_ok(prevMove2) && prevP2 != NO_PIECE && piece_is_ok(prevP2)) {
    gravity_update(contHistory2Ply[prevP2][move_to(prevMove2)][p][move_to(m)], 8 * plies * plies);
  }

  // Legacy overflow guard (gravity bounds entries, this is belt & braces):
  if(history[p][move_to(m)] >= HistoryMax)
    for(int i = 0; i < 16; i++)
      for(int j = 0; j < 64; j++)
        history[i][j] /= 2;
}


/// History::failure() registers a move as being unsuccessful.  The function is
/// called for each non-capturing move which failed to produce a beta cutoff
/// at a node where a beta cutoff was finally found.

void History::failure(Piece p, Move m, Depth d,
                      Move prevMove1, Piece prevP1,
                      Move prevMove2, Piece prevP2) {
  assert(piece_is_ok(p));
  assert(move_is_ok(m));

  int plies = int(d) / int(OnePly);
  gravity_update(history[p][move_to(m)], -8 * plies * plies);
  failureCount[p][move_to(m)]++;

  if (prevMove1 != MOVE_NONE && move_is_ok(prevMove1) && prevP1 != NO_PIECE && piece_is_ok(prevP1)) {
    gravity_update(contHistory[prevP1][move_to(prevMove1)][p][move_to(m)], -12 * plies * plies);
  }

  if (prevMove2 != MOVE_NONE && move_is_ok(prevMove2) && prevP2 != NO_PIECE && piece_is_ok(prevP2)) {
    gravity_update(contHistory2Ply[prevP2][move_to(prevMove2)][p][move_to(m)], -8 * plies * plies);
  }
}


/// History::move_ordering_score() returns an integer value used to order the
/// non-capturing moves in the MovePicker class.

int History::move_ordering_score(Piece p, Move m,
                                 Move prevMove1, Piece prevP1,
                                 Move prevMove2, Piece prevP2) const {
  assert(piece_is_ok(p));
  assert(move_is_ok(m));

  int score = history[p][move_to(m)];
  if (prevMove1 != MOVE_NONE && move_is_ok(prevMove1) && prevP1 != NO_PIECE && piece_is_ok(prevP1)) {
    score += contHistory[prevP1][move_to(prevMove1)][p][move_to(m)];
  }
  if (prevMove2 != MOVE_NONE && move_is_ok(prevMove2) && prevP2 != NO_PIECE && piece_is_ok(prevP2)) {
    score += contHistory2Ply[prevP2][move_to(prevMove2)][p][move_to(m)];
  }
  return score;
}


/// History::ok_to_prune() decides whether a move has been sufficiently
/// unsuccessful that it makes sense to prune it entirely. 

bool History::ok_to_prune(Piece p, Move m, Depth d) const {
  assert(piece_is_ok(p));
  assert(move_is_ok(m));

  return (int(d) * successCount[p][move_to(m)] < failureCount[p][move_to(m)]);
}
