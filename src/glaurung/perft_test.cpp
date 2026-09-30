// Perft correctness test for Glaurung movegen.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "position.h"
#include "movegen.h"
#include "mersenne.h"
#include "material.h"
#include "movepick.h"
#include "evaluate.h"
#include "bitbase.h"
#include "direction.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

static uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    MoveStack moves[256];
    int n = generate_legal_moves(pos, moves);
    if (depth == 1) return (uint64_t)n;
    UndoInfo u;
    uint64_t nodes = 0;
    for (int i = 0; i < n; ++i) {
        pos.do_move(moves[i].move, u);
        nodes += perft(pos, depth - 1);
        pos.undo_move(moves[i].move, u);
    }
    return nodes;
}

struct PerftPos {
    const char* name;
    const char* fen;
    int depth;
    uint64_t expected;
};

int main() {
    init_mersenne();
    init_direction_table();
    init_bitboards();
    Position::init_zobrist();
    Position::init_piece_square_tables();
    MaterialInfo::init();
    MovePicker::init_phase_table();
    init_eval(1);
    init_bitbases();

    PerftPos positions[] = {
        {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324ULL},
        {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690ULL},
        {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 7, 178633661ULL},
        {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 6, 706045033ULL},
        {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194ULL},
        {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551ULL},
    };

    int failures = 0;
    for (auto& p : positions) {
        Position pos(p.fen);
        uint64_t result = perft(pos, p.depth);
        bool ok = (result == p.expected);
        printf("%-12s depth %d: %llu (expected %llu) %s\n",
               p.name, p.depth,
               (unsigned long long)result, (unsigned long long)p.expected,
               ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }
    printf("\n%s\n", failures ? "PERFT FAILURES" : "ALL PERFT TESTS PASSED");
    return failures ? 1 : 0;
}
