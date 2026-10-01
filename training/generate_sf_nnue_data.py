# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Stockfish Official NNUE Knowledge Distillation Generator

import subprocess
import argparse
import random
import time
import os
import multiprocessing as mp

TACTICAL_OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # King's Gambit & Vienna
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
    # Sicilian Dragon & Najdorf
    "r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4",
    "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    # Evans Gambit & Italian Sharp
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq - 0 4",
    # French Winawer
    "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    # Caro-Kann Advance
    "rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    # Queen's Gambit & Semi-Slav
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4",
    # King's Indian & Grunfeld
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQkq - 3 6",
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    # Scandinavian & Modern
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
]

class StockfishTeacher:
    def __init__(self, sf_path):
        self.proc = subprocess.Popen(
            [sf_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.send("uci")
        self.wait_for("uciok")
        self.send("setoption name Use NNUE value true")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 16")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, target, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if target in line:
                return line
        return None

    def evaluate_position(self, fen, moves=None, depth=5):
        pos_cmd = f"position fen {fen}"
        if moves:
            pos_cmd += " moves " + " ".join(moves)
        self.send(pos_cmd)
        self.send(f"go depth {depth}")

        best_move = None
        score_cp = 0
        is_mate = False
        start = time.time()
        while time.time() - start < 10.0:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if "score cp" in line:
                parts = line.split()
                if "cp" in parts:
                    idx = parts.index("cp")
                    if idx + 1 < len(parts):
                        try:
                            score_cp = int(parts[idx + 1])
                        except ValueError:
                            pass
            elif "score mate" in line:
                is_mate = True
                parts = line.split()
                if "mate" in parts:
                    idx = parts.index("mate")
                    if idx + 1 < len(parts):
                        try:
                            m_ply = int(parts[idx + 1])
                            score_cp = 15000 if m_ply > 0 else -15000
                        except ValueError:
                            pass
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best_move = parts[1]
                break
        return score_cp, best_move

    def close(self):
        try:
            self.send("quit")
            self.proc.communicate(timeout=2)
        except Exception:
            self.proc.kill()

def worker_generator(args_tuple):
    sf_path, games_count, search_depth, worker_id = args_tuple
    teacher = StockfishTeacher(sf_path)
    samples = []

    for _ in range(games_count):
        fen = random.choice(TACTICAL_OPENINGS)
        moves = []
        game_records = []

        # Play shallow games using Stockfish NNUE
        for ply in range(60):
            score_cp, move = teacher.evaluate_position(fen, moves, depth=search_depth)
            if not move or move == "(none)":
                break

            # Record position with Stockfish NNUE evaluation
            game_records.append((fen, moves.copy(), score_cp))
            moves.append(move)

            # Adjudicate on clear winning advantage
            if abs(score_cp) >= 1200:
                break

        # Calculate game outcome for WDL target
        final_score = game_records[-1][2] if game_records else 0
        if final_score >= 150:
            result = 1.0
        elif final_score <= -150:
            result = 0.0
        else:
            result = 0.5

        for f_base, m_list, sc in game_records:
            # Reconstruct FEN with moves
            if not m_list:
                f_str = f_base
            else:
                f_str = f"{f_base} (moves: {' '.join(m_list)})"
            samples.append(f"{f_base} | {sc} | {result}\n")

    teacher.close()
    return samples

def main():
    parser = argparse.ArgumentParser(description="Stockfish Official NNUE Distillation Dataset Generator")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish", help="Path to Stockfish binary with official NNUE")
    parser.add_argument("--output", type=str, default="data/sf_nnue_distillation.txt", help="Output distillation dataset file")
    parser.add_argument("--games", type=int, default=500, help="Number of games to simulate with Stockfish NNUE")
    parser.add_argument("--depth", type=int, default=5, help="Stockfish NNUE search depth")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")

    args = parser.parse_args()

    print("=" * 65)
    print(" Stockfish Official NNUE Knowledge Distillation Generator")
    print(f" Teacher Engine: {args.stockfish}")
    print(f" Target Games:   {args.games} | Depth: {args.depth} | Workers: {args.workers}")
    print(f" Output File:    {args.output}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    games_per_worker = (args.games + args.workers - 1) // args.workers
    tasks = [(args.stockfish, games_per_worker, args.depth, i) for i in range(args.workers)]

    start_time = time.time()
    total_positions = 0

    with open(args.output, 'w', encoding='utf-8') as out:
        with mp.Pool(args.workers) as pool:
            for result_lines in pool.imap_unordered(worker_generator, tasks):
                for line in result_lines:
                    out.write(line)
                    total_positions += 1
                elapsed = time.time() - start_time
                print(f"Positions extracted: {total_positions:,} | Time: {elapsed:.1f}s ({int(total_positions/max(0.1, elapsed))} pos/sec)")

    print("\n" + "=" * 65)
    print(f" Distillation Complete! Extracted {total_positions:,} Stockfish NNUE positions to {args.output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
