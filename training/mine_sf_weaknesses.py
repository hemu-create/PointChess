# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Stockfish Tactical Weakness Miner & Adversarial Blunder Finder

import subprocess
import argparse
import random
import time
import os
import multiprocessing as mp

TACTICAL_OPENING_SEEDS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq - 0 4",
    "r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4",
    "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQkq - 3 6",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
]

class UCIEngine:
    def __init__(self, path, options=None):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.send("uci")
        self.wait_for("uciok")
        if options:
            for k, v in options.items():
                self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, target, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout:
            line = self.proc.stdout.readline()
            if not line: break
            line = line.strip()
            if target in line: return line
        return None

    def search(self, fen, moves=None, depth=5):
        pos_cmd = f"position fen {fen}"
        if moves:
            pos_cmd += " moves " + " ".join(moves)
        self.send(pos_cmd)
        self.send(f"go depth {depth}")

        best_move = None
        score_cp = 0
        start = time.time()
        while time.time() - start < 10.0:
            line = self.proc.stdout.readline()
            if not line: break
            line = line.strip()
            if "score cp" in line:
                parts = line.split()
                if "cp" in parts:
                    idx = parts.index("cp")
                    if idx + 1 < len(parts):
                        try: score_cp = int(parts[idx + 1])
                        except ValueError: pass
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

def mine_weakness_worker(args_tuple):
    pc_path, sf_path, num_positions, shallow_depth, deep_depth, worker_id = args_tuple
    
    sf_engine = UCIEngine(sf_path, {"Use NNUE": "true", "Threads": "1", "Hash": "16"})
    pc_engine = UCIEngine(pc_path, {"Personality": "Aggressive", "Use NNUE": "true"})

    samples = []
    found_weaknesses = 0

    for _ in range(num_positions):
        fen = random.choice(TACTICAL_OPENING_SEEDS)
        moves = []

        # Play 1-4 random plies to create novel tactical board states
        for _ in range(random.randint(1, 4)):
            _, m = pc_engine.search(fen, moves, depth=2)
            if not m or m == "(none)": break
            moves.append(m)

        # 1. Stockfish shallow evaluation
        sf_shallow_eval, sf_move = sf_engine.search(fen, moves, depth=shallow_depth)
        if not sf_move or sf_move == "(none)": continue

        # 2. Deep tactical refutation check
        moves_with_sf_choice = moves + [sf_move]
        refutation_eval, _ = pc_engine.search(fen, moves_with_sf_choice, depth=deep_depth)

        # 3. Detect Weakness:
        # If Stockfish thought it was equal/better (e.g. >= 0) but deep search finds a refutation (e.g. -250 cp or mate),
        # Stockfish has blundered into a tactical trap!
        is_blunder = (sf_shallow_eval > -100 and refutation_eval < -200) or (abs(sf_shallow_eval - refutation_eval) > 350)
        
        # Training weight: 5.0 for blunders/weaknesses, 1.0 for normal positions
        weight = 5.0 if is_blunder else 1.0
        if is_blunder:
            found_weaknesses += 1

        target_score = refutation_eval
        target_wdl = 1.0 if target_score > 150 else (0.0 if target_score < -150 else 0.5)

        samples.append(f"{fen} | {target_score} | {target_wdl} | {weight}\n")

    sf_engine.close()
    pc_engine.close()
    return samples, found_weaknesses

def main():
    parser = argparse.ArgumentParser(description="PointChess Stockfish Weakness Miner (Scalable to Billions)")
    parser.add_argument("--pointchess", type=str, default="../build/pointchess", help="Path to PointChess binary")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish", help="Path to Stockfish NNUE binary")
    parser.add_argument("--output", type=str, default="data/sf_mined_weaknesses.txt", help="Output dataset path")
    parser.add_argument("--samples", type=int, default=1000, help="Number of tactical weakness samples to mine")
    parser.add_argument("--shallow_depth", type=int, default=4, help="Stockfish shallow search depth")
    parser.add_argument("--deep_depth", type=int, default=8, help="Deep tactical refutation depth")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU worker processes")

    args = parser.parse_args()

    print("=" * 65)
    print(" PointChess Stockfish Weakness & Blunder Mining System")
    print(f" Target Positions: {args.samples:,} | Workers: {args.workers}")
    print(f" Shallow SF Depth: {args.shallow_depth} | Deep Refutation Depth: {args.deep_depth}")
    print(f" Output File:      {args.output}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    samples_per_worker = (args.samples + args.workers - 1) // args.workers
    tasks = [(args.pointchess, args.stockfish, samples_per_worker, args.shallow_depth, args.deep_depth, i) for i in range(args.workers)]

    start_time = time.time()
    total_samples = 0
    total_weaknesses = 0

    with open(args.output, 'w', encoding='utf-8') as out:
        with mp.Pool(args.workers) as pool:
            for result_lines, weaknesses in pool.imap_unordered(mine_weakness_worker, tasks):
                for line in result_lines:
                    out.write(line)
                    total_samples += 1
                total_weaknesses += weaknesses
                elapsed = time.time() - start_time
                print(f"Positions Mined: {total_samples:,}/{args.samples:,} | Weaknesses Tagged: {total_weaknesses:,} | Speed: {int(total_samples/max(0.1, elapsed))} pos/sec")

    print("\n" + "=" * 65)
    print(f" Mining Complete! Saved {total_samples:,} positions ({total_weaknesses:,} Stockfish blunders) to {args.output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
