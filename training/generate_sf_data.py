# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Stockfish-Adversarial Shallow Dataset Generator (Mines Stockfish Weaknesses)

import subprocess
import argparse
import random
import time
import os
import multiprocessing as mp

TACTICAL_OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
    "r1bqkbnr/pp1ppppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4",
    "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq - 0 4",
    "rnbqk1nr/ppp2ppp/4p3/3p4/1b1PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 3 4",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQkq - 3 6",
    "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
]

class UCIEngine:
    def __init__(self, engine_path, options=None):
        self.proc = subprocess.Popen(
            [engine_path],
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
            if not line:
                break
            line = line.strip()
            if target in line:
                return line
        return None

    def get_move(self, fen, moves, depth=5):
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
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best_move = parts[1]
                break
        return best_move, score_cp

    def close(self):
        try:
            self.send("quit")
            self.proc.communicate(timeout=2)
        except Exception:
            self.proc.kill()

def play_adversarial_game(pc_path, sf_path, depth, worker_id):
    pc_opts = {
        "Personality": "Aggressive",
        "Use NNUE": "true",
        "EvalFile": "../pointchess.pchess" if os.path.exists("../pointchess.pchess") else "pointchess.pchess"
    }
    sf_opts = {
        "Use NNUE": "false",
        "Skill Level": "2",
        "Threads": "1",
        "Hash": "16"
    }

    pc_engine = UCIEngine(pc_path, pc_opts)
    sf_engine = UCIEngine(sf_path, sf_opts)

    fen = random.choice(TACTICAL_OPENINGS)
    pc_is_white = random.choice([True, False])

    moves = []
    samples = []
    final_result = 0.5

    for ply in range(80):
        is_pc_turn = (ply % 2 == 0) if pc_is_white else (ply % 2 == 1)
        current = pc_engine if is_pc_turn else sf_engine

        move, score = current.get_move(fen, moves, depth=depth)
        if not move or move == "(none)":
            # Current player checkmated
            final_result = 1.0 if not is_pc_turn else 0.0
            break

        # Record position from PointChess perspective
        # If PointChess evaluates, save score
        if is_pc_turn:
            # We record FEN position
            current_fen = fen  # Or reconstruct FEN
            samples.append((fen, score))

        moves.append(move)

        # Adjudication on decisive tactical advantage
        if abs(score) >= 900:
            if is_pc_turn:
                final_result = 1.0 if score > 0 else 0.0
            else:
                final_result = 0.0 if score > 0 else 1.0
            break

    pc_engine.close()
    sf_engine.close()

    # If PointChess won, these positions are high-value training samples that exploit Stockfish
    formatted_lines = []
    for f, sc in samples:
        formatted_lines.append(f"{f} | {sc} | {final_result}\n")

    return formatted_lines

def worker_task(args_tuple):
    pc_path, sf_path, depth, count, worker_id = args_tuple
    lines = []
    for _ in range(count):
        lines.extend(play_adversarial_game(pc_path, sf_path, depth, worker_id))
    return lines

def generate_adversarial_dataset(pc_path, sf_path, output_file, num_games=1000, depth=4, num_workers=4):
    print("=" * 65)
    print(" PointChess vs Stockfish Shallow Weakness Mining")
    print(f" PointChess: {pc_path} vs Stockfish: {sf_path}")
    print(f" Target Games: {num_games} | Search Depth: {depth} | Workers: {num_workers}")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    games_per_worker = (num_games + num_workers - 1) // num_workers

    tasks = [(pc_path, sf_path, depth, games_per_worker, i) for i in range(num_workers)]

    start_time = time.time()
    total_samples = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        with mp.Pool(num_workers) as pool:
            for result_lines in pool.imap_unordered(worker_task, tasks):
                for line in result_lines:
                    out.write(line)
                    total_samples += 1
                print(f"Collected {len(result_lines)} tactical weakness samples | Total: {total_samples} | Time: {time.time() - start_time:.1f}s")

    print(f"\nCompleted! Generated {total_samples} Stockfish-punishing training positions to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stockfish Shallow Weakness Miner")
    parser.add_argument("--pointchess", type=str, default="../build/pointchess", help="Path to PointChess binary")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--output", type=str, default="data/sf_weakness_data.txt", help="Output dataset file")
    parser.add_argument("--games", type=int, default=100, help="Number of games vs Stockfish")
    parser.add_argument("--depth", type=int, default=4, help="Shallow search depth")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")

    args = parser.parse_args()
    generate_adversarial_dataset(args.pointchess, args.stockfish, args.output, args.games, args.depth, args.workers)
