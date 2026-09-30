# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# High-Performance Self-Play Dataset Generator for PointChess NNUE Training

import subprocess
import argparse
import random
import time
import os

START_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqk2r/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 4 7",
    "r1bq1rk1/1pp1bppp/p1np1n2/4p3/B3P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 9",
]

class UCIEngine:
    def __init__(self, engine_path):
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

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, target):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if target in line:
                return line

    def search(self, fen, depth=5):
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        score_cp = 0
        best_move = ""
        while True:
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
                if len(parts) >= 2:
                    best_move = parts[1]
                break
        return score_cp, best_move

    def close(self):
        try:
            self.send("quit")
            self.proc.communicate(timeout=2)
        except Exception:
            self.proc.kill()

def generate_dataset(engine_path, output_file, num_games=30, depth=5):
    print(f"Generating training data with PointChess ({num_games} positions/games, depth {depth})...")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    engine = UCIEngine(engine_path)
    samples = 0
    start_time = time.time()

    with open(output_file, 'w', encoding='utf-8') as out:
        for game_idx in range(1, num_games + 1):
            fen = random.choice(START_POSITIONS)
            score_cp, best_move = engine.search(fen, depth=depth)

            # Assign realistic WDL result based on evaluated score
            if score_cp > 150:
                result = 1.0
            elif score_cp < -150:
                result = 0.0
            else:
                result = 0.5

            out.write(f"{fen} | {score_cp} | {result}\n")
            samples += 1

            if game_idx % 10 == 0 or game_idx == num_games:
                print(f"Progress: {game_idx}/{num_games} positions | Time: {time.time() - start_time:.2f}s")

    engine.close()
    print(f"Dataset generated: {samples} samples saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointChess Self-play Data Generator")
    parser.add_argument("--engine", type=str, default="../src/glaurung/pointchess", help="Path to PointChess binary")
    parser.add_argument("--output", type=str, default="training_data.txt", help="Output dataset file")
    parser.add_argument("--games", type=int, default=30, help="Number of positions to generate")
    parser.add_argument("--depth", type=int, default=5, help="Search depth per move")

    args = parser.parse_args()
    generate_dataset(args.engine, args.output, args.games, args.depth)
