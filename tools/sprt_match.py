# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Real-Time UCI Match & SPRT Runner (PointChess vs Stockfish)

import subprocess
import argparse
import math
import time
import os
import sys

class UCIEngineProcess:
    def __init__(self, path, name="Engine", options=None):
        self.path = path
        self.name = name
        self.options = options or {}
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
        for k, v in self.options.items():
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

    def get_move(self, moves_list, movetime_ms=100, depth=None):
        self.send("ucinewgame") if len(moves_list) == 0 else None
        pos_cmd = "position startpos"
        if moves_list:
            pos_cmd += " moves " + " ".join(moves_list)
        self.send(pos_cmd)

        if depth is not None:
            self.send(f"go depth {depth}")
        else:
            self.send(f"go movetime {movetime_ms}")

        best_move = None
        score_cp = 0
        start = time.time()
        while time.time() - start < 15.0:
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

def sprt_calc(wins, losses, draws, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05):
    n = wins + losses + draws
    if n == 0:
        return "CONTINUE", 0.0, 0.0

    w = wins / n
    l = losses / n
    d = draws / n
    s = w + 0.5 * d

    p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))

    var_s = (w * (1 - s)**2 + l * (0 - s)**2 + d * (0.5 - s)**2) / n
    if var_s <= 1e-6:
        var_s = 0.25 / n

    llr = (p1 - p0) * (s - (p0 + p1) / 2.0) / var_s

    la = math.log(beta / (1.0 - alpha))
    lb = math.log((1.0 - beta) / alpha)

    if llr >= lb:
        status = "H1_ACCEPTED (Pass - Statistically Superior)"
    elif llr <= la:
        status = "H0_ACCEPTED (Fail - Not Superior)"
    else:
        status = "CONTINUE"

    elo_diff = -400.0 * math.log10(1.0 / max(1e-4, min(0.9999, s)) - 1.0) if s > 0 and s < 1 else 0.0
    return status, llr, elo_diff

def play_game(engine_w, engine_b, movetime_ms=100, depth=None, max_moves=100):
    moves = []
    positions_history = []
    
    for ply in range(max_moves * 2):
        current_engine = engine_w if ply % 2 == 0 else engine_b
        move, eval_cp = current_engine.get_move(moves, movetime_ms=movetime_ms, depth=depth)

        if not move or move == "(none)":
            # No move available (checkmate or stalemate)
            # If current player had no move, previous player won
            return 1.0 if (ply % 2 == 1) else 0.0, moves, "No legal moves (Mate/Stalemate)"

        moves.append(move)

        # Adjudication on decisive checkmate advantage (only if eval >= 3000 cp / mate)
        if ply >= 40 and abs(eval_cp) >= 3000:
            if eval_cp > 0:
                return 1.0 if (ply % 2 == 0) else 0.0, moves, "Adjudication (Eval > 30.0)"
            else:
                return 0.0 if (ply % 2 == 0) else 1.0, moves, "Adjudication (Eval < -30.0)"

        # Repetition detection (simplistic 3-fold check)
        if len(moves) >= 8 and moves[-1] == moves[-5] and moves[-2] == moves[-6] and moves[-3] == moves[-7] and moves[-4] == moves[-8]:
            return 0.5, moves, "3-Fold Repetition"

    return 0.5, moves, "Move Limit Draw"

def main():
    parser = argparse.ArgumentParser(description="PointChess vs Stockfish Real-Time Match Runner")
    parser.add_argument("--pointchess", type=str, default="../build/pointchess", help="Path to PointChess binary")
    parser.add_argument("--stockfish", type=str, default="/usr/games/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--movetime", type=int, default=50, help="Move time per turn in milliseconds")
    parser.add_argument("--depth", type=int, default=None, help="Fixed search depth per move (optional)")
    parser.add_argument("--personality", type=str, default="Aggressive", help="PointChess personality")
    parser.add_argument("--sf_nnue", action="store_true", default=False, help="Enable NNUE for Stockfish (default: False / Classical)")
    parser.add_argument("--sf_skill", type=int, default=10, help="Stockfish Skill Level (0-20)")

    args = parser.parse_args()

    print("=" * 65)
    print(" PointChess vs Stockfish Match Runner")
    print(f" PointChess: {args.pointchess} (Personality: {args.personality})")
    print(f" Stockfish:  {args.stockfish} (NNUE: {args.sf_nnue}, Skill Level: {args.sf_skill})")
    print(f" Games: {args.games} | Time Per Move: {args.movetime}ms")
    print("=" * 65)

    pc_opts = {
        "Personality": args.personality,
        "Use NNUE": "true",
        "EvalFile": "../pointchess.pchess" if os.path.exists("../pointchess.pchess") else "pointchess.pchess"
    }

    sf_opts = {
        "Use NNUE": "true" if args.sf_nnue else "false",
        "Skill Level": str(args.sf_skill),
        "Threads": "1",
        "Hash": "16"
    }

    wins = 0
    losses = 0
    draws = 0

    start_match = time.time()

    for game_num in range(1, args.games + 1):
        pc_is_white = (game_num % 2 == 1)
        
        if pc_is_white:
            engine_w = UCIEngineProcess(args.pointchess, "PointChess", pc_opts)
            engine_b = UCIEngineProcess(args.stockfish, "Stockfish", sf_opts)
        else:
            engine_w = UCIEngineProcess(args.stockfish, "Stockfish", sf_opts)
            engine_b = UCIEngineProcess(args.pointchess, "PointChess", pc_opts)

        print(f"\n[Game {game_num:02d}/{args.games:02d}] White: {'PointChess' if pc_is_white else 'Stockfish'} vs Black: {'Stockfish' if pc_is_white else 'PointChess'} ... ", end="", flush=True)

        res, moves, reason = play_game(engine_w, engine_b, movetime_ms=args.movetime, depth=args.depth)

        engine_w.close()
        engine_b.close()

        # Score from PointChess perspective
        if pc_is_white:
            pc_score = res
        else:
            pc_score = 1.0 - res

        if pc_score == 1.0:
            wins += 1
            outcome_str = "PointChess WON!"
        elif pc_score == 0.0:
            losses += 1
            outcome_str = "Stockfish won"
        else:
            draws += 1
            outcome_str = "Draw"

        status, llr, elo_diff = sprt_calc(wins, losses, draws)
        total_played = wins + losses + draws
        pct = (wins + 0.5 * draws) / total_played * 100.0

        print(f"{outcome_str} ({reason}, {len(moves)} plies)")
        print(f"  Score: PointChess {wins} - {losses} - {draws} ({pct:.1f}%) | Elo Diff: {elo_diff:+.1f} | Elapsed: {time.time() - start_match:.1f}s")

    print("\n" + "=" * 65)
    print(" Match Complete!")
    print(f" Final Score: PointChess {wins} - {losses} - {draws} / {args.games} ({(wins + 0.5*draws)/args.games*100:.1f}%)")
    status, llr, elo_diff = sprt_calc(wins, losses, draws)
    print(f" Elo Difference vs Stockfish: {elo_diff:+.1f} (LLR: {llr:.2f})")
    print("=" * 65)

if __name__ == "__main__":
    main()
