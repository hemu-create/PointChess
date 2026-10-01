#!/bin/bash
# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# 60,000-Game Fastchess Match Runner (PointChess vs Stockfish)

GAMES=${1:-60000}
THREADS=${2:-10}
TC=${3:-"0.5+0.01"}
ROUNDS=$((GAMES / 2))

echo "============================================================"
echo " PointChess 60,000-Game Grand Tournament vs Stockfish"
echo " Total Games: $GAMES | Concurrency: $THREADS | TC: $TC"
echo " PGN Output:  match_60k.pgn | Log: match_60k.log"
echo "============================================================"

../tools/fastchess \
  -engine cmd=../build/pointchess name=PointChess option.EvalFile=../pointchess.pchess option.Use\ NNUE=true option.Personality=Solid option.Hash=16 \
  -engine cmd=/usr/games/stockfish name=Stockfish option.Use\ NNUE=false option.Skill\ Level=2 option.Hash=16 \
  -each tc=$TC proto=uci \
  -rounds $ROUNDS -games 2 -repeat -concurrency $THREADS \
  -pgnout file=match_60k.pgn \
  -log file=match_60k.log \
  -sprt elo0=0.0 elo1=5.0 alpha=0.05 beta=0.05
