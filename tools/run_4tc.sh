#!/bin/bash
# PointChess 4-TC controlled harness: Bullet/Blitz/Rapid/Classical
# Identical HW/threads/hash/book/adjudication, only TC changes.
set -u
PC=./build/pointchess
SF=./tools/stockfish_19
FC=./tools/fastchess
NET=./pointchess.pchess
GAMES=${1:-8}   # games per TC (total, split colors). Use 200+ for real SPRT.
OUTDIR=tools/results_4tc
mkdir -p $OUTDIR
run_tc() {
  local name=$1 tc=$2
  echo "=== $name ($tc) ==="
  $FC \
    -engine cmd=$PC name=PC option.EvalFile=$NET option."Use NNUE"=true option.Personality=Default option.Hash=32 \
    -engine cmd=$SF name=SF option."Use NNUE"=true option.Hash=32 \
    -each tc=$tc proto=uci \
    -rounds $((GAMES/2)) -games 2 -repeat -concurrency 2 \
    -draw movenumber=40 movecount=5 score=10 \
    -resign movecount=3 score=600 \
    -log file=$OUTDIR/$name.log 2>&1 | tail -n 12 | tee $OUTDIR/$name.summary
}
run_tc Bullet  "0.25+0.02"
run_tc Blitz   "1+0.05"
run_tc Rapid   "3+0.1"
run_tc Classical "5+0.2"
echo "Summaries in $OUTDIR/*.summary"
