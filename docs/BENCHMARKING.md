# PointChess Benchmarking & Strength Testing

PointChess enforces strict, reproducible strength testing using the **Sequential Probability Ratio Test (SPRT)**.

## SPRT Testing Methodology

To test PointChess against Stockfish or other engines:

```bash
cd PointChess/tools
python3 sprt_match.py \
    --engine1 ../src/glaurung/pointchess \
    --engine2 /path/to/stockfish \
    --games 200 \
    --tc 10+0.1 \
    --threads 1 \
    --hash 64 \
    --elo0 0.0 \
    --elo1 5.0
```

## Running the Internal Benchmark

PointChess includes a built-in search benchmark command:

```bash
cd PointChess/src/glaurung
./pointchess bench 128 1 12
```

This runs a standardized suite of tactical and positional test positions and reports the total nodes searched and nodes per second (NPS).
