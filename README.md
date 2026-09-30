# PointChess

**PointChess** is a strong, modernized open-source chess engine derived from the Glaurung codebase lineage (Tord Romstad), enhanced with modern search heuristics, NNUE neural evaluation, GPU acceleration, and customizable playing personalities.

PointChess is licensed under **GPL-3.0-or-later**.

---

## Key Features

- **Modernized Search**: Stockfish-inspired logarithmic Late Move Reductions (LMR), Reverse Futility Pruning (RFP), adaptive null-move pruning, and principal variation search.
- **NNUE Neural Network Evaluation**: HalfKP feature transformer (40,960 inputs) with dual perspective accumulators and SCReL / ClippedReLU activations.
- **Dual Format Support**: Supports both PyTorch native `.pnet` format (Stockfish compatible) and optimized `.nnue` binary networks.
- **Multi-Backend Architecture**: Seamless switching via UCI between:
  - `CPU` (Fast SIMD NNUE)
  - `GPU` (CUDA / GPU acceleration)
  - `Classical` (High-speed tapered evaluation)
- **Chess Personalities**: UCI configurable playing styles:
  - `Aggressive` (High king attack, active piece play)
  - `Solid` (Pawn shelter, positional safety)
  - `Positional` (Space control, bishop pair, outposts)
  - `Tactical` (Sharp dynamic variations, check pressure)
  - `Gambiteer` (Material sacrifices for initiative)
  - `Dynamic` (Unbalanced compensation)
  - `Default` (Balanced master play)
- **GPU Training Pipeline**: Ready-to-run PyTorch DDP training pipeline optimized for **Kaggle 2x NVIDIA T4 GPUs** with automatic mixed precision (AMP fp16).
- **SPRT Testing**: Statistical testing tool for strength validation.

---

## Quick Start

### Build PointChess

```bash
cd src/glaurung
make -j$(nproc)
```

The resulting executable is `src/glaurung/pointchess`.

### Run Benchmark

```bash
./pointchess bench 128 1 12
```

### Run via UCI

```bash
./pointchess
```
```text
uci
setoption name Personality value Aggressive
setoption name Backend value CPU
setoption name Use NNUE value true
setoption name EvalFile value ../../nnue.pnet
position startpos
go depth 12
```

---

## NNUE GPU Training on Kaggle

PointChess includes a multi-GPU training pipeline in `training/`:

```bash
cd training
python3 kaggle_run.py
```

Or run directly with `torchrun`:
```bash
torchrun --nproc_per_node=2 train.py --epochs 30 --batch_size 512 --amp
python3 export.py --checkpoint checkpoints/best_model.pt --output_pnet ../nnue.pnet
```

---

## Directory Structure

```text
PointChess/
├── LICENSE
├── NOTICE
├── AUTHORS
├── README.md
├── CMakeLists.txt
├── TODO.md
│
├── src/
│   ├── glaurung/        Core engine (Board, Search, Movegen, UCI, Eval)
│   ├── nnue/            NNUE neural architecture & inference
│   └── eval/            Backend manager & personality engine
│
├── training/
│   ├── model.py         PyTorch NNUE architecture
│   ├── dataset.py       High-speed HalfKP dataset reader
│   ├── train.py         Multi-GPU DDP training script
│   ├── export.py        Weight exporter (.pnet and .nnue)
│   ├── generate_data.py Self-play data generator
│   └── kaggle_run.py    Kaggle T4 x2 runner
│
├── book/
│   ├── builder/         Opening book builder
│   └── README.md
│
├── tools/
│   └── sprt_match.py    SPRT match runner vs opponents
│
└── docs/
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    ├── SEARCH.md
    ├── NNUE.md
    └── BENCHMARKING.md
```

---

## License & Attribution

- **PointChess**: Copyright (C) 2024-2026 hemu-create and contributors.
- **Glaurung**: Copyright (C) 2004-2008 Tord Romstad.
- Licensed under **GNU General Public License v3.0 or later** (GPL-3.0-or-later).
