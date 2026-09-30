# PointChess Engine Architecture

PointChess is a high-performance open-source chess engine built on the lineage of **Glaurung** (Tord Romstad, GPL-3.0), modernized with:
- **Dual Evaluation System**: High-accuracy NNUE neural network (.pnet & .nnue) + Tuned Classical Evaluation.
- **Modern Search Enhancements**: Stockfish-inspired logarithmic Late Move Reduction (LMR) table, Reverse Futility Pruning (RFP / Static Null Move Pruning), adaptive null move reduction, and PV search.
- **GPU & Multi-Backend Support**: Seamless switching between CPU NNUE SIMD, GPU (CUDA / OpenCL acceleration), and Classical CPU evaluation.
- **Chess Personality Engine**: UCI configurable playing styles (`Aggressive`, `Solid`, `Positional`, `Tactical`, `Gambiteer`, `Dynamic`, `Default`).
- **Kaggle T4 x2 GPU Training Pipeline**: Distributed PyTorch training using DDP, AMP mixed precision (fp16), and export to Stockfish-compatible `.pnet` format.

## System Diagram

```text
                        ┌──────────────────────────────┐
                        │      UCI Interface           │
                        │ (Options: Backend, NNUE,     │
                        │  Personality, Threads, Hash) │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
                        │        Iterative Deepening   │
                        │       & Principal Variation  │
                        └──────────────┬───────────────┘
                                       │
             ┌─────────────────────────┼────────────────────────┐
             │                         │                        │
  ┌──────────▼──────────┐   ┌──────────▼──────────┐  ┌──────────▼──────────┐
  │ Logarithmic LMR     │   │ Reverse Futility    │  │ Transposition Table │
  │ Table (depth x mc)  │   │ Pruning (RFP)       │  │ (Lockless Multi-    │
  └─────────────────────┘   └─────────────────────┘  │  threaded SMP)      │
                                                     └─────────────────────┘
                                       │
             ┌─────────────────────────┴────────────────────────┐
             │                                                  │
  ┌──────────▼───────────────┐                       ┌──────────▼──────────┐
  │ NNUE Neural Network Eval │                       │ Classical Tapered   │
  │ (HalfKP 40960->256->32->1│                       │ Evaluation (PST,    │
  │  SIMD CPU / CUDA GPU)    │                       │ Mobility, Safety)   │
  └──────────────────────────┘                       └─────────────────────┘
```

## Supported UCI Options

- `Backend`: `CPU` (default), `GPU` (CUDA), `Classical`
- `Use NNUE`: `true` / `false`
- `EvalFile`: Path to `.pnet` or `.nnue` network file (default `nnue.pnet`)
- `Personality`: `Default`, `Aggressive`, `Solid`, `Positional`, `Tactical`, `Gambiteer`, `Dynamic`
- `GPU Device`: GPU device index (0..16)
- `GPU Batch Size`: Evaluation batch size (1..1024)
- `Threads`: Number of CPU search threads (1..8)
- `Hash`: Transposition table size in MB (4..4096)
- `OwnBook`: `true` / `false` (Polyglot opening book)
- `Book File`: Path to binary opening book
