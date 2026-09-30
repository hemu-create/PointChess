# TODO & Milestone Status

## Completed Milestones ✅

- [x] **Phase 1: Foundation & Audit**: Established codebase lineage from Glaurung 2.2 (GPL-3.0) with proper attribution.
- [x] **Phase 2: Correctness Foundation**: Full perft test suite passing (6/6 positions: startpos, kiwipete, pos3, pos4, pos5, pos6).
- [x] **Phase 3: Search Modernization**:
  - Stockfish-style logarithmic Late Move Reductions (LMR) table (`LMRTable[pv][depth][move_count]`).
  - Reverse Futility Pruning (Static Null Move Pruning / RFP) at shallow depths.
  - Adaptive dynamic Null Move reduction (`R = 3 + depth / 4`).
  - Principal Variation Search (PVS) with fast move ordering.
- [x] **Phase 3b: Chess Personality Engine**:
  - UCI options for 7 distinct personalities: `Aggressive`, `Solid`, `Positional`, `Tactical`, `Gambiteer`, `Dynamic`, `Default`.
  - Dynamic multipliers adjusting king attack, piece mobility, space, cowardice, aggressiveness, and pawn structure.
- [x] **Phase 4: High-Performance NNUE System**:
  - HalfKP feature transformer (40,960 inputs) with dual perspective accumulators.
  - SCReL / ClippedReLU non-linear activations.
  - Quantized integer SIMD inference.
  - `.pnet` (PyTorch state dict) and `.nnue` binary loader support.
- [x] **Phase 5: GPU Multi-Backend & Lc0-Style Options**:
  - UCI options `Backend` (`CPU`, `GPU`, `Classical`), `GPU Device`, `GPU Batch Size`.
  - CUDA GPU device detection and runtime management.
- [x] **Phase 6: Kaggle T4 x2 GPU Training Pipeline**:
  - Multi-GPU PyTorch DDP training with `torchrun` and automatic mixed precision (AMP fp16).
  - Self-play data generator (`generate_data.py`).
  - Direct weight exporter to `.pnet` and `.nnue` (`export.py`).
  - Single-command Kaggle runner (`kaggle_run.py`).
- [x] **Phase 7: Opening Book System**:
  - Polyglot binary book support and PGN book compiler (`book/builder/build_book.py`).
- [x] **Phase 8: Validation & Tools**:
  - Automated SPRT match runner (`tools/sprt_match.py`).
  - Both Makefile and CMake build systems verified.
  - Complete technical documentation in `docs/`.
