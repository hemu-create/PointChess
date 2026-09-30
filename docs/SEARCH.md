# PointChess Search Architecture

PointChess combines Glaurung's rock-solid foundation with modern search optimizations:

## 1. Core Framework
- **Principal Variation Search (PVS)**: Zero-window searches (`-(beta-1), -alpha`) for non-PV nodes.
- **Iterative Deepening**: Progressively deeper searches with best-move history carryover.
- **Transposition Table (TT)**: Depth-preferred replacement strategy with exact, upper-bound, and lower-bound flags.

## 2. Modern Pruning & Reductions
- **Logarithmic Late Move Reductions (LMR)**:
  `Reduction = Base + log(depth) * log(moveCount) / Scale`
  Precomputed in a 2D table `LMRTable[pv][depth][move_count]`.
- **Reverse Futility Pruning (Static Null Move Pruning / RFP)**:
  At shallow depths, if static evaluation minus a depth-dependent margin exceeds beta, prune the node immediately.
- **Adaptive Null Move Pruning (NMP)**:
  Dynamic reduction `R = 3 + depth / 4` with zugzwang verification in late endgames.
- **Futility Pruning**:
  Pruning quiet moves at frontier nodes when static eval plus margin cannot reach alpha.
- **Razoring**:
  Early quiescence drop at shallow depths when eval is far below alpha.

## 3. Move Ordering
- **PV Move from TT**: Searched first with full window.
- **Captures ordered by MVV-LVA & SEE**: Winning/equal captures searched before quiet moves.
- **Mate Killer & Killer Moves**: 2 killer moves per ply stored and tried early.
- **History Heuristic**: Dynamic Butterfly history table updated upon beta cutoffs.
