# PointChess Roadmap

## Key decisions

- **Strength target**: beat Stockfish *without NNUE* (classical eval only)
  using search + classical evaluation. NNUE comes after that foundation.
- **NNUE format**: `.pnet` (PyTorch-native, Stockfish 17+ compatible) +
  our own binary `.nnue` for the C++ engine.
- **Training hardware**: Kaggle T4 x2 (2 GPUs, DDP, mixed precision).
- **No faking**: every claimed improvement measured via SPRT or controlled
  matches. No hardcoded moves, no fabricated Elo.

## Phase 1 — Audit & skeleton ✅
- [x] Inspect repo (empty — rebuild from scratch in Glaurung tradition)
- [x] docs/ARCHITECTURE.md, docs/ROADMAP.md, TODO.md
- [ ] CMake build system, directory skeleton

## Phase 2 — Correctness foundation
- [ ] Bitboard board representation, magic bitboards
- [ ] Move generation (all special moves)
- [ ] Make/unmake, state stack
- [ ] FEN parse/serialize, UCI position parsing
- [ ] Game-end detection: check, checkmate, stalemate, 50-move,
      insufficient material, repetition
- [ ] Perft tests vs known node counts (kiwipete, pos3/4/5, etc.)

## Phase 3 — Search + classical eval (playable engine)
- [ ] Negamax + alpha-beta + PVS, iterative deepening, aspiration
- [ ] TT, Zobrist hashing
- [ ] Move ordering: hash, MVV-LVA, killers, history, countermove, cont-hist
- [ ] NMP, LMR, futility, razoring, reverse futility, singular ext, check ext
- [ ] Quiescence with stand-pat, captures, delta pruning
- [ ] Time management, mate-distance, repetition in search
- [ ] Classical tapered eval (PST + material + mobility + king safety)
- [ ] UCI + bench + self-play vs Stockfish (no NNUE) — measure Elo

## Phase 4 — NNUE
- [ ] Feature extraction (king-relative halfKP), incremental accumulators
- [ ] Quantized int8 inference, CPU-optimized
- [ ] Network loading (.nnue binary + .pnet)
- [ ] Blend/switch: classical -> NNUE, verify no regression

## Phase 5 — Training pipeline (Kaggle T4 x2)
- [ ] PGN -> positions -> features -> sharded dataset
- [ ] PyTorch model, DDP on 2x T4, AMP, resumable, validated
- [ ] export.py -> .pnet + .nnue

## Phase 6 — Self-play
- [ ] Self-play worker (configurable depth/time, random openings,
      resignation, adjudication, PGN output, position extraction)

## Phase 7 — Opening book
- [ ] Polyglot-style binary book, builder from PGN, runtime probe
- [ ] BookMode = Popular/Strong/Diverse, no hardcoded lines

## Phase 8 — Testing & strength
- [ ] Unit tests: perft, FEN, movegen, make/unmake, hashing, UCI, NNUE
- [ ] SPRT match runner vs Stockfish (controlled TC/threads/hash)
- [ ] Experiment tracking (commit, net, config, result, Elo, confidence)

## Phase 9 — Final structure
- [ ] LICENSE (GPL-3.0-or-later), NOTICE (Glaurung derivation), README
- [ ] Final repo layout per spec
