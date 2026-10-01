# PointChess 4-Billion Position NNUE Training Guide (Kaggle 2x T4 GPUs)

This document describes how to train PointChess's **Modern HalfKAv2** NNUE network on **Kaggle** using 2x NVIDIA Tesla T4 GPUs to reach peak ~3800-4000 Elo playing strength.

---

## 1. Multi-Stage 4-Billion Position Curriculum

To achieve super-GM strength and outperform Stockfish NNUE, the training data is structured in three stages:

1. **Stage 1 (0 to 1 Billion Positions)**:
   - Broad Opening & Self-Play Diversity (11 piece types HalfKAv2 with SCReL).
   - Generated with native parallel C++ generator (`pointchess genselfplay`).
2. **Stage 2 (1B to 2.5 Billion Positions)**:
   - Stockfish Official NNUE Knowledge Distillation (`training/generate_sf_nnue_data.py`).
   - Learns nuanced endgame and positional evaluations.
3. **Stage 3 (2.5B to 4 Billion Positions)**:
   - Adversarial Tactical Weakness Mining (`training/mine_sf_weaknesses.py`).
   - Identifies positions where Stockfish shallow search or heuristic evaluations fail and tags them with 5x training weight.

---

## 2. 32-Byte Compact Binary Sharding

To stream billions of positions without exceeding RAM or disk limits:

```bash
cd PointChess/training
python3 pack_binary.py --input raw_data.txt --output_prefix shards/pointchess_shard --shard_size 10000000
```

---

## 3. High-Throughput Kaggle 2x T4 Training (PyTorch DDP + AMP fp16)

Run with `torchrun` on Kaggle:

```bash
cd PointChess/training
torchrun --nproc_per_node=2 train_4b.py \
    --data_file /kaggle/input/pointchess-200k-dataset/pointchess_200k_tactical.txt \
    --max_positions 4000000000 \
    --batch_size 4096 \
    --lr 2e-3 \
    --output_dir /kaggle/working/checkpoints_4b
```

---

## 4. Exporting to `.pchess` and `.pnet`

```bash
python3 export.py \
    --checkpoint checkpoints_4b/best_model.pt \
    --output_pchess ../pointchess.pchess \
    --output_pnet ../nnue.pnet
```

---

## 5. Loading in PointChess via UCI

```text
uci
setoption name EvalFile value pointchess.pchess
setoption name Use NNUE value true
setoption name Personality value Solid
position startpos
go depth 14
```
