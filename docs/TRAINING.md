# PointChess NNUE Training Guide (Kaggle T4 x2)

This document describes how to train PointChess NNUE networks on **Kaggle** using 2x NVIDIA T4 GPUs.

## 1. Setup on Kaggle

In your Kaggle notebook:
1. Set the accelerator to **GPU T4 x2**.
2. Clone or upload the PointChess repository.
3. Verify the GPUs:
   ```python
   import torch
   print("GPUs:", torch.cuda.device_count()) # Should print 2
   ```

## 2. Running Training via Single Command

To run the automated data generation, multi-GPU training, and `.pnet` export:

```bash
cd PointChess/training
python3 kaggle_run.py
```

## 3. Manual Multi-GPU Training (torchrun)

To train with custom hyperparameters using PyTorch DistributedDataParallel (DDP):

```bash
cd PointChess/training
torchrun --nproc_per_node=2 train.py \
    --data_file data/selfplay.txt \
    --epochs 30 \
    --batch_size 512 \
    --lr 1e-3 \
    --activation screl \
    --amp \
    --output_dir checkpoints
```

## 4. Exporting to `.pnet` (Stockfish compatible) and `.nnue`

```bash
python3 export.py \
    --checkpoint checkpoints/best_model.pt \
    --output_pnet ../nnue.pnet \
    --output_nnue ../pointchess.nnue
```

## 5. Using the Network in PointChess

Set the network file via UCI:
```text
setoption name Use NNUE value true
setoption name EvalFile value nnue.pnet
position startpos
go depth 12
```
