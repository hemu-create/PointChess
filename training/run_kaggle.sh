#!/bin/bash
# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Kaggle T4 x2 One-Click Execution Script

set -e

GAMES=${1:-200000}
EPOCHS=${2:-25}
THREADS=${3:-4}

echo "============================================================"
echo " PointChess Kaggle 2x T4 Training Pipeline"
echo " Games: $GAMES | Epochs: $EPOCHS | CPU Threads: $THREADS"
echo "============================================================"

# Check GPUs
python3 -c "import torch; print(f'CUDA GPUs detected: {torch.cuda.device_count()}')"

# Build engine
echo "Building PointChess engine..."
cd ..
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ../training

# Run training orchestrator
echo "Starting multi-million game dataset generation and 2x T4 GPU DDP training..."
python3 kaggle_run.py --games "$GAMES" --threads "$THREADS" --epochs "$EPOCHS" --batch_size 1024

echo "============================================================"
echo " Kaggle Pipeline Complete! Created nnue.pnet & pointchess.nnue"
echo "============================================================"
