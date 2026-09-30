# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Kaggle T4 x2 Single-Command Orchestrator (200k to 1M+ Games Scalable)

import os
import sys
import argparse
import subprocess
import torch

def main():
    parser = argparse.ArgumentParser(description="PointChess Kaggle T4 x2 Multi-Million Game Training Pipeline")
    parser.add_argument("--games", type=int, default=200000, help="Number of self-play games to generate (e.g. 200000 or 1000000)")
    parser.add_argument("--threads", type=int, default=12, help="CPU threads for parallel self-play generation")
    parser.add_argument("--depth", type=int, default=2, help="Search depth for self-play data generation")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per GPU")
    parser.add_argument("--data_file", type=str, default="data/dataset.txt", help="Path to save/load dataset")

    args = parser.parse_args()

    print("=" * 60)
    print(" PointChess Kaggle T4 x2 Training & Export Pipeline")
    print(f" Target Games: {args.games:,} | Training Epochs: {args.epochs}")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    print(f"Detected CUDA devices: {num_gpus}")
    for i in range(num_gpus):
        print(f" GPU {i}: {torch.cuda.get_device_name(i)}")

    # 1. High-Speed Multi-Threaded Self-Play Dataset Generation
    os.makedirs(os.path.dirname(os.path.abspath(args.data_file)), exist_ok=True)
    if not os.path.exists(args.data_file) or os.path.getsize(args.data_file) == 0:
        print(f"\n[Step 1/3] Generating {args.games:,} games using native multi-threaded C++ engine ({args.threads} threads)...")
        engine_bin = "../build/pointchess" if os.path.exists("../build/pointchess") else "../src/glaurung/pointchess"
        cmd_gen = [
            engine_bin, "genselfplay",
            str(args.games),
            str(args.threads),
            str(args.depth),
            args.data_file
        ]
        subprocess.run(cmd_gen, check=True)
    else:
        print(f"\n[Step 1/3] Found existing dataset at {args.data_file} ({os.path.getsize(args.data_file) / 1024 / 1024:.2f} MB)")

    # 2. Train with PyTorch DDP on 2x T4 GPUs
    print(f"\n[Step 2/3] Training NNUE on Kaggle 2x T4 GPUs (PyTorch DDP + AMP fp16, {args.epochs} epochs)...")
    if num_gpus >= 2:
        train_cmd = [
            "torchrun", "--nproc_per_node=2", "train.py",
            "--data_file", args.data_file,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", "1e-3",
            "--activation", "screl",
            "--amp"
        ]
    else:
        train_cmd = [
            sys.executable, "train.py",
            "--data_file", args.data_file,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", "1e-3",
            "--activation", "screl",
            "--amp"
        ]

    subprocess.run(train_cmd, check=True)

    # 3. Export to .pnet and .nnue
    print("\n[Step 3/3] Exporting trained network to .pnet and .nnue formats...")
    ckpt_path = "checkpoints/best_model.pt"
    export_cmd = [
        sys.executable, "export.py",
        "--checkpoint", ckpt_path if os.path.exists(ckpt_path) else "",
        "--output_pnet", "../nnue.pnet",
        "--output_nnue", "../pointchess.nnue"
    ]
    subprocess.run(export_cmd, check=True)

    print("\n" + "=" * 60)
    print(" Pipeline complete! Exported: nnue.pnet and pointchess.nnue")
    print(f" Trained on {args.games:,} self-play games successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
