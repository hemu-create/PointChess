# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# High-Throughput 4-Billion Position PyTorch DDP Trainer for Kaggle 2x T4 GPUs

import os
import sys
import argparse
import time
import math
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import PointChessNNUE
from streaming_dataset import StreamingChessDataset
from dataset import collate_halfkp

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank, is_distributed

def cleanup_ddp(is_distributed):
    if is_distributed:
        dist.destroy_process_group()

def train_4b(args):
    rank, world_size, local_rank, is_distributed = setup_ddp()
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print("=" * 65)
        print(" PointChess 4-Billion Position High-Throughput Trainer")
        print(f" GPUs: {world_size}x Tesla T4 | Device: {device} | DDP: {is_distributed}")
        print(f" Batch Size: {args.batch_size:,} | Total Target: {args.max_positions:,} positions")
        print(f" Mixed Precision: AMP fp16 | Activation: SCReL")
        print("=" * 65)

    # 1. Model
    model = PointChessNNUE(activation="screl").to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    raw_model = model.module if is_distributed else model

    # 2. Optimizer & LR Scheduler (OneCycleLR with warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # 3. Data stream
    dataset = StreamingChessDataset(args.data_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_halfkp
    )

    os.makedirs(args.output_dir, exist_ok=True)
    positions_seen = 0
    batch_idx = 0
    start_time = time.time()
    last_log_time = time.time()
    accum_loss = 0.0

    model.train()
    for batch in loader:
        w_f = batch['w_features'].to(device, non_blocking=True)
        w_off = batch['w_offsets'].to(device, non_blocking=True)
        b_f = batch['b_features'].to(device, non_blocking=True)
        b_off = batch['b_offsets'].to(device, non_blocking=True)
        stm = batch['stm'].to(device, non_blocking=True)
        target_eval = batch['score'].to(device, non_blocking=True)
        target_wdl = batch['result'].to(device, non_blocking=True)
        weights = batch.get('weight', torch.ones_like(target_eval)).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            pred_score = model(w_f, w_off, b_f, b_off, stm)
            # Weighted loss prioritizing tactical weaknesses and blunders
            raw_loss = raw_model.compute_loss(pred_score, target_eval, target_wdl, lambda_val=0.8)
            weighted_loss = (raw_loss * weights).mean()

        scaler.scale(weighted_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        curr_batch_size = len(target_eval)
        positions_seen += curr_batch_size * world_size
        batch_idx += 1
        accum_loss += weighted_loss.item()

        # Real-time streaming metrics logging
        if batch_idx % args.log_interval == 0 and is_main:
            now = time.time()
            speed = (args.log_interval * args.batch_size * world_size) / (now - last_log_time)
            avg_loss = accum_loss / args.log_interval
            print(f"[{positions_seen:,}/{args.max_positions:,} pos] | Loss: {avg_loss:.4f} | Throughput: {int(speed):,} pos/sec | Elapsed: {int(now - start_time)}s")
            accum_loss = 0.0
            last_log_time = now

        # Periodic checkpoint
        if batch_idx % args.save_interval == 0 and is_main:
            ckpt_path = os.path.join(args.output_dir, f"model_{positions_seen // 1000000}M.pt")
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({'model_state_dict': raw_model.state_dict(), 'positions_seen': positions_seen}, ckpt_path)
            torch.save({'model_state_dict': raw_model.state_dict(), 'positions_seen': positions_seen}, best_path)
            print(f" -> Checkpointed to {ckpt_path}")

        if positions_seen >= args.max_positions:
            break

    if is_main:
        best_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save({'model_state_dict': raw_model.state_dict(), 'positions_seen': positions_seen}, best_path)
        print("\n" + "=" * 65)
        print(f" 4B Training Complete! Processed {positions_seen:,} positions in {time.time() - start_time:.1f}s")
        print("=" * 65)

    cleanup_ddp(is_distributed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointChess 4-Billion Position PyTorch DDP Trainer")
    parser.add_argument("--data_file", type=str, required=True, help="Path to input dataset file or shards")
    parser.add_argument("--max_positions", type=int, default=4000000000, help="Total target positions to train on")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=100, help="Batches between logs")
    parser.add_argument("--save_interval", type=int, default=2000, help="Batches between checkpoints")
    parser.add_argument("--output_dir", type=str, default="checkpoints_4b", help="Output directory for checkpoints")

    args = parser.parse_args()
    train_4b(args)
