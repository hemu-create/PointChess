# PointBeta High-Speed PyTorch Trainer
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import argparse
import time
import torch
from torch.utils.data import DataLoader

from model_pointbeta import PointBetaNNUE
from dataset_pointbeta import PointBetaDataset, collate_pointbeta

def train_pointbeta(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f" PointBeta-4096 Factorized Quad-Accumulator Trainer")
    print(f" Device: {device} | Batch Size: {args.batch_size:,} | Max Positions: {args.max_positions:,}")
    print("=" * 65)

    model = PointBetaNNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    dataset = PointBetaDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, collate_fn=collate_pointbeta)

    os.makedirs(args.output_dir, exist_ok=True)
    positions_seen = 0
    batch_idx = 0
    epoch = 0
    start_time = time.time()
    last_log_time = time.time()
    accum_loss = 0.0

    model.train()
    while positions_seen < args.max_positions:
        epoch += 1
        for batch in loader:
            wk_f = batch['wk_f'].to(device, non_blocking=True); wk_o = batch['wk_o'].to(device, non_blocking=True)
            bk_f = batch['bk_f'].to(device, non_blocking=True); bk_o = batch['bk_o'].to(device, non_blocking=True)
            wp_f = batch['wp_f'].to(device, non_blocking=True); wp_o = batch['wp_o'].to(device, non_blocking=True)
            bp_f = batch['bp_f'].to(device, non_blocking=True); bp_o = batch['bp_o'].to(device, non_blocking=True)
            wr_f = batch['wr_f'].to(device, non_blocking=True); wr_o = batch['wr_o'].to(device, non_blocking=True)
            br_f = batch['br_f'].to(device, non_blocking=True); br_o = batch['br_o'].to(device, non_blocking=True)
            ws_f = batch['ws_f'].to(device, non_blocking=True); ws_o = batch['ws_o'].to(device, non_blocking=True)
            bs_f = batch['bs_f'].to(device, non_blocking=True); bs_o = batch['bs_o'].to(device, non_blocking=True)

            stm = batch['stm'].to(device, non_blocking=True)
            target_eval = torch.clamp(batch['score'].to(device, non_blocking=True), -2000.0, 2000.0)
            target_wdl  = batch['result'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(wk_f, wk_o, bk_f, bk_o,
                             wp_f, wp_o, bp_f, bp_o,
                             wr_f, wr_o, br_f, br_o,
                             ws_f, ws_o, bs_f, bs_o,
                             stm)
                loss = model.compute_loss(pred, target_eval, target_wdl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            curr_batch_size = len(target_eval)
            positions_seen += curr_batch_size
            batch_idx += 1
            accum_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                now = time.time()
                speed = (args.log_interval * args.batch_size) / (now - last_log_time)
                avg_loss = accum_loss / args.log_interval
                print(f"[{positions_seen:,}/{args.max_positions:,} pos] | Loss: {avg_loss:.4f} | Speed: {int(speed):,} pos/sec | Elapsed: {int(now - start_time)}s")
                accum_loss = 0.0
                last_log_time = now

            if batch_idx % args.save_interval == 0:
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({'model_state_dict': model.state_dict(), 'positions_seen': positions_seen}, best_path)

            if positions_seen >= args.max_positions:
                break

    best_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save({'model_state_dict': model.state_dict(), 'positions_seen': positions_seen}, best_path)
    print("\n" + "=" * 65)
    print(f" PointBeta Training Complete! Processed {positions_seen:,} positions in {time.time() - start_time:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointBeta-4096 PyTorch NNUE Trainer")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--max_positions", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="checkpoints_pointbeta")
    args = parser.parse_args()
    train_pointbeta(args)
