# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Ultra-Compact 32-Byte Binary Sharded Position Packer for 4 Billion Positions

import struct
import argparse
import os
import time

PIECE_MAP = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

def pack_fen_to_bytes(fen_line):
    """
    Packs a text FEN line 'FEN | score | wdl | weight' into a 32-byte binary struct:
    - 64-bit White Piece Bitboard (uint64)
    - 64-bit Black Piece Bitboard (uint64)
    - White King Sq (uint8), Black King Sq (uint8)
    - Side to Move (uint8), Active Pieces Count (uint8)
    - Score in Centipawns (int16)
    - WDL Outcome (int8: 0=Loss, 1=Draw, 2=Win)
    - Adversarial Weakness Weight (uint8: 1..255)
    - Reserved (8 bytes padding for 32-byte SIMD alignment)
    """
    parts = fen_line.strip().split('|')
    if len(parts) < 3:
        return None

    fen_str = parts[0].strip()
    score_cp = int(float(parts[1].strip()))
    wdl_val = float(parts[2].strip())
    weight = int(float(parts[3].strip())) if len(parts) >= 4 else 1

    wdl_byte = 2 if wdl_val >= 0.75 else (1 if wdl_val >= 0.25 else 0)
    score_cp = max(-32000, min(32000, score_cp))

    fen_parts = fen_str.split()
    board_str = fen_parts[0]
    stm_str = fen_parts[1] if len(fen_parts) > 1 else 'w'
    stm = 0 if stm_str == 'w' else 1

    white_bb = 0
    black_bb = 0
    w_king_sq = 4
    b_king_sq = 60
    piece_count = 0

    ranks = board_str.split('/')
    for r_idx, rank_str in enumerate(ranks):
        rank = 7 - r_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                sq = rank * 8 + file
                p_code = PIECE_MAP.get(ch, 0)
                if p_code <= 6:
                    white_bb |= (1 << sq)
                    if ch == 'K': w_king_sq = sq
                else:
                    black_bb |= (1 << sq)
                    if ch == 'k': b_king_sq = sq
                piece_count += 1
                file += 1

    # Format: <QQBBBBhbB8x (32 bytes total)
    packed = struct.pack(
        '<QQBBBBhbB7x',
        white_bb,
        black_bb,
        w_king_sq,
        b_king_sq,
        stm,
        piece_count,
        score_cp,
        wdl_byte,
        min(255, weight)
    )
    return packed

def convert_text_to_shards(input_file, output_prefix="shards/shard", shard_size=10000000):
    print(f"Packing {input_file} into 32-byte binary shards (Shard size: {shard_size:,})...")
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)

    shard_idx = 0
    record_count = 0
    out_file = None
    start_time = time.time()

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if record_count % shard_size == 0:
                if out_file:
                    out_file.close()
                shard_path = f"{output_prefix}_{shard_idx:04d}.bin"
                out_file = open(shard_path, 'wb')
                print(f" -> Started new binary shard: {shard_path}")
                shard_idx += 1

            packed = pack_fen_to_bytes(line)
            if packed:
                out_file.write(packed)
                record_count += 1

            if record_count % 1000000 == 0:
                elapsed = time.time() - start_time
                print(f"Packed {record_count:,} positions | Speed: {int(record_count/elapsed):,} pos/sec")

    if out_file:
        out_file.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f" Sharding Complete! Packed {record_count:,} positions into {shard_idx} shards in {total_time:.2f}s")
    print(f" Binary Size: {record_count * 32 / 1024 / 1024:.2f} MB")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointChess 4-Billion Binary Shard Packer")
    parser.add_argument("--input", type=str, required=True, help="Input text dataset file")
    parser.add_argument("--output_prefix", type=str, default="shards/pointchess_shard", help="Output shard prefix")
    parser.add_argument("--shard_size", type=int, default=10000000, help="Positions per binary shard")

    args = parser.parse_args()
    convert_text_to_shards(args.input, args.output_prefix, args.shard_size)
