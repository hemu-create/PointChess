# PointBeta Model Exporter to Custom .pchess Binary
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import struct
import argparse
import numpy as np
import torch
from model_pointbeta import PointBetaNNUE

MAGIC_HEADER = b"POINTBETA_V1_4096\0\0\0\0\0\0\0" # 24 bytes

L0_SCALE = 256
L1_SCALE = 64
L2_SCALE = 64
L3_SCALE = 64
OUT_SCALE = 16

def export_pointbeta(checkpoint_path, output_path):
    print(f"Loading PointBeta model: {checkpoint_path}")
    model = PointBetaNNUE()
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        print("Checkpoint weights loaded successfully.")
    else:
        print("Using initialized model weights.")

    model.eval()

    with torch.no_grad():
        wk_w = (model.ft_king.weight.detach().cpu().numpy() * L0_SCALE).astype(np.int16)
        wk_b = (model.bias_king.detach().cpu().numpy() * L0_SCALE).astype(np.int16)

        wp_w = (model.ft_pawn.weight.detach().cpu().numpy() * L0_SCALE).astype(np.int16)
        wp_b = (model.bias_pawn.detach().cpu().numpy() * L0_SCALE).astype(np.int16)

        wr_w = (model.ft_pair.weight.detach().cpu().numpy() * L0_SCALE).astype(np.int16)
        wr_b = (model.bias_pair.detach().cpu().numpy() * L0_SCALE).astype(np.int16)

        ws_w = (model.ft_safety.weight.detach().cpu().numpy() * L0_SCALE).astype(np.int16)
        ws_b = (model.bias_safety.detach().cpu().numpy() * L0_SCALE).astype(np.int16)

        l1_w = (model.l1.weight.detach().cpu().numpy() * L1_SCALE).astype(np.int8)
        l1_b = (model.l1.bias.detach().cpu().numpy() * L1_SCALE * L0_SCALE).astype(np.int32)

        l2_w = (model.l2.weight.detach().cpu().numpy() * L2_SCALE).astype(np.int8)
        l2_b = (model.l2.bias.detach().cpu().numpy() * L2_SCALE * L1_SCALE).astype(np.int32)

        l3_w = (model.l3.weight.detach().cpu().numpy() * L3_SCALE).astype(np.int8)
        l3_b = (model.l3.bias.detach().cpu().numpy() * L3_SCALE * L2_SCALE).astype(np.int32)

        out_w = (model.out.weight.detach().cpu().numpy().squeeze(0) * OUT_SCALE).astype(np.int8)
        out_b = int(model.out.bias.detach().cpu().numpy()[0] * OUT_SCALE * L3_SCALE)

    with open(output_path, 'wb') as f:
        f.write(MAGIC_HEADER)
        f.write(wk_w.tobytes()); f.write(wk_b.tobytes())
        f.write(wp_w.tobytes()); f.write(wp_b.tobytes())
        f.write(wr_w.tobytes()); f.write(wr_b.tobytes())
        f.write(ws_w.tobytes()); f.write(ws_b.tobytes())

        f.write(l1_w.tobytes()); f.write(l1_b.tobytes())
        f.write(l2_w.tobytes()); f.write(l2_b.tobytes())
        f.write(l3_w.tobytes()); f.write(l3_b.tobytes())

        f.write(out_w.tobytes()); f.write(struct.pack('<i', out_b))

    print(f"Successfully exported PointBeta network to: {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PointBeta-4096 NNUE to .pchess format")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_pointbeta/best_model.pt")
    parser.add_argument("--output", type=str, default="pointbeta.pchess")
    args = parser.parse_args()
    export_pointbeta(args.checkpoint, args.output)
