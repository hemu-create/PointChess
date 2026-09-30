# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# PointChess Custom .pchess and .pnet Weight Exporter (Modern HalfKAv2)

import os
import sys
import argparse
import struct
import numpy as np
import torch

from model import (
    PointChessNNUE,
    HALF_KA_FEATURES,
    ACCUMULATOR_SIZE,
    L1_SIZE,
    L2_SIZE,
    WEIGHT_SCALE_L0,
    WEIGHT_SCALE_L1,
    WEIGHT_SCALE_L2,
    WEIGHT_SCALE_OUT
)

def export_model(checkpoint_path, output_pchess_path, output_pnet_path=None):
    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device("cpu")
    model = PointChessNNUE(activation="screl")

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        print("Checkpoint weights loaded successfully.")
    else:
        print("Using initialized model weights.")

    model.eval()

    # 1. Export custom PointChess .pchess format
    if output_pchess_path:
        print(f"Exporting to custom PointChess format: {output_pchess_path}")
        with torch.no_grad():
            ft_w = (model.feature_transformer.weight.detach().cpu().numpy() * WEIGHT_SCALE_L0).astype(np.int16)
            ft_b = (model.feature_bias.detach().cpu().numpy() * WEIGHT_SCALE_L0).astype(np.int16)

            l1_w = (model.l1.weight.detach().cpu().numpy() * WEIGHT_SCALE_L1).astype(np.int8)
            l1_b = (model.l1.bias.detach().cpu().numpy() * WEIGHT_SCALE_L1 * WEIGHT_SCALE_L0).astype(np.int32)

            l2_w = (model.l2.weight.detach().cpu().numpy() * WEIGHT_SCALE_L2).astype(np.int8)
            l2_b = (model.l2.bias.detach().cpu().numpy() * WEIGHT_SCALE_L2 * WEIGHT_SCALE_L1).astype(np.int32)

            out_w = (model.out.weight.detach().cpu().numpy().squeeze(0) * WEIGHT_SCALE_OUT).astype(np.int8)
            out_b = int(model.out.bias.detach().cpu().numpy()[0] * WEIGHT_SCALE_OUT * WEIGHT_SCALE_L2)

            with open(output_pchess_path, 'wb') as f:
                # Custom PointChess Magic Header (24 bytes)
                magic = b"POINTCHESS_V2_HALFKAV2\0\0"
                f.write(magic)

                # Feature transformer
                f.write(ft_w.tobytes())
                f.write(ft_b.tobytes())

                # L1
                f.write(l1_w.tobytes())
                f.write(l1_b.tobytes())

                # L2
                f.write(l2_w.tobytes())
                f.write(l2_b.tobytes())

                # Output
                f.write(out_w.tobytes())
                f.write(struct.pack('<i', out_b))

        print(f" -> Successfully exported {output_pchess_path} ({os.path.getsize(output_pchess_path) / 1024 / 1024:.2f} MB)")

    # 2. Export .pnet format (PyTorch container)
    if output_pnet_path:
        print(f"Exporting to .pnet format: {output_pnet_path}")
        torch.save({
            'format': 'pointchess_pnet_v2',
            'architecture': 'Modern_HalfKAv2_SCReL_2x256_32_32_1',
            'features': HALF_KA_FEATURES,
            'state_dict': model.state_dict(),
            'quantization': {
                'L0_scale': WEIGHT_SCALE_L0,
                'L1_scale': WEIGHT_SCALE_L1,
                'L2_scale': WEIGHT_SCALE_L2,
                'OUT_scale': WEIGHT_SCALE_OUT
            }
        }, output_pnet_path)
        print(f" -> Successfully exported {output_pnet_path} ({os.path.getsize(output_pnet_path) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained PointChess NNUE to custom .pchess and .pnet formats")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to input PyTorch checkpoint (.pt)")
    parser.add_argument("--output_pchess", type=str, default="pointchess.pchess", help="Output path for custom .pchess file")
    parser.add_argument("--output_pnet", type=str, default="nnue.pnet", help="Output path for .pnet file")

    args = parser.parse_args()
    export_model(args.checkpoint, args.output_pchess, args.output_pnet)
