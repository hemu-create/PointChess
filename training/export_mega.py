# PointChess Mega-NNUE exporter: writes .pchess v3 with FT-size header.
# Magic: b"POINTCHESS_V3_HALFKAV2\0\0" (24B) + int32 ft_size, then variable-size tensors.
# SPDX-License-Identifier: GPL-3.0-or-later
import os, struct, argparse
import numpy as np
import torch
from model_mega import MegaNNUE

V3_MAGIC = b"POINTCHESS_V3_HALFKAV2\0\0"
L0, L1, L2, OUT = 256, 64, 64, 16

def export_mega(checkpoint, out_path, ft_size):
    model = MegaNNUE(ft_size=ft_size)
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("checkpoint loaded")
    model.eval()
    with torch.no_grad():
        ft_w = (model.feature_transformer.weight.detach().cpu().numpy() * L0).astype(np.int16)
        ft_b = (model.feature_bias.detach().cpu().numpy() * L0).astype(np.int16)
        l1_w = (model.l1.weight.detach().cpu().numpy() * L1).astype(np.int8)
        l1_b = (model.l1.bias.detach().cpu().numpy() * L1 * L0).astype(np.int32)
        l2_w = (model.l2.weight.detach().cpu().numpy() * L2).astype(np.int8)
        l2_b = (model.l2.bias.detach().cpu().numpy() * L2 * L1).astype(np.int32)
        out_w = (model.out.weight.detach().cpu().numpy().squeeze(0) * OUT).astype(np.int8)
        out_b = int(model.out.bias.detach().cpu().numpy()[0] * OUT * L2)
    assert ft_w.shape == (45056, ft_size), ft_w.shape
    with open(out_path, "wb") as f:
        f.write(V3_MAGIC)
        f.write(struct.pack("<i", ft_size))
        f.write(ft_w.tobytes()); f.write(ft_b.tobytes())
        f.write(l1_w.tobytes()); f.write(l1_b.tobytes())
        f.write(l2_w.tobytes()); f.write(l2_b.tobytes())
        f.write(out_w.tobytes()); f.write(struct.pack("<i", out_b))
    print(f"wrote {out_path} ({os.path.getsize(out_path)/1024/1024:.1f} MB, FT={ft_size})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default="pointchess1024.pchess")
    ap.add_argument("--ft_size", type=int, default=1024)
    a = ap.parse_args()
    export_mega(a.checkpoint, a.output, a.ft_size)
