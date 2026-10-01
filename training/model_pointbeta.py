# PointBeta NNUE: Factorized Quad-Accumulator Architecture (4096 L0 Dimension)
# SPDX-License-Identifier: GPL-3.0-or-later
import torch
import torch.nn as nn
import torch.nn.functional as F

# Subspace Feature Dimensions
NUM_KING_FEATURES   = 64 * 9 * 64   # 36,864 (Dynamic King-Relative non-pawns)
NUM_PAWN_FEATURES   = 2 * (64 + 64 + 64) # 384 (King-Independent Pawns)
NUM_PAIR_FEATURES   = 8 * 16 + 8 * 16 + 26 * 8 # 464 (Rays & Batteries)
NUM_SAFETY_FEATURES = 64 * 8 * 25   # 12,800 (5x5 King Ring Zones)

# Accumulator Sizes (per perspective)
FT_KING_SIZE   = 1024
FT_PAWN_SIZE   = 512
FT_PAIR_SIZE   = 256
FT_SAFETY_SIZE = 256

L0_SIZE = (FT_KING_SIZE + FT_PAWN_SIZE + FT_PAIR_SIZE + FT_SAFETY_SIZE) * 2 # 4096
L1_SIZE = 1024
L2_SIZE = 256
L3_SIZE = 64

class SCReL(nn.Module):
    """Squared Clipped ReLU: clamp(x, 0, 1)^2"""
    def forward(self, x):
        c = torch.clamp(x, 0.0, 1.0)
        return c * c

class PointBetaNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Quad Accumulator Feature Transformers
        self.ft_king   = nn.EmbeddingBag(NUM_KING_FEATURES,   FT_KING_SIZE,   mode='sum', sparse=False)
        self.ft_pawn   = nn.EmbeddingBag(NUM_PAWN_FEATURES,   FT_PAWN_SIZE,   mode='sum', sparse=False)
        self.ft_pair   = nn.EmbeddingBag(NUM_PAIR_FEATURES,   FT_PAIR_SIZE,   mode='sum', sparse=False)
        self.ft_safety = nn.EmbeddingBag(NUM_SAFETY_FEATURES, FT_SAFETY_SIZE, mode='sum', sparse=False)

        self.bias_king   = nn.Parameter(torch.zeros(FT_KING_SIZE))
        self.bias_pawn   = nn.Parameter(torch.zeros(FT_PAWN_SIZE))
        self.bias_pair   = nn.Parameter(torch.zeros(FT_PAIR_SIZE))
        self.bias_safety = nn.Parameter(torch.zeros(FT_SAFETY_SIZE))

        self.act = SCReL()

        # 2. Deep Mixing Network (4096 -> 1024 -> 256 -> 64 -> 1)
        self.l1  = nn.Linear(L0_SIZE, L1_SIZE)
        self.l2  = nn.Linear(L1_SIZE, L2_SIZE)
        self.l3  = nn.Linear(L2_SIZE, L3_SIZE)
        self.out = nn.Linear(L3_SIZE, 1)

        self._init_weights()

    def _init_weights(self):
        for ft in (self.ft_king, self.ft_pawn, self.ft_pair, self.ft_safety):
            nn.init.normal_(ft.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias_king)
        nn.init.zeros_(self.bias_pawn)
        nn.init.zeros_(self.bias_pair)
        nn.init.zeros_(self.bias_safety)

        for layer in (self.l1, self.l2, self.l3):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.01)

        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.zeros_(self.out.bias)

    def forward(self, wk_f, wk_o, bk_f, bk_o,
                wp_f, wp_o, bp_f, bp_o,
                wr_f, wr_o, br_f, br_o,
                ws_f, ws_o, bs_f, bs_o,
                stm):
        # Accumulators for White perspective
        w_k = self.act(self.ft_king(wk_f, wk_o)     + self.bias_king)
        w_p = self.act(self.ft_pawn(wp_f, wp_o)     + self.bias_pawn)
        w_r = self.act(self.ft_pair(wr_f, wr_o)     + self.bias_pair)
        w_s = self.act(self.ft_safety(ws_f, ws_o)   + self.bias_safety)
        w_rep = torch.cat([w_k, w_p, w_r, w_s], dim=-1)

        # Accumulators for Black perspective
        b_k = self.act(self.ft_king(bk_f, bk_o)     + self.bias_king)
        b_p = self.act(self.ft_pawn(bp_f, bp_o)     + self.bias_pawn)
        b_r = self.act(self.ft_pair(br_f, br_o)     + self.bias_pair)
        b_s = self.act(self.ft_safety(bs_f, bs_o)   + self.bias_safety)
        b_rep = torch.cat([b_k, b_p, b_r, b_s], dim=-1)

        # Perspective selection
        stm_mask = (stm == 0).unsqueeze(1).float()
        us   = stm_mask * w_rep + (1.0 - stm_mask) * b_rep
        them = stm_mask * b_rep + (1.0 - stm_mask) * w_rep

        l0 = torch.cat([us, them], dim=-1) # 4096
        l1 = self.act(self.l1(l0))         # 1024
        l2 = self.act(self.l2(l1))         # 256
        l3 = self.act(self.l3(l2))         # 64
        out = self.out(l3)
        return out.squeeze(-1)

    def compute_loss(self, pred_score, target_eval, target_wdl, lambda_val=0.8):
        wdl_loss = F.binary_cross_entropy_with_logits(pred_score / 400.0, target_wdl)
        eval_loss = F.mse_loss(pred_score, target_eval)
        return lambda_val * eval_loss + (1.0 - lambda_val) * (wdl_loss * 400.0)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    m = PointBetaNNUE()
    print(f"PointBeta-4096 initialized successfully! Total Parameters: {m.param_count():,}")
