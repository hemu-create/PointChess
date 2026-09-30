# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Modern HalfKAv2 NNUE PyTorch Architecture with SCReL

import torch
import torch.nn as nn
import torch.nn.functional as F

# Modern HalfKAv2 Dimensions
NUM_PIECE_TYPES = 11  # 6 Friendly (P, N, B, R, Q, K), 5 Enemy (P, N, B, R, Q)
NUM_PIECE_SQUARES = 64
HALF_KA_PIECE_FEATURES = NUM_PIECE_TYPES * NUM_PIECE_SQUARES  # 704
HALF_KA_FEATURES = 64 * HALF_KA_PIECE_FEATURES  # 45,056
ACCUMULATOR_SIZE = 256
L1_SIZE = 32
L2_SIZE = 32

WEIGHT_SCALE_L0 = 256
WEIGHT_SCALE_L1 = 64
WEIGHT_SCALE_L2 = 64
WEIGHT_SCALE_OUT = 16

class SCReL(nn.Module):
    """Modern Squared Clipped ReLU (SCReL): clamp(x, 0, 1)^2"""
    def __init__(self, max_val=1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, x):
        clamped = torch.clamp(x, 0.0, self.max_val)
        return clamped * clamped

class PointChessNNUE(nn.Module):
    def __init__(self, activation="screl"):
        super().__init__()
        # Feature Transformer (45,056 -> 256)
        self.feature_transformer = nn.EmbeddingBag(
            HALF_KA_FEATURES,
            ACCUMULATOR_SIZE,
            mode='sum',
            sparse=False
        )
        self.feature_bias = nn.Parameter(torch.zeros(ACCUMULATOR_SIZE))

        # SCReL Activation
        self.act = SCReL(max_val=1.0)

        # Layer 1: 512 (256 W + 256 B) -> 32
        self.l1 = nn.Linear(ACCUMULATOR_SIZE * 2, L1_SIZE)
        # Layer 2: 32 -> 32
        self.l2 = nn.Linear(L1_SIZE, L2_SIZE)
        # Output: 32 -> 1
        self.out = nn.Linear(L2_SIZE, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.feature_bias)
        nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        nn.init.constant_(self.l1.bias, 0.01)
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity='relu')
        nn.init.constant_(self.l2.bias, 0.01)
        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.zeros_(self.out.bias)

    def forward(self, w_features, w_offsets, b_features, b_offsets, stm):
        w_acc = self.feature_transformer(w_features, w_offsets) + self.feature_bias
        b_acc = self.feature_transformer(b_features, b_offsets) + self.feature_bias

        w_act = self.act(w_acc)
        b_act = self.act(b_acc)

        stm_mask = (stm == 0).unsqueeze(1).float()
        us_act = stm_mask * w_act + (1.0 - stm_mask) * b_act
        them_act = stm_mask * b_act + (1.0 - stm_mask) * w_act

        l0 = torch.cat([us_act, them_act], dim=-1)
        l1 = self.act(self.l1(l0))
        l2 = self.act(self.l2(l1))
        out = self.out(l2)
        return out.squeeze(-1)

    def compute_loss(self, pred_score, target_eval, target_wdl, lambda_val=0.8):
        wdl_loss = F.binary_cross_entropy_with_logits(pred_score / 400.0, target_wdl)
        eval_loss = F.mse_loss(pred_score, target_eval)
        return lambda_val * eval_loss + (1.0 - lambda_val) * (wdl_loss * 400.0)
