# PointChess Mega-NNUE: parametrized HalfKAv2 FT width (256/1024/2048/4096/6144)
# SPDX-License-Identifier: GPL-3.0-or-later
import torch
import torch.nn as nn
import torch.nn.functional as F

HALF_KA_FEATURES = 45056
L1_SIZE = 32
L2_SIZE = 32

class SCReL(nn.Module):
    def forward(self, x):
        c = torch.clamp(x, 0.0, 1.0)
        return c * c

class MegaNNUE(nn.Module):
    def __init__(self, ft_size=256):
        super().__init__()
        assert ft_size in (256, 512, 1024, 2048, 4096, 6144), "unsupported FT width"
        self.ft_size = ft_size
        self.feature_transformer = nn.EmbeddingBag(HALF_KA_FEATURES, ft_size, mode='sum', sparse=False)
        self.feature_bias = nn.Parameter(torch.zeros(ft_size))
        self.act = SCReL()
        self.l1 = nn.Linear(ft_size * 2, L1_SIZE)
        self.l2 = nn.Linear(L1_SIZE, L2_SIZE)
        self.out = nn.Linear(L2_SIZE, 1)
        nn.init.normal_(self.feature_transformer.weight, std=0.02)
        nn.init.zeros_(self.feature_bias)
        nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        nn.init.constant_(self.l1.bias, 0.01)
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity='relu')
        nn.init.constant_(self.l2.bias, 0.01)
        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.zeros_(self.out.bias)

    def forward(self, w_f, w_o, b_f, b_o, stm):
        w_acc = self.feature_transformer(w_f, w_o) + self.feature_bias
        b_acc = self.feature_transformer(b_f, b_o) + self.feature_bias
        w_act, b_act = self.act(w_acc), self.act(b_acc)
        m = (stm == 0).unsqueeze(1).float()
        us = m * w_act + (1 - m) * b_act
        them = m * b_act + (1 - m) * w_act
        l0 = torch.cat([us, them], dim=-1)
        return self.out(self.act(self.l2(self.act(self.l1(l0))))).squeeze(-1)

    def compute_loss(self, pred_score, target_eval, target_wdl, lambda_val=0.8):
        wdl_loss = F.binary_cross_entropy_with_logits(pred_score / 400.0, target_wdl)
        eval_loss = F.mse_loss(pred_score, target_eval)
        return lambda_val * eval_loss + (1.0 - lambda_val) * (wdl_loss * 400.0)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    for ft in (256, 1024, 2048, 4096, 6144):
        m = MegaNNUE(ft)
        print(f"FT={ft}: params={m.param_count():,} ft_MB={HALF_KA_FEATURES*ft*2/1024/1024:.1f}MB int16")
