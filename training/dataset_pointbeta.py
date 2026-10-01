# PointBeta Feature Extraction & Dataset Pipeline
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
from torch.utils.data import IterableDataset

# Dynamic non-pawn piece mapping (9 types):
# White perspective:
# Friendly: N(0), B(1), R(2), Q(3)
# Enemy: n(4), b(5), r(6), q(7), k(8)
DYNAMIC_MAP_WHITE = {
    'N': 0, 'B': 1, 'R': 2, 'Q': 3,
    'n': 4, 'b': 5, 'r': 6, 'q': 7, 'k': 8
}

DYNAMIC_MAP_BLACK = {
    'n': 0, 'b': 1, 'r': 2, 'q': 3,
    'N': 4, 'B': 5, 'R': 6, 'Q': 7, 'K': 8
}

def extract_pointbeta_features(fen_str):
    """
    Extracts the 4 specialized PointBeta feature sets for White and Black perspectives.
    Returns: (wk, bk, wp, bp, wr, br, ws, bs, stm)
    """
    parts = fen_str.strip().split()
    board_part = parts[0]
    stm_part = parts[1] if len(parts) > 1 else 'w'
    stm = 0 if stm_part == 'w' else 1

    ranks = board_part.split('/')
    pieces = []
    w_ksq = 4
    b_ksq = 60

    w_pawns = []
    b_pawns = []

    for r_idx, rank_str in enumerate(ranks):
        rank = 7 - r_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                sq = rank * 8 + file
                if ch == 'K': w_ksq = sq
                elif ch == 'k': b_ksq = sq
                elif ch == 'P': w_pawns.append(sq)
                elif ch == 'p': b_pawns.append(sq)
                pieces.append((ch, sq))
                file += 1

    w_k = w_ksq
    b_k = b_ksq ^ 56

    # 1. Dynamic King-Relative Features (36,864)
    wk_feats = []
    bk_feats = []
    for ch, sq in pieces:
        # White perspective
        pt_w = DYNAMIC_MAP_WHITE.get(ch)
        if pt_w is not None:
            wk_feats.append(w_k * 576 + pt_w * 64 + sq)
        # Black perspective
        pt_b = DYNAMIC_MAP_BLACK.get(ch)
        if pt_b is not None:
            bk_feats.append(b_k * 576 + pt_b * 64 + (sq ^ 56))

    # 2. King-Independent Pawn Features (384)
    wp_feats = []
    bp_feats = []
    for p_sq in w_pawns:
        wp_feats.append(p_sq)              # Friendly pawn
        bp_feats.append(64 + (p_sq ^ 56))  # Enemy pawn from black perspective
    for p_sq in b_pawns:
        bp_feats.append(p_sq ^ 56)         # Friendly pawn for black
        wp_feats.append(64 + p_sq)         # Enemy pawn for white

    # 3. Ray & Battery Interaction Features (464)
    wr_feats = []
    br_feats = []
    file_occupancy = [[] for _ in range(8)]
    for ch, sq in pieces:
        if ch in 'RQrq':
            f = sq % 8
            file_occupancy[f].append((ch, sq))

    for f in range(8):
        if len(file_occupancy[f]) >= 2:
            wr_feats.append(f * 16 + 1)
            br_feats.append(f * 16 + 1)

    if not wr_feats: wr_feats.append(0)
    if not br_feats: br_feats.append(0)

    # 4. 5x5 King Zone Safety Features (12,800)
    ws_feats = []
    bs_feats = []
    w_kr = w_ksq // 8
    w_kf = w_ksq % 8
    b_kr = b_ksq // 8
    b_kf = b_ksq % 8

    for ch, sq in pieces:
        r, f = sq // 8, sq % 8
        dr_w, df_w = r - w_kr, f - w_kf
        if -2 <= dr_w <= 2 and -2 <= df_w <= 2:
            off_w = (dr_w + 2) * 5 + (df_w + 2)
            pt_w = DYNAMIC_MAP_WHITE.get(ch, 0)
            ws_feats.append(w_k * 200 + (pt_w % 8) * 25 + off_w)

        dr_b, df_b = (sq ^ 56) // 8 - (b_ksq ^ 56) // 8, (sq % 8) - (b_ksq % 8)
        if -2 <= dr_b <= 2 and -2 <= df_b <= 2:
            off_b = (dr_b + 2) * 5 + (df_b + 2)
            pt_b = DYNAMIC_MAP_BLACK.get(ch, 0)
            bs_feats.append(b_k * 200 + (pt_b % 8) * 25 + off_b)

    if not ws_feats: ws_feats.append(0)
    if not bs_feats: bs_feats.append(0)

    return wk_feats, bk_feats, wp_feats, bp_feats, wr_feats, br_feats, ws_feats, bs_feats, stm

def collate_pointbeta(batch):
    def pack(field_w, field_b):
        w_f, b_f = [], []
        w_o, b_o = [0], [0]
        for item in batch:
            w_f.append(item[field_w])
            b_f.append(item[field_b])
            w_o.append(w_o[-1] + len(item[field_w]))
            b_o.append(b_o[-1] + len(item[field_b]))
        w_o.pop()
        b_o.pop()
        return (torch.cat(w_f), torch.tensor(w_o, dtype=torch.long),
                torch.cat(b_f), torch.tensor(b_o, dtype=torch.long))

    wk_f, wk_o, bk_f, bk_o = pack('wk', 'bk')
    wp_f, wp_o, bp_f, bp_o = pack('wp', 'bp')
    wr_f, wr_o, br_f, br_o = pack('wr', 'br')
    ws_f, ws_o, bs_f, bs_o = pack('ws', 'bs')

    stms   = torch.stack([item['stm'] for item in batch])
    scores = torch.stack([item['score'] for item in batch])
    wdls   = torch.stack([item['result'] for item in batch])

    return {
        'wk_f': wk_f, 'wk_o': wk_o, 'bk_f': bk_f, 'bk_o': bk_o,
        'wp_f': wp_f, 'wp_o': wp_o, 'bp_f': bp_f, 'bp_o': bp_o,
        'wr_f': wr_f, 'wr_o': wr_o, 'br_f': br_f, 'br_o': br_o,
        'ws_f': ws_f, 'ws_o': ws_o, 'bs_f': bs_f, 'bs_o': bs_o,
        'stm': stms, 'score': scores, 'result': wdls
    }

class PointBetaDataset(IterableDataset):
    def __init__(self, data_file):
        super().__init__()
        self.data_file = data_file

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_idx, line in enumerate(f):
                if worker_info is not None and line_idx % worker_info.num_workers != worker_info.id:
                    continue
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split('|')
                if len(parts) >= 3:
                    fen = parts[0].strip()
                    score = float(parts[1].strip())
                    result = float(parts[2].strip())
                    wk, bk, wp, bp, wr, br, ws, bs, stm = extract_pointbeta_features(fen)
                    yield {
                        'wk': torch.tensor(wk, dtype=torch.long),
                        'bk': torch.tensor(bk, dtype=torch.long),
                        'wp': torch.tensor(wp, dtype=torch.long),
                        'bp': torch.tensor(bp, dtype=torch.long),
                        'wr': torch.tensor(wr, dtype=torch.long),
                        'br': torch.tensor(br, dtype=torch.long),
                        'ws': torch.tensor(ws, dtype=torch.long),
                        'bs': torch.tensor(bs, dtype=torch.long),
                        'stm': torch.tensor(stm, dtype=torch.long),
                        'score': torch.tensor(score, dtype=torch.float32),
                        'result': torch.tensor(result, dtype=torch.float32)
                    }
