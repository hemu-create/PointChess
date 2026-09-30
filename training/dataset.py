# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# High-Speed Modern HalfKAv2 Dataset Loader

import torch
from torch.utils.data import Dataset

# 11 Piece types from friendly perspective:
# 0: Friendly P, 1: Friendly N, 2: Friendly B, 3: Friendly R, 4: Friendly Q, 5: Friendly K
# 6: Enemy P, 7: Enemy N, 8: Enemy B, 9: Enemy R, 10: Enemy Q
PIECE_MAP_WHITE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10
}

PIECE_MAP_BLACK = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10
}

def fen_to_halfka_features(fen_str):
    """
    Parses a FEN string into White and Black Modern HalfKAv2 sparse feature lists.
    Returns: (w_features, b_features, stm)
    """
    parts = fen_str.strip().split()
    board_part = parts[0]
    stm_part = parts[1] if len(parts) > 1 else 'w'

    stm = 0 if stm_part == 'w' else 1

    ranks = board_part.split('/')
    pieces = []
    w_king_sq = 4
    b_king_sq = 60

    for r_idx, rank_str in enumerate(ranks):
        rank = 7 - r_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                sq = rank * 8 + file
                if ch == 'K':
                    w_king_sq = sq
                elif ch == 'k':
                    b_king_sq = sq
                pieces.append((ch, sq))
                file += 1

    w_k = w_king_sq
    b_k = b_king_sq ^ 56

    w_features = []
    b_features = []

    for ch, sq in pieces:
        # White perspective:
        w_p = PIECE_MAP_WHITE.get(ch)
        if w_p is not None:
            w_f = w_k * 704 + w_p * 64 + sq
            w_features.append(w_f)

        # Black perspective:
        b_p = PIECE_MAP_BLACK.get(ch)
        if b_p is not None:
            b_sq = sq ^ 56
            b_f = b_k * 704 + b_p * 64 + b_sq
            b_features.append(b_f)

    return w_features, b_features, stm

class ChessDataset(Dataset):
    def __init__(self, data_file=None, max_samples=None, synthetic_samples=10000):
        self.samples = []
        if data_file:
            self._load_file(data_file, max_samples)
        else:
            self._generate_synthetic(synthetic_samples)

    def _load_file(self, data_file, max_samples=None):
        count = 0
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('|')
                if len(parts) >= 3:
                    fen = parts[0].strip()
                    score = float(parts[1].strip())
                    result = float(parts[2].strip())
                    self.samples.append((fen, score, result))
                elif len(parts) == 1 and '/' in line:
                    self.samples.append((line, 0.0, 0.5))
                count += 1
                if max_samples and count >= max_samples:
                    break

    def _generate_synthetic(self, num_samples):
        fens = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 10.0, 0.5),
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 30.0, 0.55),
            ("r1bqk2r/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 4 7", 20.0, 0.52),
            ("r1bq1rk1/1pp1bppp/p1np1n2/4p3/B3P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 9", 25.0, 0.54),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 15.0, 0.51),
        ]
        for i in range(num_samples):
            fen, sc, res = fens[i % len(fens)]
            sc_noisy = sc + (i % 7 - 3) * 5.0
            self.samples.append((fen, sc_noisy, res))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, score, result = self.samples[idx]
        w_f, b_f, stm = fen_to_halfka_features(fen)
        return {
            'w_features': torch.tensor(w_f, dtype=torch.long),
            'b_features': torch.tensor(b_f, dtype=torch.long),
            'stm': torch.tensor(stm, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float32),
            'result': torch.tensor(result, dtype=torch.float32)
        }

def collate_halfkp(batch):
    w_feats = []
    b_feats = []
    w_offsets = [0]
    b_offsets = [0]
    stms = []
    scores = []
    results = []

    for item in batch:
        w_f = item['w_features']
        b_f = item['b_features']
        w_feats.append(w_f)
        b_feats.append(b_f)
        w_offsets.append(w_offsets[-1] + len(w_f))
        b_offsets.append(b_offsets[-1] + len(b_f))
        stms.append(item['stm'])
        scores.append(item['score'])
        results.append(item['result'])

    w_offsets.pop()
    b_offsets.pop()

    return {
        'w_features': torch.cat(w_feats),
        'w_offsets': torch.tensor(w_offsets, dtype=torch.long),
        'b_features': torch.cat(b_feats),
        'b_offsets': torch.tensor(b_offsets, dtype=torch.long),
        'stm': torch.stack(stms),
        'score': torch.stack(scores),
        'result': torch.stack(results)
    }
