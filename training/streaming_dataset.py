# PointChess - a chess engine in the Glaurung tradition.
# SPDX-License-Identifier: GPL-3.0-or-later
# Streaming Billion-Position Dataset Reader for PyTorch

import torch
from torch.utils.data import IterableDataset
from dataset import fen_to_halfka_features

class StreamingChessDataset(IterableDataset):
    def __init__(self, data_file, buffer_size=10000):
        super().__init__()
        self.data_file = data_file
        self.buffer_size = buffer_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if not os.path.exists(self.data_file):
            return

        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_idx, line in enumerate(f):
                # Worker sharding for multi-worker DataLoader
                if worker_info is not None:
                    if line_idx % worker_info.num_workers != worker_info.id:
                        continue

                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('|')
                if len(parts) >= 3:
                    fen = parts[0].strip()
                    score = float(parts[1].strip())
                    result = float(parts[2].strip())
                    weight = float(parts[3].strip()) if len(parts) >= 4 else 1.0

                    w_f, b_f, stm = fen_to_halfka_features(fen)
                    yield {
                        'w_features': torch.tensor(w_f, dtype=torch.long),
                        'b_features': torch.tensor(b_f, dtype=torch.long),
                        'stm': torch.tensor(stm, dtype=torch.long),
                        'score': torch.tensor(score, dtype=torch.float32),
                        'result': torch.tensor(result, dtype=torch.float32),
                        'weight': torch.tensor(weight, dtype=torch.float32)
                    }
