# PointChess NNUE Architecture

## 1. Feature Representation (HalfKP)

Features are indexed from the perspective of both kings:
- 64 possible king squares
- 10 non-king piece types (5 friendly pieces: P, N, B, R, Q and 5 enemy pieces: p, n, b, r, q)
- 64 squares per piece
- Total features per perspective: `64 * 10 * 64 = 40,960` features.

Formula:
```text
Feature_Index = King_Square * 640 + Piece_Type * 64 + Piece_Square
```

## 2. Network Topology

```text
Input Features (40,960 sparse)
   │
   ▼
Feature Transformer (Accumulator: 256 per perspective)
   │
   ▼
Activation: ClippedReLU (0..127) or SCReL (clamp(x, 0, 1)^2)
   │
   ▼
Concatenation: [Us_Acc (256), Them_Acc (256)] = 512
   │
   ▼
Linear Layer 1: 512 -> 32 (Int8 weights)
   │
   ▼
Linear Layer 2: 32 -> 32 (Int8 weights)
   │
   ▼
Output Layer: 32 -> 1 (Int8 weights -> Centipawn score)
```

## 3. Quantization & Formats

- **`.pnet`**: Standard PyTorch container format containing full precision floating-point tensors and state dict, compatible with modern Stockfish / PyTorch loaders.
- **`.nnue`**: Quantized int16 / int8 binary file with zero parsing overhead for instant CPU memory loading.
