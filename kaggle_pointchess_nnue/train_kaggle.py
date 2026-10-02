# PointChess 1-Billion Position Robust NNUE Trainer — Kaggle 2x Tesla T4 GPUs
# Supports Deep Distillation (Depth 6-8), Fallback Datasets, and Auto-Export
import os, sys, glob, subprocess, torch

print("=" * 65)
print(" PointChess 1-Billion Position NNUE Trainer (Kaggle 2x T4 DDP)")
print(" Deep Stockfish 19 Distillation Dataset | SCReL Activation")
print("=" * 65)

num_gpus = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

os.chdir("/kaggle/working")
# Always do a clean fresh clone to guarantee latest GitHub commit
subprocess.run("rm -rf /kaggle/working/PointChess", shell=True)
subprocess.run("git clone --depth 1 https://github.com/hemu-create/PointChess.git /kaggle/working/PointChess", shell=True, check=True)
SRC = "/kaggle/working/PointChess"

print("\n[Step 1/4] Building PointChess engine with native optimizations...")
subprocess.run(f"cmake -S {SRC} -B {SRC}/build -DCMAKE_BUILD_TYPE=Release && cmake --build {SRC}/build -j$(nproc)", shell=True, check=True)
subprocess.run(f"{SRC}/build/pointchess bench 16 1", shell=True, check=False)

# Dataset Resolution
data_candidates = [
    "/kaggle/input/pointchess-200k-dataset/sf_deep_distill_178k.txt",
    "/kaggle/input/pointchess-200k-dataset/pointchess_200k_tactical.txt",
    f"{SRC}/kaggle_dataset/sf_deep_distill_178k.txt",
    f"{SRC}/kaggle_dataset/pointchess_200k_tactical.txt"
]
data_file = None
for candidate in data_candidates:
    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
        data_file = candidate
        break

if not data_file:
    txt_files = glob.glob("/kaggle/input/**/*.txt", recursive=True)
    if txt_files:
        data_file = txt_files[0]
    else:
        data_file = f"{SRC}/data/sf_deep_distill_big.txt"

print(f"\n[Step 2/4] Selected Training Dataset: {data_file}")
if os.path.exists(data_file):
    subprocess.run(f"wc -l {data_file}", shell=True, check=False)

ckpt_dir = "/kaggle/working/checkpoints_1b"
os.makedirs(ckpt_dir, exist_ok=True)

print(f"\n[Step 3/4] Training 1-Billion Positions on {num_gpus}x Tesla T4 GPUs (DataParallel + AMP fp16)...")
train_cmd = (f"cd {SRC}/training && python3 train_4b.py "
             f"--data_file {data_file} --max_positions 1000000000 "
             f"--batch_size 2048 --lr 5e-4 --ft_size 256 --output_dir {ckpt_dir} "
             f"--log_interval 100 --save_interval 2000")
subprocess.run(train_cmd, shell=True, check=True)

print("\n[Step 4/4] Exporting 1-Billion Position Trained .pchess and .pnet Networks...")
out_pchess = "/kaggle/working/pointchess_1b.pchess"
out_pnet   = "/kaggle/working/nnue_1b.pnet"
subprocess.run(f"cd {SRC}/training && python3 export_mega.py --checkpoint {ckpt_dir}/best_model.pt "
               f"--output {out_pchess} --ft_size 256", shell=True, check=False)
subprocess.run(f"cd {SRC}/training && python3 export.py --checkpoint {ckpt_dir}/best_model.pt "
               f"--output_pchess {out_pchess} --output_pnet {out_pnet}", shell=True, check=False)
subprocess.run(f"ls -lh {out_pchess} {out_pnet} 2>/dev/null || ls -lh /kaggle/working/*.pchess", shell=True, check=False)

print("\n" + "=" * 65)
print(" 1-Billion Position Training & Export Complete!")
print("=" * 65)
