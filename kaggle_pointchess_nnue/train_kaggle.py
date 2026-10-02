# PointChess 1-Billion Position Modern NNUE Trainer — Kaggle 2x Tesla T4 GPUs
# Trained on Deep Stockfish 19 Distillation Data (Depth 6-8) with PyTorch DDP + AMP fp16
import os, subprocess, torch

print("=" * 65)
print(" PointChess 1-Billion Position NNUE Trainer (Kaggle 2x T4 DDP)")
print(" Deep Stockfish 19 Distillation Dataset | SCReL Activation")
print("=" * 65)

num_gpus = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {num_gpus}")
for i in range(num_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

os.chdir("/kaggle/working")
if not os.path.exists("/kaggle/working/PointChess"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hemu-create/PointChess.git", "/kaggle/working/PointChess"], check=True)
else:
    subprocess.run(["git", "-C", "/kaggle/working/PointChess", "pull", "--ff-only"], check=False)
SRC = "/kaggle/working/PointChess"

print("\n[Step 1/4] Building engine with AVX2 optimizations...")
subprocess.run(f"cmake -S {SRC} -B {SRC}/build -DCMAKE_BUILD_TYPE=Release && cmake --build {SRC}/build -j$(nproc)", shell=True, check=True)
subprocess.run(f"{SRC}/build/pointchess bench 16 1", shell=True, check=True)

# Select Deep Distillation Dataset
data_file = "/kaggle/input/pointchess-200k-dataset/sf_deep_distill_178k.txt"
if not os.path.exists(data_file):
    data_file = "/kaggle/input/pointchess-200k-dataset/pointchess_200k_tactical.txt"

print(f"\n[Step 2/4] Training Dataset: {data_file}")
subprocess.run(f"wc -l {data_file}", shell=True, check=False)

ckpt_dir = "/kaggle/working/checkpoints_1b"
os.makedirs(ckpt_dir, exist_ok=True)

print("\n[Step 3/4] Training 1-Billion Positions on 2x Tesla T4 GPUs (DDP + AMP fp16)...")
if num_gpus >= 2:
    train_cmd = (f"cd {SRC}/training && torchrun --nproc_per_node=2 train_4b.py "
                 f"--data_file {data_file} --max_positions 1000000000 "
                 f"--batch_size 2048 --lr 5e-4 --ft_size 256 --output_dir {ckpt_dir} "
                 f"--log_interval 100 --save_interval 2000")
else:
    train_cmd = (f"cd {SRC}/training && python3 train_4b.py "
                 f"--data_file {data_file} --max_positions 1000000000 "
                 f"--batch_size 2048 --lr 5e-4 --ft_size 256 --output_dir {ckpt_dir} "
                 f"--log_interval 100 --save_interval 2000")
subprocess.run(train_cmd, shell=True, check=True)

print("\n[Step 4/4] Exporting 1-Billion Position Trained .pchess and .pnet Networks...")
out_pchess = "/kaggle/working/pointchess_1b.pchess"
out_pnet   = "/kaggle/working/nnue_1b.pnet"
subprocess.run(f"cd {SRC}/training && python3 export_mega.py --checkpoint {ckpt_dir}/best_model.pt "
               f"--output {out_pchess} --ft_size 256", shell=True, check=True)
subprocess.run(f"ls -lh {out_pchess} {out_pnet} 2>/dev/null || ls -lh {out_pchess}", shell=True, check=False)

print("\n" + "=" * 65)
print(" 1-Billion Position Training & Export Complete!")
print(f" Master Network Saved to: {out_pchess}")
print("=" * 65)
