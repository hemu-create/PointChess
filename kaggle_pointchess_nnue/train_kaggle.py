# PointChess 4-Billion Position HalfKAv2 Trainer — Kaggle T4 x2 (lean launcher)
# Clones fresh source from GitHub, builds engine, streams 200k dataset in a loop to 4B, exports .pchess
import os, subprocess, torch

print("=" * 65)
print(" PointChess 4B NNUE Trainer (HalfKAv2 SCReL | 2x T4 DDP + AMP)")
print("=" * 65)
ng = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {ng}")
for i in range(ng):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

os.chdir("/kaggle/working")
if not os.path.exists("/kaggle/working/PointChess"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hemu-create/PointChess.git", "/kaggle/working/PointChess"], check=True)
else:
    subprocess.run(["git", "-C", "/kaggle/working/PointChess", "pull", "--ff-only"], check=False)
SRC = "/kaggle/working/PointChess"

print("\n[1/4] Building engine...")
subprocess.run(f"cmake -S {SRC} -B {SRC}/build -DCMAKE_BUILD_TYPE=Release && cmake --build {SRC}/build -j$(nproc)", shell=True, check=True)
subprocess.run(f"{SRC}/build/pointchess bench 64 1 8", shell=True, check=True)

data_file = "/kaggle/input/pointchess-200k-dataset/pointchess_200k_tactical.txt"
if not os.path.exists(data_file):
    alt = f"{SRC}/kaggle_dataset/pointchess_200k_tactical.txt"
    data_file = alt if os.path.exists(alt) else data_file
print(f"\n[2/4] Dataset: {data_file}")
if os.path.exists(data_file):
    print(f"  lines: ", flush=True)
    subprocess.run(f"wc -l {data_file}", shell=True, check=False)
else:
    print("  WARNING: dataset file not found, generating fallback selfplay...")
    data_file = "/kaggle/working/selfplay_fb.txt"
    subprocess.run(f"{SRC}/build/pointchess genselfplay 50000 $(nproc) 2 {data_file}", shell=True, check=True)

ckpt_dir = "/kaggle/working/checkpoints_4b"
os.makedirs(ckpt_dir, exist_ok=True)
print("\n[3/4] Training to 4,000,000,000 positions (cycling dataset, DDP+AMP)...")
if ng >= 2:
    cmd = (f"cd {SRC}/training && torchrun --nproc_per_node=2 train_4b.py "
           f"--data_file {data_file} --max_positions 4000000000 "
           f"--batch_size 4096 --lr 2e-3 --output_dir {ckpt_dir}")
else:
    cmd = (f"cd {SRC}/training && python3 train_4b.py "
           f"--data_file {data_file} --max_positions 4000000000 "
           f"--batch_size 4096 --lr 2e-3 --output_dir {ckpt_dir}")
subprocess.run(cmd, shell=True, check=True)

print("\n[4/4] Exporting .pchess + .pnet...")
subprocess.run(f"cd {SRC}/training && python3 export.py --checkpoint {ckpt_dir}/best_model.pt "
               f"--output_pchess /kaggle/working/pointchess.pchess --output_pnet /kaggle/working/nnue.pnet",
               shell=True, check=True)
subprocess.run("ls -lh /kaggle/working/pointchess.pchess /kaggle/working/nnue.pnet", shell=True, check=False)
print("4B training + export complete.")
