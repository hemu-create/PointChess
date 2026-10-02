# PointChess Mega-1024 on DEEP Stockfish labels — Kaggle 2x Tesla T4 GPUs
# Big net (92M params) + quality labels (depth 6-8 SF distillation).
import os, glob, subprocess, torch

print("=" * 65)
print(" PointChess Mega-1024 Deep-Label Trainer (Kaggle 2x T4)")
print("=" * 65)

num_gpus = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {num_gpus}")

os.chdir("/kaggle/working")
subprocess.run("rm -rf /kaggle/working/PointChess", shell=True)
subprocess.run("git clone --depth 1 https://github.com/hemu-create/PointChess.git /kaggle/working/PointChess", shell=True, check=True)
SRC = "/kaggle/working/PointChess"

print("\n[Step 1/4] Building engine...")
subprocess.run(f"cmake -S {SRC} -B {SRC}/build -DCMAKE_BUILD_TYPE=Release && cmake --build {SRC}/build -j$(nproc)", shell=True, check=True)
subprocess.run(f"{SRC}/build/pointchess bench 16 1", shell=True, check=False)

cands = [
    "/kaggle/input/pointchess-200k-dataset/sf_deep_distill_178k.txt",
    "/kaggle/input/pointchess-200k-dataset/pointchess_200k_tactical.txt",
    f"{SRC}/kaggle_dataset/sf_deep_distill_178k.txt",
]
data_file = next((c for c in cands if os.path.exists(c) and os.path.getsize(c) > 0), None)
if not data_file:
    txts = glob.glob("/kaggle/input/**/*.txt", recursive=True)
    data_file = txts[0] if txts else f"{SRC}/data/sf_deep_distill_big.txt"
print(f"\n[Step 2/4] Dataset: {data_file}")
subprocess.run(f"wc -l {data_file}", shell=True, check=False)

ckpt_dir = "/kaggle/working/checkpoints_1024deep"
os.makedirs(ckpt_dir, exist_ok=True)
print("\n[Step 3/4] Training Mega-1024 on deep labels (600M, lr 5e-4)...")
train_cmd = (f"cd {SRC}/training && python3 train_4b.py "
             f"--data_file {data_file} --max_positions 600000000 "
             f"--batch_size 2048 --lr 5e-4 --ft_size 1024 --output_dir {ckpt_dir} "
             f"--log_interval 100 --save_interval 2000")
subprocess.run(train_cmd, shell=True, check=True)

print("\n[Step 4/4] Exporting v3 .pchess (FT=1024)...")
out = "/kaggle/working/pointchess1024deep.pchess"
subprocess.run(f"cd {SRC}/training && python3 export_mega.py --checkpoint {ckpt_dir}/best_model.pt "
               f"--output {out} --ft_size 1024", shell=True, check=False)
subprocess.run(f"ls -lh {out}", shell=True, check=False)
print("DONE")
