# PointChess Mega-1024 HalfKAv2 Trainer — Kaggle T4 x2 (lean launcher)
# Clones fresh source, builds AVX2 engine, trains MegaNNUE FT=1024 with clamped
# targets, exports v3 .pchess (88MB). 600M-position budget fits ~9h on 2x T4.
import os, subprocess, torch

print("=" * 65)
print(" PointChess Mega-1024 Trainer (HalfKAv2 SCReL | 2x T4 DDP + AMP)")
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

print("\n[1/4] Building engine (AVX2 + eval cache)...")
subprocess.run(f"cmake -S {SRC} -B {SRC}/build -DCMAKE_BUILD_TYPE=Release && cmake --build {SRC}/build -j$(nproc)", shell=True, check=True)
subprocess.run(f"{SRC}/build/pointchess bench 16 1", shell=True, check=True)

data_file = "/kaggle/input/pointchess-200k-dataset/pointchess_200k_tactical.txt"
if not os.path.exists(data_file):
    alt = f"{SRC}/kaggle_dataset/pointchess_200k_tactical.txt"
    data_file = alt if os.path.exists(alt) else data_file
print(f"\n[2/4] Dataset: {data_file}")
subprocess.run(f"wc -l {data_file}", shell=True, check=False)

ckpt_dir = "/kaggle/working/checkpoints_1024"
os.makedirs(ckpt_dir, exist_ok=True)
print("\n[3/4] Training Mega-1024 to 600,000,000 positions (cycling, DDP+AMP, lr 5e-4)...")
if ng >= 2:
    cmd = (f"cd {SRC}/training && torchrun --nproc_per_node=2 train_4b.py "
           f"--data_file {data_file} --max_positions 600000000 "
           f"--batch_size 2048 --lr 5e-4 --ft_size 1024 --output_dir {ckpt_dir}")
else:
    cmd = (f"cd {SRC}/training && python3 train_4b.py "
           f"--data_file {data_file} --max_positions 600000000 "
           f"--batch_size 2048 --lr 5e-4 --ft_size 1024 --output_dir {ckpt_dir}")
subprocess.run(cmd, shell=True, check=True)

print("\n[4/4] Exporting v3 .pchess (FT=1024)...")
subprocess.run(f"cd {SRC}/training && python3 export_mega.py --checkpoint {ckpt_dir}/best_model.pt "
               f"--output /kaggle/working/pointchess1024.pchess --ft_size 1024",
               shell=True, check=True)
subprocess.run("ls -lh /kaggle/working/pointchess1024.pchess", shell=True, check=False)
print("Mega-1024 training + export complete.")
