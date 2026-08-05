# Flexiv Rizon True VLA

This repository contains the training and evaluation pipeline for a True VLA policy in MuJoCo.

## Prerequisite

First install and run the data-collection project:

https://github.com/Alchemist77/mujoco-vla-data-collection

That project generates the demonstration dataset required for training this VLA model.

## Workflow

1. Install the data-collection repository.
2. Collect demonstration data.
3. Copy the generated `.npz` dataset into this repository.
4. Train the True VLA model.
5. Test the trained checkpoint and save evaluation videos.


## Installation

### 1. Clone the previous data-collection project

```bash
git clone https://github.com/Alchemist77/mujoco-vla-data-collection.git
cd mujoco-vla-data-collection
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 2. Collect data


### 3. Clone the new VLA repository

```bash
git clone https://github.com/Alchemist77/Flexiv-Rizon4S-VLA.git
cd Flexiv-Rizon4S-VLA
```

### 4. Train

```bash
python3 scripts/train_true_vla.py \
  --dataset-path data/mujoco_dataset_224_ep100_remove_home.npz \
  --clip-path models/CLIP-ViT-L-14-laion2B-s32B-b82K \
  --save-path checkpoints/true_vla_best.pt \
  --batch-size 128 \
  --epochs 100 \
  --device cuda \
  --freeze-clip \
  --early-stop-patience 10 \
  --num-workers 16
```

### 4. Test and save MP4

```bash
python3 scripts/test_true_vla_100.py \
  --checkpoint checkpoints/true_vla_best.pt \
  --device cuda \
  --episodes 10 \
  --max-steps 10000 \
  --render-size 256 \
  --model-image-size 224 \
  --save-video \
  --video-dir videos_true_vla \
  --seed 0
```



