# Image Classification with PyTorch Lightning and Fine‑Tuning ResNet50

## Overview
This repository contains two scripts for image classification:

- **PartA.py**: Defines and trains a custom Convolutional Neural Network (CNN) from scratch on the iNaturalist dataset using PyTorch Lightning, with support for single runs and hyperparameter sweeps via Weights & Biases (wandb).
- **PartB.py**: Fine‑tunes a pretrained ResNet50 backbone on a custom dataset, exploring multiple strategies (freeze_all, freeze_last_k, full_finetuning, gradual_unfreeze).

## Repository Structure
```
├── PartA.py            # Training CNN from scratch with wandb integration and sweeps
├── PartB.py            # Fine‑tuning pretrained ResNet50 with Lightning
├── data/               # Dataset root (user-provided)
│   ├── train/          # Training images (class subfolders)
│   ├── val/            # Validation images (for PartB)
│   └── test/           # Test images (for PartA)
├── requirements.txt    # Python dependencies
└── README.md           # This documentation
```

## Prerequisites
- **OS**: Linux/macOS/Windows
- **Python**: 3.8 or higher
- **Hardware**: GPU recommended (CUDA-enabled)
- **Disk**: Sufficient space for dataset and model artifacts

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_dir>
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> **requirements.txt** should include:
> ```
> torch             # PyTorch
> torchvision       # Computer vision utils
> pytorch-lightning # Trainer and LightningModule
> wandb             # Experiment tracking
> pandas            # Logging hyperparameters
> matplotlib        # Plotting test grids
> plotly-express    # Interactive visualization (optional)
> pillow            # Image processing
> ```

## Data Preparation
- **PartA** expects:
  ```
  data_dir/
  ├── train/    # subfolders per class, images
  └── test/     # subfolders per class, images
  ```
  Validation set is automatically split from `train/` (20% per class).

- **PartB** expects:
  ```
  data_dir/
  ├── train/    # subfolders per class, images
  └── val/      # subfolders per class, images
  ```

Ensure the directory structure matches above before running.

## Usage

### Part A: Training CNN from Scratch

#### Single Run
Train a single model with specified hyperparameters:
```bash
python PartA.py \
  --run_mode single \
  --max_epochs 30 \
  --lr 0.001 \
  --batch_size 32 \
  --num_filters 32 \
  --kernel_size 3 \
  --activation relu \
  --dropout 0.1 \
  --batchnorm True \
  --hidden_sizes 128 64 \
  --augmentation True \
  --filter_organization same \
  --img_height 224 \
  --img_width 224 \
  --num_workers 4 \
  --num_conv_layers 5 \
  --use_checkpoint False \
  --accumulation_steps 4 \
  --data_dir ./data \
  --precision 16-mixed
```
- Logs metrics (`train_loss`, `val_loss`, `train_acc`, `val_acc`) to wandb.
- Saves test prediction grid (`test_predictions_grid.png`).
- Default uses CPU unless `--gpus` > 0 and CUDA available.

#### Hyperparameter Sweep
Run a Bayesian sweep over predefined ranges:
```bash
python PartA.py --run_mode sweep --sweep_count 20 --data_dir ./data
```
- Uses wandb Bayesian sweeps (lr, dropout, activation, etc.).
- `sweep_count` controls number of trials.
- Results logged under `iNat_assignment` project in wandb.

### Part B: Fine‑Tuning ResNet50

```bash
python PartB.py \
  --data_dir ./data \
  --batch_size 32 \
  --epochs 20 \
  --lr 1e-3 \
  --finetune_strategy freeze_all \
  --freeze_last_k 20 \
  --wandb_project fine_tuning_project \
  --use_fp16
```

- **Strategies**:
  - `freeze_all`: only final classifier is trained.
  - `freeze_last_k`: last _k_ parameter groups unfrozen.
  - `full_finetuning`: all layers trainable with layer‑wise LRs.
  - `gradual_unfreeze`: unfreeze backbone layers gradually after epoch 5.

- Model checkpoints saved on best `val_acc`.
- Early stopping on `val_acc` patience = 5.

$1- **W&B Report**: [Comprehensive report for both Part A & Part B]([https://wandb.ai/cs24m044-iit-madras-alumni-association/iNat_assignment/reports/DA6401-Assignment-2--VmlldzoxMjIxNjQwNg/edit?draftId=VmlldzoxMjIxODg4OA%3D%3D](https://wandb.ai/cs24m044-iit-madras-alumni-association/iNat_assignment/reports/DA6401-Assignment-2--VmlldzoxMjIxNjQwNg?accessToken=wzwlrobmdquve68ukl7jd5qncm8xxpdr11960mz23awqicdlygwz38y0knqcaw6e))

## Outputs
- **PartA**:
  - `test_predictions_grid.png`
  - wandb run artifacts: filter visualization, guided backprop images, hyperparameter tables.
- **PartB**:
  - Checkpoint files (`.ckpt`) for best validation accuracy.
  - wandb dashboard with training curves.


