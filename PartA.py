import os
# Use threads for wandb internal operations to reduce interference with DataLoader workers.
os.environ["WANDB_START_METHOD"] = "thread"
from PIL import Image

import math
import time
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, utils

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import plotly.express as px

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

import wandb
from wandb.sdk.lib.service_connection import WandbServiceNotOwnedError
from wandb import Api

# -------------------------------
# Disable wandb in DataLoader workers.
# -------------------------------
def worker_init_fn(worker_id):
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_SILENT"] = "true"

# -------------------------------
# Monkey‑patch wandb.teardown to avoid multiprocessing errors.
# -------------------------------
_orig_teardown = wandb.teardown
def _safe_teardown(*args, **kwargs):
    if hasattr(wandb, 'run') and wandb.run is not None and hasattr(wandb.run, '_settings'):
        if os.getpid() == wandb.run._settings._start_pid:
            try:
                return _orig_teardown(*args, **kwargs)
            except WandbServiceNotOwnedError:
                return
    else:
        return
wandb.teardown = _safe_teardown

# -------------------------------
# Performance tweaks
# -------------------------------
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# -------------------------------
# Helper to parse boolean arguments from strings.
# -------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# -------------------------------
# TorchScript‑wrapped Mish.
# -------------------------------
@torch.jit.script
def jit_mish(x):
    return x * torch.tanh(F.softplus(x))

class MishScript(nn.Module):
    def forward(self, x):
        return jit_mish(x)

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "mish":
        return MishScript()
    raise ValueError(f"Unsupported activation: {name}")

# -------------------------------
# ConvBlock.
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, act, bn, drop, pool_k=2, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
        self.bn   = nn.BatchNorm2d(out_ch) if bn else None
        self.act  = get_activation(act)
        self.pool = nn.MaxPool2d(pool_k)
        self.drop = nn.Dropout2d(drop) if drop > 0 else None

    def forward_function(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        if self.drop:
            x = self.drop(x)
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self.forward_function, x)
        else:
            return self.forward_function(x)

# -------------------------------
# LightningModule.
# -------------------------------
class CNNModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        H, W = config["img_height"], config["img_width"]
        self.num_conv_layers = min(config["num_conv_layers"], 5)
        self.kernel_size = config["kernel_size"]
        self.base_filters = config["num_filters"]
        self.activation = config["activation"]
        self.use_batchnorm = config["batchnorm"]
        self.dropout_rate = config["dropout"]
        self.filter_organization = config["filter_organization"]
        self.use_checkpoint = config["use_checkpoint"]

        layers, in_ch = [], 3
        for i in range(self.num_conv_layers):
            if self.filter_organization == "same":
                out_ch = self.base_filters
            elif self.filter_organization == "double":
                out_ch = self.base_filters * (2 ** i)
            elif self.filter_organization == "half":
                out_ch = max(1, self.base_filters // (2 ** i))
            else:
                raise ValueError("Unsupported filter organization")
            layers.append(ConvBlock(
                in_ch, out_ch, self.kernel_size,
                self.activation, self.use_batchnorm,
                self.dropout_rate, pool_k=2,
                use_checkpoint=self.use_checkpoint,
            ))
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            conv_out = self.conv_layers(dummy)
        self.flat_dim = conv_out.view(1, -1).size(1)

        hidden, in_f = [], self.flat_dim
        for h in config["hidden_sizes"]:
            hidden += [nn.Linear(in_f, h), get_activation(self.activation)]
            in_f = h
        self.hidden_layers = nn.Sequential(*hidden)
        self.out = nn.Linear(in_f, 111)

        try:
            self = torch.compile(self)
            print(f"Model compiled! (Process {os.getpid()})")
        except Exception as e:
            print("Compilation not supported:", e)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.hidden_layers(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        imgs, lbls = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, lbls)
        acc = (logits.argmax(dim=1) == lbls).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, lbls = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, lbls)
        acc = (logits.argmax(dim=1) == lbls).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        imgs, lbls = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, lbls)
        acc = (logits.argmax(dim=1) == lbls).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"images": imgs, "preds": logits.argmax(dim=1), "labels": lbls}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.config["lr"],
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def visualize_filters(self):
        f = self.conv_layers[0].conv.weight.data.clone()
        f = (f - f.min()) / (f.max() - f.min())
        return utils.make_grid(f, nrow=int(math.sqrt(f.size(0))), normalize=True)

    def guided_backprop(self, img, layer_idx, neuron_idx):
        gradients = []
        def hook(m, grad_in, grad_out):
            gradients.append(grad_in[0])
        h = self.conv_layers[layer_idx].conv.register_full_backward_hook(hook)
        out = self(img.unsqueeze(0))
        self.zero_grad()
        out[0, neuron_idx].backward(retain_graph=True)
        h.remove()
        return gradients[0].cpu()

# -------------------------------
# DataModule.
# -------------------------------
class INatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, config, test_batch_size=None):
        super().__init__()
        self.data_dir = data_dir
        self.cfg = config
        self.batch_size = config["batch_size"]
        self.test_batch_size = test_batch_size or config["batch_size"]
        self.img_h = config["img_height"]
        self.img_w = config["img_width"]
        self.num_workers = config["num_workers"]
        self.aug = config["augmentation"]

    def setup(self, stage=None):
        if self.aug:
            tf_train = transforms.Compose([
                transforms.RandomResizedCrop((self.img_h, self.img_w), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            tf_train = transforms.Compose([
                transforms.Resize((self.img_h, self.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        tf_test = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        tr = datasets.ImageFolder(os.path.join(self.data_dir, "train"), tf_train)
        cls2idx = {}
        for idx, (_, lbl) in enumerate(tr.samples):
            cls2idx.setdefault(lbl, []).append(idx)
        t_idx, v_idx = [], []
        torch.manual_seed(42)
        for lbl, idxs in cls2idx.items():
            n_val = max(1, int(0.2 * len(idxs)))
            perm = torch.randperm(len(idxs))
            v_idx.extend([idxs[i] for i in perm[:n_val]])
            t_idx.extend([idxs[i] for i in perm[n_val:]])
        self.train_ds = Subset(tr, t_idx)
        self.val_ds = Subset(tr, v_idx)
        self.test_ds = datasets.ImageFolder(os.path.join(self.data_dir, "test"), tf_test)

    def train_dataloader(self):
        persistent = self.num_workers > 0
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        persistent = self.num_workers > 0
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self):
        persistent = self.num_workers > 0
        return DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
            worker_init_fn=worker_init_fn
        )

# -------------------------------
# Logging utilities.
# -------------------------------

def generate_simple_test_grid(model, dm, rows=10, cols=3):
    """
    Fallback simple grid: displays pred vs true as titles, colored borders for mistakes.
    """
    model.eval()
    all_imgs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for xb, yb in dm.test_dataloader():
            xb = xb.to(model.device)
            preds = model(xb).argmax(dim=1).cpu()
            all_imgs.extend(xb.cpu())
            all_preds.extend(preds)
            all_labels.extend(yb)
            if len(all_imgs) >= rows * cols:
                break

    all_imgs = all_imgs[:rows*cols]
    all_preds = all_preds[:rows*cols]
    all_labels = all_labels[:rows*cols]

    # Create matplotlib figure for local save
    fig, axes = plt.subplots(rows, cols, figsize=(15, 40))
    axes = axes.flatten()
    for i, (img, pred, label) in enumerate(zip(all_imgs, all_preds, all_labels)):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        axes[i].imshow(img_np)
        is_correct = (pred == label)
        title_color = 'green' if is_correct else 'red'
        axes[i].set_title(f"Pred: {pred.item()} | True: {label.item()}", color=title_color)
        axes[i].axis('off')
        if not is_correct:
            for spine in axes[i].spines.values():
                spine.set_color('red')
                spine.set_linewidth(2)

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    fig.suptitle(f"Test Predictions (10×3) — Acc: {acc*100:.2f}%", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("test_predictions_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Also log a PIL version to wandb:
    grid = utils.make_grid(all_imgs, nrow=cols, normalize=True)
    np_img = (grid.mul(255).permute(1, 2, 0).byte().cpu().numpy())
    pil_img = Image.fromarray(np_img)
    captions = [f"P:{p.item()} {'✓' if p==l else '✗'} T:{l.item()}" 
                for p, l in zip(all_preds, all_labels)]
    wandb.log({
        "Test_Grid_Matplotlib": wandb.Image(pil_img, caption=f"Acc {acc*100:.2f}% — " + " | ".join(captions))
    })
    return grid

def generate_test_grid(model, dm, rows=10, cols=3):
    """
    Build an interactive WandB Table of test images with
    predicted label, confidence and true label.
    """
    model.eval()
    table = wandb.Table(columns=["image", "predicted", "confidence", "actual"])
    count = 0

    # Inverse normalization for correct colors
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    inv_norm = transforms.Normalize(
        mean=(-mean / std).tolist(),
        std=(1.0 / std).tolist()
    )
    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for xb, yb in dm.test_dataloader():
            xb = xb.to(model.device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            probs = F.softmax(logits, dim=1).cpu()
            confs = probs[range(len(preds)), preds]

            for img_tensor, pred, conf, true in zip(xb.cpu(), preds, confs, yb):
                # undo preprocessing
                img = inv_norm(img_tensor)
                img = torch.clamp(img, 0, 1)
                pil = to_pil(img)

                # add to table
                table.add_data(
                    wandb.Image(pil, caption=f"P:{pred} ({conf:.2f}) / T:{true}"),
                    pred.item(),
                    float(conf),
                    true.item()
                )

                count += 1
                if count >= rows * cols:
                    break
            if count >= rows * cols:
                break

    # log the interactive table and scalar accuracy
    acc = sum(r[1] == r[3] for r in table.data) / len(table.data)
    wandb.log({
        "Interactive_Test_Results": table,
        "test_accuracy": acc
    })
    return table


def log_additional(config, acc):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    wandb.log({"Acc_vs_Time": wandb.Table(
        data=[[now, config["lr"], config["num_filters"], config["activation"], acc]],
        columns=["Time", "LR", "Filters", "Act", "Val_Acc"]
    )})
    keys = ['lr', 'batch_size', 'num_filters', 'kernel_size', 'activation',
            'dropout', 'batchnorm', 'hidden_sizes', 'augmentation',
            'filter_organization', 'img_height', 'img_width',
            'num_workers', 'num_conv_layers', 'use_checkpoint', 'accumulation_steps', 'max_epochs']
    tbl = wandb.Table(columns=keys)
    tbl.add_data(*[config.get(k) for k in keys])
    wandb.log({"Hyperparams": tbl})

# -------------------------------
# Experiment runner.
# -------------------------------

def run_experiment(args):
    hyper_keys = [
        "lr", "batch_size", "num_filters", "kernel_size", "activation",
        "dropout", "batchnorm", "hidden_sizes", "augmentation",
        "filter_organization", "img_height", "img_width",
        "num_workers", "num_conv_layers", "use_checkpoint", "accumulation_steps", "max_epochs"
    ]
    in_sweep_mode = (getattr(args, "run_mode", None) in ["sweep", "sweep_job"])
    if in_sweep_mode and wandb.run is not None:
        for k in hyper_keys:
            if k not in wandb.config:
                wandb.config.update({k: getattr(args, k)}, allow_val_change=True)
        config = dict(wandb.config)
    else:
        config = {k: getattr(args, k) for k in hyper_keys}

    run_name = (
        f"lr={config['lr']}_bs={config['batch_size']}_filters={config['num_filters']}_"
        f"kernel={config['kernel_size']}_act={config['activation']}_dropout={config['dropout']}_"
        f"bn={config['batchnorm']}_hidden={'-'.join(map(str, config['hidden_sizes']))}_"
        f"aug={config['augmentation']}_org={config['filter_organization']}_"
        f"conv_layers={config['num_conv_layers']}_acc={config['accumulation_steps']}_"
        f"epochs={config['max_epochs']}"
    )

    if not in_sweep_mode and wandb.run is None:
        wandb.init(
            project="iNat_assignment",
            config=config,
            name=run_name
        )
        config = dict(wandb.config)
    else:
        if wandb.run is not None:
            wandb.run.name = run_name

    max_epochs_val = config["max_epochs"]
    print(f"Using max_epochs: {max_epochs_val}")

    dm = INatDataModule(args.data_dir, config, test_batch_size=args.test_batch_size)
    model = CNNModel(config)

    logger = WandbLogger(experiment=wandb.run)
    early_stop = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    trainer_kwargs = {
        "max_epochs": max_epochs_val,
        "logger": logger,
        "callbacks": [early_stop],
        "log_every_n_steps": 50,
        "precision": args.precision,
        "accumulate_grad_batches": config["accumulation_steps"],
    }

    print(f"Trainer kwargs max_epochs: {trainer_kwargs['max_epochs']}")

    if torch.cuda.is_available() and args.gpus > 0:
        trainer_kwargs.update(accelerator="gpu", devices=args.gpus)
    else:
        trainer_kwargs.update(accelerator="cpu", devices=1)
    trainer = pl.Trainer(**trainer_kwargs)

    try:
        trainer.fit(model, dm)
    except Exception as e:
        print(f"Training failed: {e}")
    else:
        generate_test_grid(model, dm)
        wandb.log({"First_Layer_Filters": wandb.Image(model.visualize_filters())})
        try:
            img, _ = next(iter(dm.test_dataloader()))
            img = img[0].to(model.device)
            with torch.amp.autocast(model.device.type):
                grad = model.guided_backprop(img, model.num_conv_layers - 1, 0)
            grad_vis = torch.abs(grad)
            if grad_vis.ndim == 4 and grad_vis.shape[0] == 1:
                grad_vis = grad_vis.squeeze(0)
            elif grad_vis.ndim == 3:
                pass
            else:
                grad_vis = None
            if grad_vis is not None:
                C, H, W = grad_vis.shape
                if C > 3:
                    grad_vis = grad_vis.mean(dim=0, keepdim=True)
                if grad_vis.shape[0] == 1:
                    grad_vis = grad_vis.squeeze(0)
                mn, mx = grad_vis.min(), grad_vis.max()
                grad_vis = (grad_vis - mn) / (mx - mn) if mx > mn else grad_vis
                wandb.log({"Guided_BP": wandb.Image(grad_vis)})
        except Exception as e:
            print("Error in guided backprop:", e)

        try:
            test_result = trainer.test(model, dm, verbose=False)
            final_acc = test_result[0].get("test_acc", 0.0)
        except Exception as e:
            print("Error during test evaluation:", e)
            final_acc = 0.0

        log_additional(config, final_acc)
    finally:
        if wandb.run is not None:
            wandb.finish()

# -------------------------------
# Top-level Sweep Run Function (for a single sweep job).
# -------------------------------
def sweep_run(args, sweep_id):
    if wandb.run is None:
        wandb.init(project="iNat_assignment")

    sweep_config = dict(wandb.config)
    print(f"Original sweep_config: {sweep_config}")

    for k, v in sweep_config.items():
        setattr(args, k, v)

    args.run_mode = "sweep_job"
    print("Sweep job config:", sweep_config, f"(Process {os.getpid()})")

    run_experiment(args)

# -------------------------------
# Main entry point.
# -------------------------------
def main():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--run_mode", choices=["single", "sweep"], required=True,
                             help="Choose 'single' for one run, or 'sweep' for sequential sweeps.")
    base_parser.add_argument("--sweep_count", type=int, default=10)
    base_parser.add_argument("--data_dir", default="./iNat_dataset")
    base_parser.add_argument("--only_test", action="store_true")
    base_parser.add_argument("--test_batch_size", type=int, default=None)
    base_parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    base_parser.add_argument("--num_cpus", type=int, default=os.cpu_count() or 4)
    base_parser.add_argument("--precision", choices=["32", "16-mixed"], default="16-mixed")

    args, remaining = base_parser.parse_known_args()

    parser = argparse.ArgumentParser(parents=[base_parser])
    if args.run_mode == "single":
        parser.add_argument("--max_epochs", "--epochs", type=int, required=True, dest="max_epochs")
    else:
        parser.add_argument("--max_epochs", "--epochs", type=int, default=20, dest="max_epochs")

    if args.run_mode == "single":
        parser.add_argument("--lr", type=float, required=True)
        parser.add_argument("--batch_size", type=int, required=True)
        parser.add_argument("--num_filters", type=int, required=True)
        parser.add_argument("--kernel_size", type=int, required=True)
        parser.add_argument("--activation", required=True)
        parser.add_argument("--dropout", type=float, required=True)
        parser.add_argument("--batchnorm", type=lambda x: str(x).lower() == "true", required=True)
        parser.add_argument("--hidden_sizes", type=int, nargs="+", required=True)
        parser.add_argument("--augmentation", type=str2bool, nargs='?', const=True, required=True,
                            help="Set to true to enable data augmentation; false otherwise.")
        parser.add_argument("--filter_organization", choices=["same", "double", "half"], required=True)
        parser.add_argument("--img_height", type=int, required=True)
        parser.add_argument("--img_width", type=int, required=True)
        parser.add_argument("--num_workers", type=int, required=True)
        parser.add_argument("--num_conv_layers", type=int, required=True)
        parser.add_argument("--use_checkpoint", type=lambda x: str(x).lower() == "true", required=True)
        parser.add_argument("--accumulation_steps", type=int, required=True)
    else:
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_filters", type=int, default=32)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--activation", default="relu")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--batchnorm", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[128])
        parser.add_argument("--augmentation", type=str2bool, nargs='?', const=True, default=True,
                            help="Set to true to enable data augmentation; false otherwise.")
        parser.add_argument("--filter_organization", choices=["same", "double", "half"], default="same")
        parser.add_argument("--img_height", type=int, default=224)
        parser.add_argument("--img_width", type=int, default=224)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--num_conv_layers", type=int, default=5)
        parser.add_argument("--use_checkpoint", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--accumulation_steps", type=int, default=4)

    args = parser.parse_args()

    if args.run_mode == "sweep":
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "test_acc", "goal": "maximize"},
            "parameters": {
                "lr":                {"values": [0.0005, 0.001]},
                "batch_size":        {"values": [32]},
                "num_filters":       {"values": [64]},
                "kernel_size":       {"values": [3]},
                "activation":        {"values": ["mish", "gelu"]},
                "dropout":           {"values": [0.1, 0.2]},
                "batchnorm":         {"values": [True]},
                "hidden_sizes":      {"values": [[256, 128]]},
                "filter_organization":{"values": ["double"]},
                "augmentation":      {"values": [True]},
                # Fixed dataset / training parameters:
                "img_height":        {"value": 224},
                "img_width":         {"value": 224},
                "num_workers":       {"value": 4},
                "num_conv_layers":   {"value": 5},
                "use_checkpoint":    {"values": [False]},
                "accumulation_steps":{"values": [4]},
                "max_epochs":        {"value": 20}
            },
            "program": "PartA.py"
        }

        sweep_id = wandb.sweep(sweep_config, project="iNat_assignment")
        sweep_runner = partial(sweep_run, args, sweep_id)
        print(f"Starting agent for sweep {sweep_id} with count {args.sweep_count}")
        wandb.agent(sweep_id, function=sweep_runner, count=args.sweep_count)
        print("Agent completed")
    else:
        run_experiment(args)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
