# PartB.py — Fine‑tune ResNet50 on a naturalist dataset with multiple strategies

import os                              # for filesystem operations
import argparse                        # for parsing command‑line arguments
import torch                           # core PyTorch library
import torch.nn as nn                  # neural network modules
import torch.optim as optim            # optimization algorithms
from torch.utils.data import DataLoader  # data loading utility
from torchvision import transforms, datasets, models  # vision datasets & models

import pytorch_lightning as pl                       # Lightning for training loop
from pytorch_lightning.loggers import WandbLogger    # W&B integration
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # training callbacks

# -----------------------------------------------
# LightningModule that wraps a pretrained ResNet50
# and applies different fine‑tuning strategies.
# -----------------------------------------------
class FineTuneModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 finetune_strategy: str,
                 freeze_last_k: int = 0,
                 lr: float = 1e-3):
        """
        Args:
            num_classes: number of output classes (e.g., 10 for iNaturalist).
            finetune_strategy: one of
                - "freeze_all":    freeze all except final head
                - "freeze_last_k": freeze all then unfreeze last k param groups
                - "full_finetuning": unfreeze all layers
                - "gradual_unfreeze": unfreeze head, then layers gradually
            freeze_last_k: for "freeze_last_k", how many parameter groups at end to unfreeze
            lr: base learning rate
        """
        super().__init__()
        # save_hyperparameters populates self.hparams with all __init__ args
        self.save_hyperparameters()

        # ---- 1) Load pretrained ResNet50 and replace classifier head ----
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # new head: Linear → ReLU → Dropout → Linear(num_classes)
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # ---- 2) Freeze BatchNorm running stats in backbone ----
        # so that statistics learned on ImageNet are preserved
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        # ---- 3) Apply chosen fine‑tuning strategy to rest of parameters ----
        strat = finetune_strategy
        if strat == "freeze_all":
            # freeze everything except new head
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        elif strat == "freeze_last_k":
            # freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False
            # unfreeze last k parameter groups if valid
            params = list(self.model.named_parameters())
            if 0 < freeze_last_k <= len(params):
                for name, param in params[-freeze_last_k:]:
                    param.requires_grad = True
            else:
                # fallback to full fine‑tuning if k out of range
                for param in self.model.parameters():
                    param.requires_grad = True

        elif strat == "full_finetuning":
            # unfreeze every parameter
            for param in self.model.parameters():
                param.requires_grad = True

        elif strat == "gradual_unfreeze":
            # initially only head is trainable
            for name, param in self.model.named_parameters():
                param.requires_grad = ("fc" in name)

        else:
            raise ValueError(f"Invalid fine‑tuning strategy: {strat}")

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass through the backbone + head."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Single training step: compute loss and log accuracy."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step: compute & log val loss/accuracy."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Set up optimizer with layer‑wise learning rates:
        - backbone_params   : lr * 0.05
        - head_params       : lr * 0.5
        Then attach a ReduceLROnPlateau scheduler on val_loss.
        """
        base_lr = self.hparams.lr

        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                head_params.append(param)
            elif param.requires_grad:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': base_lr * 0.05},
            {'params': head_params,     'lr': base_lr * 0.5}
        ])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        # Lightning requires specifying monitor for ReduceLROnPlateau
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def on_epoch_start(self):
        """
        For gradual unfreezing: after epoch 5, unfreeze one extra backbone layer per epoch.
        """
        if (self.hparams.finetune_strategy == 'gradual_unfreeze' and
                self.current_epoch >= 5):
            to_unfreeze = self.current_epoch - 4
            layers = [n for n, _ in self.model.named_parameters() if 'layer' in n]
            for name, param in self.model.named_parameters():
                if name in layers[-to_unfreeze:]:
                    param.requires_grad = True

# -----------------------------------------------
# Data augmentation transforms
# -----------------------------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(256),                # resize shorter side to 256
    transforms.RandomCrop(224),            # random crop to 224×224
    transforms.RandomHorizontalFlip(),     # random flip
    transforms.ColorJitter(               # slight color jitter
        brightness=0.1, contrast=0.1, saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),            # deterministic center crop
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

def get_data_loaders(data_dir: str, batch_size: int):
    """
    Build PyTorch DataLoaders for training and validation.
    Expects subfolders 'train' and 'val' under data_dir, each containing
    class‑labeled subdirectories.
    """
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transforms
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader

def main():
    """
    Parse command‑line args, set up data, model, logger, callbacks, and train.
    """
    parser = argparse.ArgumentParser(
        description="Fine‑tune a pretrained ResNet50 on a naturalist dataset"
    )
    parser.add_argument('--data_dir',       type=str, required=True,
                        help="Path to dataset root (with 'train' & 'val').")
    parser.add_argument('--batch_size',     type=int, default=32,
                        help="Batch size for both train & val loaders.")
    parser.add_argument('--epochs',         type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument('--lr',             type=float, default=1e-3,
                        help="Base learning rate.")
    parser.add_argument('--finetune_strategy',
                        choices=['freeze_all', 'freeze_last_k',
                                 'full_finetuning', 'gradual_unfreeze'],
                        default='freeze_all',
                        help="Which fine‑tuning strategy to apply.")
    parser.add_argument('--freeze_last_k',  type=int, default=20,
                        help="(if freeze_last_k) number of param groups to unfreeze.")
    parser.add_argument('--wandb_project',  type=str, default='fine_tuning_project',
                        help="Weights & Biases project name.")
    parser.add_argument('--use_fp16',      action='store_true',
                        help="Enable 16‑bit precision training.")
    args = parser.parse_args()

    # prepare data
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    num_classes = len(train_loader.dataset.classes)

    # build model
    model = FineTuneModel(
        num_classes=num_classes,
        finetune_strategy=args.finetune_strategy,
        freeze_last_k=args.freeze_last_k,
        lr=args.lr
    )

    # setup W&B logger
    wandb_logger = WandbLogger(project=args.wandb_project)

    # callbacks: checkpoint on best val_acc, early stop after plateau
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_acc:.2f}'
    )
    early_stop = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=5,
        verbose=True
    )

    # configure Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',  # GPU if available else CPU
        devices=1,
        precision=16 if args.use_fp16 else 32,
        logger=wandb_logger,
        callbacks=[checkpoint, early_stop]
    )

    # run training
    trainer.fit(model, train_loader, val_loader)
    # upon completion, best checkpoint is loaded by default

if __name__ == '__main__':
    main()
