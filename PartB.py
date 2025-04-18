# import os
# import argparse
# from functools import partial

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms, datasets, models

# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# import wandb

# # ================================================================
# # Fine-tuning Strategies:
# # We explore 4 strategies:
# # 1. "freeze_all": Freeze all pretrained layers except the final classifier.
# # 2. "freeze_last_k": Freeze all layers except the last k parameter groups.
# # 3. "full_finetuning": Unfreeze all layers and use layer-wise learning rates.
# # 4. "gradual_unfreeze": Start with all pretrained layers frozen (except the final classifier)
# #      and gradually unfreeze them during training.
# # ================================================================

# class FineTuneModel(pl.LightningModule):
#     def __init__(self, num_classes: int, finetune_strategy: str, freeze_last_k: int = 0, lr: float = 1e-3):
#         """
#         Args:
#             num_classes (int): number of output classes (e.g., 10 for the naturalist dataset)
#             finetune_strategy (str): Strategy for fine-tuning (freeze_all, freeze_last_k, full_finetuning, gradual_unfreeze)
#             freeze_last_k (int): If using the freeze_last_k strategy, number of parameter groups from the end to unfreeze.
#             lr (float): Base learning rate.
#         """
#         super().__init__()
#         self.save_hyperparameters()

#         # ------------------------------------------------
#         # Load a pretrained ResNet50 model and adjust the final layer.
#         # (Question 1: Adjusting input dimensions is handled via transforms,
#         #  and the final classification layer is replaced to match num_classes)
#         # ------------------------------------------------
#         self.model = models.resnet50(pretrained=True)
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, num_classes)

#         # ------------------------------------------------
#         # Apply fine-tuning strategies.
#         # ------------------------------------------------
#         if finetune_strategy == "freeze_all":
#             # Freeze all layers except final classifier
#             for name, param in self.model.named_parameters():
#                 if "fc" not in name:
#                     param.requires_grad = False
#             print("Fine-tuning strategy: Freeze all layers except final classifier.")

#         elif finetune_strategy == "freeze_last_k":
#             # Freeze all layers first
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             # Then unfreeze the last k parameter groups (if k is a valid index)
#             params = list(self.model.named_parameters())
#             if freeze_last_k > 0 and freeze_last_k <= len(params):
#                 for name, param in params[-freeze_last_k:]:
#                     param.requires_grad = True
#                 print(f"Fine-tuning strategy: Freeze all layers except the last {freeze_last_k} parameter groups.")
#             else:
#                 print("Warning: freeze_last_k parameter is out of range. Using full fine-tuning.")
#                 for param in self.model.parameters():
#                     param.requires_grad = True

#         elif finetune_strategy == "full_finetuning":
#             # Unfreeze all parameters
#             for param in self.model.parameters():
#                 param.requires_grad = True
#             print("Fine-tuning strategy: Full fine-tuning of all layers.")

#         elif finetune_strategy == "gradual_unfreeze":
#             # Initially freeze all pretrained layers (except classifier).
#             for name, param in self.model.named_parameters():
#                 if "fc" not in name:
#                     param.requires_grad = False
#             print("Fine-tuning strategy: Gradual unfreezing starting from frozen pretrained layers.")

#         else:
#             raise ValueError("Invalid fine-tuning strategy provided.")

#         self.lr = lr
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         images, labels = batch
#         outputs = self.forward(images)
#         loss = self.criterion(outputs, labels)
#         acc = (outputs.argmax(dim=1) == labels).float().mean()
#         self.log("train_loss", loss, prog_bar=True)
#         self.log("train_acc", acc, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         images, labels = batch
#         outputs = self.forward(images)
#         loss = self.criterion(outputs, labels)
#         acc = (outputs.argmax(dim=1) == labels).float().mean()
#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_acc", acc, prog_bar=True)

#     def configure_optimizers(self):
#         # ------------------------------------------------
#         # Implement layer-wise learning rates when fine-tuning all parameters.
#         # Lower the learning rate for pretrained (backbone) layers and use a higher one for new layers.
#         # ------------------------------------------------
#         if self.hparams.finetune_strategy in ["full_finetuning", "gradual_unfreeze"]:
#             backbone_params = []
#             classifier_params = []
#             for name, param in self.model.named_parameters():
#                 if "fc" in name:
#                     classifier_params.append(param)
#                 else:
#                     backbone_params.append(param)
#             optimizer = optim.Adam([
#                 {'params': backbone_params, 'lr': self.lr * 0.1},
#                 {'params': classifier_params, 'lr': self.lr}
#             ])
#             return optimizer
#         else:
#             # For freeze_all or freeze_last_k, only the unfrozen parameters are trained.
#             optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
#             return optimizer

#     def on_epoch_start(self):
#         # Implement gradual unfreezing strategy if selected.
#         if self.hparams.finetune_strategy == "gradual_unfreeze":
#             current_epoch = self.current_epoch
#             # Start unfreezing after 5 epochs; unfreeze 2 layers per epoch.
#             if current_epoch >= 5:
#                 layers_to_unfreeze = (current_epoch - 5) * 2
#                 # Get list of freezable layers (excluding classifier)
#                 freezable_layers = [name for name, _ in self.model.named_parameters() if "fc" not in name]
#                 total_freezable = len(freezable_layers)
#                 # Unfreeze the last "layers_to_unfreeze" layers if available.
#                 for name, param in self.model.named_parameters():
#                     if "fc" not in name:
#                         layer_index = freezable_layers.index(name)
#                         if layer_index >= total_freezable - layers_to_unfreeze:
#                             if not param.requires_grad:
#                                 print(f"Epoch {current_epoch}: Unfreezing layer {name}")
#                             param.requires_grad = True


# def get_data_loaders(data_dir: str, batch_size: int):
#     """
#     Create train and validation data loaders with enhanced data augmentation.
    
#     The expected folder structure is:
#       data_dir/
#          train/
#              class1/...
#              class2/...
#              ...
#          val/
#              class1/...
#              class2/...
#              ...
#     """
#     # ImageNet statistics for normalization
#     imagenet_mean = [0.485, 0.456, 0.406]
#     imagenet_std = [0.229, 0.224, 0.225]

#     # Enhanced training transforms: random crop, color jitter, random rotation added
#     train_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),             # Random crop for training
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
#     ])

#     # Validation transforms: center crop is typical; no data augmentation
#     val_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
#     ])

#     train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
#     val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
#     return train_loader, val_loader

# def main(args):
#     # Initialize wandb logger for experiment tracking.
#     wandb_logger = WandbLogger(
#         project=args.wandb_project,
#         log_model='all'
#     )
#     wandb_logger.experiment.config.update(vars(args))

#     # Create callbacks
#     checkpoint_callback = pl.callbacks.ModelCheckpoint(
#         monitor='val_acc',
#         mode='max',
#         save_top_k=1,
#         save_last=True,
#         filename='{epoch}-{val_acc:.2f}'
#     )
#     early_stop_callback = pl.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=3,
#         mode='min'
#     )

#     # Get the data loaders
#     train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    
#     # Number of classes is determined from the training dataset folder structure.
#     num_classes = len(train_loader.dataset.classes)
#     print(f"Detected {num_classes} classes.")

#     # Create model instance
#     model = FineTuneModel(
#         num_classes=num_classes,
#         finetune_strategy=args.finetune_strategy,
#         freeze_last_k=args.freeze_last_k,
#         lr=args.lr
#     )

#     # Setup Trainer with GPU (if available) and additional callbacks.
#     trainer = pl.Trainer(
#     max_epochs=args.epochs,
#     accelerator="gpu" if torch.cuda.is_available() else "cpu",
#     devices=1 if torch.cuda.is_available() else None,
#     logger=wandb_logger,
#     precision=16 if args.use_fp16 else 32,
#     callbacks=[checkpoint_callback, early_stop_callback]
# )



#     # Train the model
#     trainer.fit(model, train_loader, val_loader)

#     # The wandb run will have all the logs and plots for the report.
#     wandb.finish()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Fine-tune a pretrained model on a naturalist dataset")
#     parser.add_argument('--data_dir', type=str, required=True,
#                         help="Directory containing 'train' and 'val' subdirectories")
#     parser.add_argument('--wandb_project', type=str, default="fine_tuning_project",
#                         help="wandb project name")
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help="Batch size for training")
#     parser.add_argument('--epochs', type=int, default=10,
#                         help="Number of training epochs")
#     parser.add_argument('--lr', type=float, default=1e-3,
#                         help="Base learning rate")
#     parser.add_argument('--finetune_strategy', type=str, default="freeze_all",
#                         choices=["freeze_all", "freeze_last_k", "full_finetuning", "gradual_unfreeze"],
#                         help="Fine-tuning strategy to use")
#     parser.add_argument('--freeze_last_k', type=int, default=20,
#                         help="(For freeze_last_k) Number of parameter groups from the end to unfreeze")
#     parser.add_argument('--use_fp16', action="store_true",
#                         help="Enable mixed precision training")
#     args = parser.parse_args()

#     main(args)

# PartB.py — Fixed configure_optimizers by using self.hparams.lr

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class FineTuneModel(pl.LightningModule):
    def __init__(self, num_classes: int, finetune_strategy: str,
                 freeze_last_k: int = 0, lr: float = 1e-3):
        super().__init__()
        # Save all __init__ args to self.hparams (includes lr) :contentReference[oaicite:10]{index=10}
        self.save_hyperparameters()

        # Load pretrained ResNet50 and replace the classifier head
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # Freeze BatchNorm layers to keep running stats fixed :contentReference[oaicite:11]{index=11}
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        # Apply fine-tuning strategy (unchanged logic)
        strat = finetune_strategy
        if strat == "freeze_all":
            for n, p in self.model.named_parameters():
                if "fc" not in n:
                    p.requires_grad = False

        elif strat == "freeze_last_k":
            for p in self.model.parameters():
                p.requires_grad = False
            params = list(self.model.named_parameters())
            if 0 < freeze_last_k <= len(params):
                for n, p in params[-freeze_last_k:]:
                    p.requires_grad = True
            else:
                for p in self.model.parameters():
                    p.requires_grad = True

        elif strat == "full_finetuning":
            for p in self.model.parameters():
                p.requires_grad = True

        elif strat == "gradual_unfreeze":
            for n, p in self.model.named_parameters():
                p.requires_grad = ("fc" in n)

        else:
            raise ValueError("Invalid fine-tuning strategy")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Pull lr from saved hyperparameters instead of self.lr :contentReference[oaicite:12]{index=12}
        base_lr = self.hparams.lr

        # Separate params: backbone vs. head
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                head_params.append(param)
            elif param.requires_grad:
                backbone_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': base_lr * 0.05},
            {'params': head_params,    'lr': base_lr * 0.5}
        ])

        # Reduce LR on plateau of val_loss :contentReference[oaicite:13]{index=13}
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        # Must include monitor key for ReduceLROnPlateau in Lightning :contentReference[oaicite:14]{index=14}
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def on_epoch_start(self):
        # Gradual unfreeze: one layer per epoch after epoch 5
        if self.hparams.finetune_strategy == 'gradual_unfreeze' and self.current_epoch >= 5:
            to_unfreeze = self.current_epoch - 4
            layers = [n for n, _ in self.model.named_parameters() if 'layer' in n]
            for name, param in self.model.named_parameters():
                if name in layers[-to_unfreeze:]:
                    param.requires_grad = True

# Data augmentation
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

def get_data_loaders(data_dir, batch_size):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_transforms)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--finetune_strategy',
                        choices=['freeze_all','freeze_last_k','full_finetuning','gradual_unfreeze'],
                        default='freeze_all')
    parser.add_argument('--freeze_last_k', type=int, default=20)
    parser.add_argument('--wandb_project', type=str, default='fine_tuning_project')
    parser.add_argument('--use_fp16',      action='store_true')
    args = parser.parse_args()

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    num_classes = len(train_loader.dataset.classes)

    model = FineTuneModel(
        num_classes=num_classes,
        finetune_strategy=args.finetune_strategy,
        freeze_last_k=args.freeze_last_k,
        lr=args.lr
    )
    wandb_logger = WandbLogger(project=args.wandb_project)

    # EarlyStopping on val_acc, patience=5 (stop on best accuracy) :contentReference[oaicite:15]{index=15}&#8203;:contentReference[oaicite:16]{index=16}
    early_stop = EarlyStopping(monitor='val_acc', mode='max', patience=5, verbose=True)
    checkpoint = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        precision=16 if args.use_fp16 else 32,
        callbacks=[checkpoint, early_stop],
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)
    # Best model checkpoint is automatically loaded afterwards

if __name__ == '__main__':
    main()
