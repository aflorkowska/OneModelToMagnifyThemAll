"""
Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_

Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

Implemented by https://github.com/milesial/Pytorch-UNet/tree/master
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.cli import instantiate_class

from networks.unet.unet_utils import (
    DoubleConv,
    Down,
    Up,
    OutConv,
    ConditionalNormDoubleConv,
    ConditionalNormDown,
    ConditionalNormUp,
)

from torchmetrics import MetricCollection

# PC
from torchmetrics.segmentation import DiceScore as Dice
from torchmetrics.classification import JaccardIndex

# HPC
# from torchmetrics import Dice
# from torchmetrics import JaccardIndex


class UNet(pl.LightningModule):
    def __init__(
        self,
        segmentation_type,  # "binary_segmentation" or "multiclass_segmentation"
        n_channels,
        n_classes,
        features_start=64,
        bilinear=False,
        variant="base",  # 'base', 'conditional_normalization'
        num_conditions=None,  # Required for conditional variants
        class_weights_dict=None,
        optimizer_init=None,
        lr_scheduler_init=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.segmentation_type = segmentation_type
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.variant = variant
        self.num_conditions = num_conditions

        # Validate parameters
        if variant == "conditional_normalization":
            if num_conditions is None:
                raise ValueError(
                    f"num_conditions must be provided for variant '{variant}'"
                )

        factor = 2 if bilinear else 1

        if variant == "base":
            self.inc = DoubleConv(n_channels, features_start)
            self.down1 = Down(features_start, features_start * 2)
            self.down2 = Down(features_start * 2, features_start * 4)
            self.down3 = Down(features_start * 4, features_start * 8)
            self.down4 = Down(features_start * 8, features_start * 16 // factor)
            self.up1 = Up(features_start * 16, features_start * 8 // factor, bilinear)
            self.up2 = Up(features_start * 8, features_start * 4 // factor, bilinear)
            self.up3 = Up(features_start * 4, features_start * 2 // factor, bilinear)
            self.up4 = Up(features_start * 2, features_start, bilinear)
            self.outc = OutConv(features_start, n_classes)

        elif variant == "conditional_normalization":
            self.inc = ConditionalNormDoubleConv(
                num_conditions, n_channels, features_start
            )
            self.down1 = ConditionalNormDown(
                num_conditions, features_start, features_start * 2
            )
            self.down2 = ConditionalNormDown(
                num_conditions, features_start * 2, features_start * 4
            )
            self.down3 = ConditionalNormDown(
                num_conditions, features_start * 4, features_start * 8
            )
            self.down4 = ConditionalNormDown(
                num_conditions, features_start * 8, features_start * 16 // factor
            )
            self.up1 = ConditionalNormUp(
                num_conditions,
                features_start * 16,
                features_start * 8 // factor,
                bilinear,
            )
            self.up2 = ConditionalNormUp(
                num_conditions,
                features_start * 8,
                features_start * 4 // factor,
                bilinear,
            )
            self.up3 = ConditionalNormUp(
                num_conditions,
                features_start * 4,
                features_start * 2 // factor,
                bilinear,
            )
            self.up4 = ConditionalNormUp(
                num_conditions, features_start * 2, features_start, bilinear
            )
            self.outc = OutConv(features_start, n_classes)

        else:
            raise ValueError(
                f"Unknown variant: {variant}. Must be 'base' or 'conditional_normalization'"
            )

        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init

        LOSS_MAP = {
            "binary_segmentation": nn.BCEWithLogitsLoss,
            "multiclass_segmentation": nn.CrossEntropyLoss,
        }

        METRIC_CONFIG = {
            "binary_segmentation": {
                "task": "binary",
                "num_classes": 1,
                "average": "macro",
            },
            "multiclass_segmentation": {
                "task": "multiclass",
                "num_classes": n_classes,
                "average": "macro",
            },
        }
        if segmentation_type not in LOSS_MAP or segmentation_type not in METRIC_CONFIG:
            raise ValueError(f"Unknown segmentation_type: {segmentation_type}")

        weights = None
        if class_weights_dict is not None:
            weights = torch.tensor(
                [class_weights_dict[i] for i in range(len(class_weights_dict))],
                dtype=torch.float32,
            )

        if segmentation_type == "binary_segmentation":
            self.criterion = LOSS_MAP[segmentation_type](weight=weights)
        elif segmentation_type == "multiclass_segmentation":
            self.criterion = LOSS_MAP[segmentation_type](weight=weights)
        else:
            raise ValueError(f"Unknown segmentation_type: {segmentation_type}")

        config = METRIC_CONFIG[segmentation_type]

        base_metrics = MetricCollection(
            {
                "dice": Dice(
                    average=config["average"],
                    num_classes=config["num_classes"],
                ),
                "iou": JaccardIndex(
                    average=config["average"],
                    num_classes=config["num_classes"],
                    task=config["task"],
                ),
            }
        )

        self.train_metric = base_metrics.clone(prefix="train_")
        self.valid_metric = base_metrics.clone(prefix="valid_")
        self.test_metric = base_metrics.clone(prefix="test_")
        self.float()

    def forward(self, x, pixel_size=None):
        if self.variant == "base":
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.variant == "conditional_normalization":
            x1 = self.inc(x, pixel_size)
            x2 = self.down1(x1, pixel_size)
            x3 = self.down2(x2, pixel_size)
            x4 = self.down3(x3, pixel_size)
            x5 = self.down4(x4, pixel_size)
            x = self.up1(x5, x4, pixel_size)
            x = self.up2(x, x3, pixel_size)
            x = self.up3(x, x2, pixel_size)
            x = self.up4(x, x1, pixel_size)
            logits = self.outc(x)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
        return logits

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metric, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_step(batch, self.valid_metric, "valid")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_step(batch, self.test_metric, "test")

    def _shared_step(self, batch, metric, mode):
        if not hasattr(self, "nan_batch_counter"):
            self.nan_batch_counter = 0

        def log_and_save_nan(reason: str, data_dict: dict):
            self.nan_batch_counter += 1
            print(
                f"[{mode.upper()}] Skipping batch due to {reason} (skipped: {self.nan_batch_counter})"
            )
            self.log(
                f"{mode}/nan_batches",
                self.nan_batch_counter,
                prog_bar=True,
                logger=True,
            )
            os.makedirs("nan_batches", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nan_batches/{mode}_nan_{self.nan_batch_counter}_{reason.replace(' ', '_')}_{timestamp}.pt"
            torch.save(data_dict, filename)

        x, pixel_size, y = batch

        if torch.isnan(x).any() or torch.isinf(x).any():
            log_and_save_nan("invalid_input_x", {"x": x, "y": y})
            return None
        if torch.isnan(y).any() or torch.isinf(y).any():
            log_and_save_nan("invalid_input_y", {"x": x, "y": y})
            return None

        if self.variant == "base":
            y_pred = self(x)
        else:
            y_pred = self(x, pixel_size)

        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            log_and_save_nan("invalid prediction", {"x": x, "y": y})
            return None

        if self.segmentation_type == "binary_segmentation":
            y = y.float()
            y = y.squeeze(1)
            y_pred = y_pred.squeeze(1)
            loss = self.criterion(y_pred, y)
            probs = torch.sigmoid(y_pred)
            preds = (probs > 0.5).float()
        else:  # multiclass
            y = y.long()
            y = y.squeeze(1)
            loss = self.criterion(y_pred, y)
            probs = torch.softmax(y_pred, dim=1)
            preds = torch.argmax(probs, dim=1)

        self.log(f"{mode}_loss", loss, prog_bar=True)
        metrics_result = metric(preds, y)
        for key, value in metrics_result.items():
            self.log(f"{key}", value, prog_bar=True)

        if torch.isnan(loss) or torch.isinf(loss):
            log_and_save_nan("invalid loss", {"x": x, "y": y})
            return None

        return loss

    def configure_optimizers(self):
        optimizer = instantiate_class(
            [{"params": self.parameters()}], self.optimizer_init_config
        )
        if self.lr_scheduler_init_config is None:
            return optimizer
        scheduler = instantiate_class(optimizer, self.lr_scheduler_init_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_init_config.get("interval", "epoch"),
            },
        }


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = 2
    input_size = (3, 224, 224)
    mask_size = (224, 224)

    print("=" * 50)
    print("Testing UNet2D with variant='base'")
    print("=" * 50)

    model_base = UNet(
        segmentation_type="multiclass_segmentation",
        n_channels=3,
        n_classes=classes,
        features_start=32,
        variant="base",
    ).to(device)

    x = torch.randn(BATCH_SIZE, *input_size).to(device)
    y = torch.randint(0, classes, (BATCH_SIZE, *mask_size)).to(device)

    summary(model_base, input_size=(BATCH_SIZE, *input_size))

    y_pred = model_base(x)
    print("Output shape:", y_pred.shape)
    pred_mask = torch.argmax(y_pred, dim=1)
    print("Pred mask shape:", pred_mask.shape)
    print("Base variant test passed!\n")

    print("=" * 50)
    print("Testing UNet2D with variant='conditional_normalization'")
    print("=" * 50)

    model_cn = UNet(
        segmentation_type="multiclass_segmentation",
        n_channels=3,
        n_classes=classes,
        features_start=32,
        variant="conditional_normalization",
        num_conditions=2,
    ).to(device)

    pixel_size = torch.randn(BATCH_SIZE, 2).to(device)

    summary(model_cn, input_size=[(BATCH_SIZE, *input_size), (BATCH_SIZE, 2)])

    y_pred = model_cn(x, pixel_size)
    print("Output shape:", y_pred.shape)
    pred_mask = torch.argmax(y_pred, dim=1)
    print("Pred mask shape:", pred_mask.shape)
    print("Conditional normalization variant test passed!\n")

    print("=" * 50)
    print("All UNet2D variant tests passed successfully!")
    print("=" * 50)
