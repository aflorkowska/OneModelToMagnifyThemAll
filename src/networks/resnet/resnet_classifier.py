# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import pytorch_lightning as pl
from networks.resnet.models.resnet18 import ResNet18
from networks.resnet.models.resnet34 import ResNet34
from networks.resnet.models.resnet50 import ResNet50
from networks.resnet.models.resnet101 import ResNet101
from networks.resnet.models.resnet152 import ResNet152
from lightning.pytorch.utilities import grad_norm
from pytorch_lightning.cli import instantiate_class
from torchmetrics import MetricCollection, Accuracy, Recall, F1Score


class ResNetClassifier(pl.LightningModule):
    def __init__(
        self,
        resnet_type,
        classification_type,
        image_channels,
        num_classes,
        class_weights_dict,
        optimizer_init,
        lr_scheduler_init,
        variant="base",
        num_conditions=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classification_type = classification_type
        self.variant = variant

        # Validate parameters
        if variant == "conditional_normalization" and num_conditions is None:
            raise ValueError(f"num_conditions is required for variant='{variant}'")

        LOSS_MAP = {
            "binary_classification": nn.BCEWithLogitsLoss,
            "multilabel_classification": nn.BCEWithLogitsLoss,
            "multiclass_classification": nn.CrossEntropyLoss,
        }

        METRIC_CONFIG = {
            "binary_classification": ("binary", {}),
            "multilabel_classification": (
                "multilabel",
                {"num_labels": num_classes, "average": "macro"},
            ),
            "multiclass_classification": (
                "multiclass",
                {"num_classes": num_classes, "average": "macro"},
            ),
        }
        if (
            self.classification_type not in LOSS_MAP
            or self.classification_type not in METRIC_CONFIG
        ):
            raise ValueError(f"Unknown classification_type: {self.classification_type}")

        if self.classification_type == "binary_classification" and num_classes == 2:
            output_num_classes = 1
        else:
            output_num_classes = num_classes

        self.model = self._get_resnet_model(
            resnet_type, image_channels, output_num_classes, variant, num_conditions
        ).float()

        self.optimizer_init_config = optimizer_init
        self.lr_scheduler_init_config = lr_scheduler_init

        class_weights_tensor = None
        if class_weights_dict is not None:
            num_classes_from_dict = len(class_weights_dict)
            class_weights_list = [
                class_weights_dict[i] for i in range(num_classes_from_dict)
            ]
            class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32)

        self.criterion = LOSS_MAP[self.classification_type](weight=class_weights_tensor)
        task, metric_kwargs = METRIC_CONFIG[self.classification_type]

        metrics = MetricCollection(
            {
                name: metric(task=task, **metric_kwargs)
                for name, metric in {
                    "accuracy": Accuracy,
                    "recall": Recall,
                    "f1": F1Score,
                }.items()
            }
        )

        self.train_metric = metrics.clone(prefix="train_")
        self.valid_metric = metrics.clone(prefix="valid_")
        self.test_metric = metrics.clone(prefix="test_")

    def _get_resnet_model(
        self, resnet_type, image_channels, classes, variant, num_conditions
    ):
        match resnet_type:
            case "resnet18":
                return ResNet18(
                    image_channels=image_channels,
                    num_classes=classes,
                    variant=variant,
                    num_conditions=num_conditions,
                )
            case "resnet34":
                return ResNet34(
                    image_channels=image_channels,
                    num_classes=classes,
                    variant=variant,
                    num_conditions=num_conditions,
                )
            case "resnet50":
                return ResNet50(
                    image_channels=image_channels,
                    num_classes=classes,
                    variant=variant,
                    num_conditions=num_conditions,
                )
            case "resnet101":
                return ResNet101(
                    image_channels=image_channels,
                    num_classes=classes,
                    variant=variant,
                    num_conditions=num_conditions,
                )
            case "resnet152":
                return ResNet152(
                    image_channels=image_channels,
                    num_classes=classes,
                    variant=variant,
                    num_conditions=num_conditions,
                )
            case _:
                raise ValueError(f"Unsupported resnet_type: {resnet_type}")

    def forward(self, x, pixel_size=None):
        if self.variant == "base":
            return self.model(x)
        else:
            return self.model(x, pixel_size)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metric, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_step(batch, self.valid_metric, "valid")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._shared_step(batch, self.test_metric, "test")

    # https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model.fc, norm_type=2)
        for key, value in norms.items():
            self.log_dict({key: value}, prog_bar=True)

    def _shared_step(self, batch, metric, mode):
        if not hasattr(self, "nan_batch_counter"):
            self.nan_batch_counter = 0

        def log_and_save_nan(reason: str, data_dict: dict):
            self.nan_batch_counter += 1
            msg = f"[{mode.upper()}] Skipping batch due to {reason} (skipped: {self.nan_batch_counter})"
            print(msg)
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

        x, pixel_size, y, _ = batch
        if torch.isnan(x).any() or torch.isinf(x).any():
            log_and_save_nan(
                "invalid_input_x", {"x": x, "y": y, "pixel_size": pixel_size}
            )
            return None

        y_pred = self(x, pixel_size)

        if self.classification_type == "binary_classification":
            y_pred = y_pred.squeeze(1)

        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            log_and_save_nan(
                "invalid prediction (y_pred)",
                {"x": x, "y": y, "pixel_size": pixel_size},
            )
            return None

        loss = self.criterion(y_pred, y)
        self.log_dict({f"{mode}_loss": loss}, prog_bar=True)

        if self.classification_type in {
            "binary_classification",
            "multilabel_classification",
        }:
            y = y.float()
            probs = torch.sigmoid(y_pred)
            preds = (probs > 0.5).float()
        elif self.classification_type == "multiclass_classification":
            y = y.long()
            probs = torch.softmax(y_pred, dim=1)
            preds = torch.argmax(probs, dim=1)
        else:
            raise ValueError(f"Unknown classification_type: {self.classification_type}")

        metrics_result = metric(preds, y)

        for key, value in metrics_result.items():
            self.log(f"{key}", value, prog_bar=True)

        if self.classification_type == "binary_classification":
            metrics_class0 = metric(1 - preds, 1 - y)
            self.log(
                f"{mode}_recall_class_0",
                metrics_class0[f"{mode}_recall"],
                prog_bar=True,
            )
            self.log(f"{mode}_f1_class_0", metrics_class0[f"{mode}_f1"], prog_bar=True)
            f1_macro = (metrics_result[f"{mode}_f1"] + metrics_class0[f"{mode}_f1"]) / 2
            self.log(f"{mode}_f1_macro", f1_macro, prog_bar=True)

        if torch.isnan(loss) or torch.isinf(loss):
            log_and_save_nan("invalid loss", {"x": x, "y": y, "pixel_size": pixel_size})
            return None

        return loss

    def configure_optimizers(self):
        param_groups = [{"params": self.parameters()}]
        optimizer = instantiate_class(param_groups, self.optimizer_init_config)

        scheduler_cfg = self.lr_scheduler_init_config
        if scheduler_cfg is None:
            return optimizer

        scheduler = instantiate_class(optimizer, scheduler_cfg)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": scheduler_cfg.get("interval", "epoch"),
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


from torchinfo import summary

if __name__ == "__main__":
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = 1000
    input_size = (3, 224, 224)

    input_tensor = torch.randn(BATCH_SIZE, *input_size).to(device)

    print("Testing base variant...")
    net = ResNetClassifier(
        resnet_type="resnet18",
        classification_type="multiclass_classification",
        image_channels=3,
        num_classes=classes,
        class_weights_dict={},
        optimizer_init={},
        lr_scheduler_init={},
        variant="base",
    ).to(device)

    summary(net, input_size=(BATCH_SIZE, 3, 224, 224), device="cuda")

    y = net(input_tensor)
    assert y.size() == torch.Size([BATCH_SIZE, classes])
    print("Output shape:", y.size())

    print("\nTesting conditional_normalization variant...")
    pixel_size = torch.stack(
        [
            torch.full((BATCH_SIZE,), 0.5, device=device),
            torch.full((BATCH_SIZE,), 0.5, device=device),
        ],
        dim=1,
    )

    net_cn = ResNetClassifier(
        resnet_type="resnet18",
        classification_type="multiclass_classification",
        image_channels=3,
        num_classes=classes,
        class_weights_dict={},
        optimizer_init={},
        lr_scheduler_init={},
        variant="conditional_normalization",
        num_conditions=2,
    ).to(device)

    y_cn = net_cn(input_tensor, pixel_size)
    assert y_cn.size() == torch.Size([BATCH_SIZE, classes])
    print("Output shape:", y_cn.size())

    print("\n All variants tested successfully!")
