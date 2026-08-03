import os
import nrrd
import torch
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from typing import Tuple, Literal
from paths.paths import TEMP_PATH
from torch.utils.data import DataLoader
from networks.unet.unet import UNet
from datasets.single_wsi_dataset.inference_single_wsi_dataset import (
    InferenceSingleWSIDataset,
)
from datasets.single_wsi_dataset.histopathology_transform import TransformConfig
from evaluation.utils.evaluators.abstract.abstract_evaluator import AbstractEvaluator
from datasets.single_wsi_dataset.training_single_wsi_dataset import DownstreamTask
from training.evaluation_timer import EvaluationTimer

from torchmetrics import MetricCollection

# PC
# from torchmetrics.segmentation import DiceScore as Dice
# from torchmetrics.classification import JaccardIndex

# HPC
from torchmetrics import Dice
from torchmetrics import JaccardIndex


class UnetEvaluator(AbstractEvaluator):
    def __init__(
        self,
        data_dir_path: Path,
        output_dir_path: Path,
        output_mask_path: Path,
        model_type: Literal[
            "unet",
            "unetCN",
        ],
        unique_name: str,
        model_checkpoint_path: Path,
        csv_file_path: Path,
        patch_size: Tuple[int, int],
        stride,
        batch_size,
        transform_config: TransformConfig | None = None,
        tested_pixel_size: Tuple[float, float] | None = None,
        trained_pixel_size: Tuple[float, float] | None = None,
        downstream_task: DownstreamTask = DownstreamTask.NONE,
        exclude_background_in_classification_targets: bool = True,
        background_label: int = 0,
        sample_background_patches: bool = False,
        priority_class: int | None = None,
        pixel_size_tolerance_percent_coeff: Tuple[float, float] = (0.05, 0.05),
        n_channels: int = 3,
        features_start: int = 64,
        bilinear: bool = False,
        image_channels: int = 3,
        padding_value: int = 255,
        class_weights_dict={},
        optimizer_init={},
        lr_scheduler_init={},
        num_conditions: int | None = 2,
    ) -> None:
        self.class_weights_dict = class_weights_dict
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.n_channels = n_channels
        self.features_start = features_start
        self.bilinear = bilinear
        self.image_channels = image_channels
        self.num_conditions = num_conditions
        self.stride = stride
        self.batch_size = batch_size
        self.threshold = 0.5
        self.output_mask_path = output_mask_path

        super().__init__(
            data_dir_path=data_dir_path,
            output_dir_path=output_dir_path,
            model_checkpoint_path=model_checkpoint_path,
            model_type=model_type,
            unique_name=unique_name,
            csv_file_path=csv_file_path,
            patch_size=patch_size,
            transform_config=transform_config,
            tested_pixel_size=tested_pixel_size,
            trained_pixel_size=trained_pixel_size,
            downstream_task=downstream_task,
            exclude_background_in_classification_targets=exclude_background_in_classification_targets,
            background_label=background_label,
            sample_background_patches=sample_background_patches,
            priority_class=priority_class,
            pixel_size_tolerance_percent_coeff=pixel_size_tolerance_percent_coeff,
            padding_value=padding_value,
        )

        METRIC_CONFIG = {
            "binary_segmentation": {
                "task": "binary",
                "num_classes": 1,
                "average": "macro",
            },
            "multiclass_segmentation": {
                "task": "multiclass",
                "num_classes": self._num_classes,
                "average": "macro",
            },
        }

        config = METRIC_CONFIG[self._downstream_task_detailed]

        self.global_metrics = MetricCollection(
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
        ).to(self.device)

        self.patch_metrics = MetricCollection(
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
        ).to(self.device)

    def run_evaluation(self):
        test_summary = []
        print("Loading model ... ")
        self.global_metrics.reset()

        timer = EvaluationTimer(self.output_csv_path.parent)
        timer.on_eval_start()

        with torch.no_grad():
            temp_dir_path = Path(TEMP_PATH)
            temp_dir_path.mkdir(parents=True, exist_ok=True)

            for idx in range(len(self._loaded_data)):
                img_path_str = str(self._loaded_data.iloc[idx]["image_path"])
                img_path = self.data_dir_path / Path(img_path_str)

                print(
                    f"Evaluating WSI {idx} / {len(self._loaded_data)}: {img_path_str}"
                )

                tmp_folder = tempfile.mkdtemp(dir=temp_dir_path)
                tmp_folder_path = Path(tmp_folder)
                tmp_img_path = tmp_folder_path / Path(img_path.name)
                shutil.copy(img_path, tmp_img_path)

                try:
                    testing_dataset = InferenceSingleWSIDataset(
                        img_path=tmp_img_path,
                        pixel_size=(self.tested_pixel_size, self.tested_pixel_size),
                        patch_size=self.patch_size,
                        bg_mask_path=self.data_dir_path
                        / Path(self._loaded_data.iloc[idx]["mask_tissue_path"]),
                        stride=self.stride,
                        transform_config=self.transform_config,
                        downstream_task=self.downstream_task,
                        ground_truth_mask_path=self.data_dir_path
                        / Path(self._loaded_data.iloc[idx]["dense_label_path"]),
                        image_label=self._loaded_data.iloc[idx]["weak_label"],
                        image_gt_all_labels=self._loaded_data.iloc[idx][
                            "all_labels_in_dataset"
                        ],
                        exclude_background_in_classification_targets=self.exclude_background_in_classification_targets,
                        background_label=self.background_label,
                        sample_background_patches=self.sample_background_patches,
                        priority_class=self.priority_class,
                        pixel_size_tolerance_percent_coeff=self.pixel_size_tolerance_percent_coeff,
                        padding_value=self.padding_value,
                    )

                    testing_dataloader = DataLoader(
                        testing_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                    )

                    n_rows, n_cols = testing_dataset.get_grid_shape()
                    patch_h, patch_w = self.patch_size
                    if self.stride is None:
                        self.stride = self.patch_size

                    full_h = (n_rows - 1) * self.stride[0] + patch_h
                    full_w = (n_cols - 1) * self.stride[1] + patch_w

                    dtype = (
                        torch.float32
                        if self._downstream_task_detailed == "binary_segmentation"
                        else torch.long
                    )

                    full_mask = torch.zeros(
                        (full_h, full_w), dtype=dtype, device=self.device
                    )
                    full_gt_mask = torch.zeros(
                        (full_h, full_w), dtype=dtype, device=self.device
                    )

                    img_index = 0

                    for batch in testing_dataloader:
                        try:
                            x, pixel_size, y = batch
                            x = x.to(self.device)
                            pixel_size = pixel_size.to(self.device)
                            y = y.to(self.device)
                            timer.on_batch_data_ready(x.shape[0])

                            if self.model_type == "unet":
                                y_pred = self.model(x)
                            elif self.model_type in ["unetCN"]:
                                y_pred = self.model(x, pixel_size)
                            else:
                                raise ValueError(f"UNKNOWN MODEL TYPE {self.model_type} IN UNET EVALUATOR")

                            timer.on_batch_end()
                            if timer.limit_reached:
                                break

                            if self._downstream_task_detailed == "binary_segmentation":
                                y = y.float().squeeze(1)
                                y_pred = y_pred.squeeze(1)
                                probs = torch.sigmoid(y_pred)
                                preds = (probs > self.threshold).float()
                            else:
                                y = y.long().squeeze(1)
                                probs = torch.softmax(y_pred, dim=1)
                                preds = torch.argmax(probs, dim=1)

                            self.global_metrics.update(preds, y)

                            for b in range(x.size(0)):
                                patch_row, patch_col = (
                                    testing_dataset.get_patch_position(img_index)
                                )

                                current_pred = preds[b]
                                current_gt = y[b]

                                row_start = patch_row * self.stride[0]
                                col_start = patch_col * self.stride[1]
                                row_end = row_start + current_pred.shape[0]
                                col_end = col_start + current_pred.shape[1]

                                full_mask[row_start:row_end, col_start:col_end] = (
                                    current_pred
                                )
                                full_gt_mask[row_start:row_end, col_start:col_end] = (
                                    current_gt
                                )

                                self.patch_metrics.reset()
                                self.patch_metrics.update(
                                    current_pred.unsqueeze(0),
                                    current_gt.unsqueeze(0),
                                )
                                patch_result = self.patch_metrics.compute()

                                test_summary.append(
                                    {
                                        "img_path": img_path_str,
                                        "patch_idx": img_index,
                                        "patch_size": self.patch_size,
                                        "pixel_size": pixel_size[b].cpu()[0].item(),
                                        "model_trained_on_pixel_size": (
                                            self.trained_pixel_size
                                            if self.trained_pixel_size
                                            else "None"
                                        ),
                                        "stride": (
                                            self.stride if self.stride else "None"
                                        ),
                                        "dice_patch": patch_result["dice"].item(),
                                        "iou_patch": patch_result["iou"].item(),
                                    }
                                )

                                img_index += 1

                        except Exception as e:
                            print(
                                f"[WARNING] Batch failed | WSI={img_path_str} | patch_idx={img_index} | {repr(e)}"
                            )
                            continue

                    main_folder = (
                        self.output_mask_path / Path(self.output_csv_path).stem
                    )
                    os.makedirs(main_folder, exist_ok=True)

                    img_stem = Path(img_path_str).stem
                    header = {"encoding": "gzip"}

                    if self._downstream_task_detailed == "binary_segmentation":
                        pred_path = (
                            main_folder / f"{img_stem}_preds_thr_{self.threshold}.nrrd"
                        )
                    else:
                        pred_path = main_folder / f"{img_stem}_preds.nrrd"

                    nrrd.write(str(pred_path), full_mask.cpu().numpy(), header=header)
                    nrrd.write(
                        str(main_folder / f"{img_stem}_gt.nrrd"),
                        full_gt_mask.cpu().numpy(),
                        header=header,
                    )

                    print(f"Saved WSI outputs for {img_stem}")

                except Exception as e:
                    print(f"[ERROR] Failed to process WSI {img_path_str} | {repr(e)}")
                    continue
                finally:
                    shutil.rmtree(tmp_folder_path)

                if timer.limit_reached:
                    break

        timer.save()

        global_results = self.global_metrics.compute()
        print("Global metrics on entire dataset:", global_results)

        global_metrics_path = (
            self.output_csv_path.parent
            / f"{self.output_csv_path.stem}_global_metrics.csv"
        )

        df_global_metrics = pd.DataFrame(
            {
                metric: [value.item() if torch.is_tensor(value) else value]
                for metric, value in global_results.items()
            }
        )

        df_global_metrics.to_csv(global_metrics_path, index=False)

        df_test_summary = pd.DataFrame(test_summary)
        df_test_summary.to_csv(self.output_csv_path, index=False)
        print(f"Saved detailed test summary to {self.output_csv_path}")
        print("Evaluation completed.")

    def _get_model(self):
        if self.model_type == "unet":
            return UNet.load_from_checkpoint(
                strict=True,
                checkpoint_path=self.model_checkpoint_path,
                segmentation_type=self._downstream_task_detailed,
                n_channels=self.n_channels,
                features_start=self.features_start,
                bilinear=self.bilinear,
                n_classes=self._num_classes,
                class_weights_dict=self.class_weights_dict,
                optimizer_init=self.optimizer_init,
                lr_scheduler_init=self.lr_scheduler_init,
                variant="base",
            )
        elif self.model_type == "unetCN":
            return UNet.load_from_checkpoint(
                strict=True,
                checkpoint_path=self.model_checkpoint_path,
                segmentation_type=self._downstream_task_detailed,
                n_channels=self.n_channels,
                features_start=self.features_start,
                bilinear=self.bilinear,
                n_classes=self._num_classes,
                class_weights_dict=self.class_weights_dict,
                optimizer_init=self.optimizer_init,
                lr_scheduler_init=self.lr_scheduler_init,
                variant="conditional_normalization",
                num_conditions=self.num_conditions,
            )
        else:
            raise ValueError(f"UNKNOWN MODEL TYPE {self.model_type} IN UNET EVALUATOR")
