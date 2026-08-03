import torch
import shutil
import tempfile
import pandas as pd
from typing import Tuple
from pathlib import Path
from typing import Literal
from paths.paths import TEMP_PATH
from torch.utils.data import DataLoader
from networks.resnet.resnet_classifier import ResNetClassifier
from datasets.single_wsi_dataset.inference_single_wsi_dataset import (
    InferenceSingleWSIDataset,
)
from datasets.single_wsi_dataset.histopathology_transform import TransformConfig
from evaluation.utils.evaluators.abstract.abstract_evaluator import AbstractEvaluator
from datasets.single_wsi_dataset.training_single_wsi_dataset import DownstreamTask
from training.evaluation_timer import EvaluationTimer


class ResnetEvaluator(AbstractEvaluator):
    def __init__(
        self,
        data_dir_path: Path,
        output_dir_path: Path,
        model_type: Literal[
            "resnet18",
            "resnetCN18",
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
        padding_value: int = 255,
        class_weights_dict={},
        optimizer_init={},
        lr_scheduler_init={},
        num_conditions: int | None = 2,
    ) -> None:
        self.class_weights_dict = class_weights_dict
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.num_conditions = num_conditions
        self.stride = stride
        self.batch_size = batch_size
        self.threshold = 0.5

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

    def run_evaluation(self):
        test_summary = []
        print("Loading model ... ")

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

                    img_index = 0

                    for batch in testing_dataloader:
                        try:
                            x, pixel_size, y, _ = batch
                            x = x.to(self.device)
                            pixel_size = pixel_size.to(self.device)
                            y = y.to(self.device)
                            timer.on_batch_data_ready(x.shape[0])

                            if self.model_type == "resnet18":
                                y_pred = self.model(x)
                            elif self.model_type in ["resnetCN18"]:
                                y_pred = self.model(x, pixel_size)
                            else:
                                raise ValueError(
                                    f"UNKNOWN MODEL TYPE {self.model_type} IN RESNET EVALUATOR"
                                )

                            timer.on_batch_end()
                            if timer.limit_reached:
                                break

                            if (
                                self._downstream_task_detailed
                                == "binary_classification"
                            ):
                                y_pred = y_pred.squeeze(1)

                            if self._downstream_task_detailed in {
                                "binary_classification",
                                "multilabel_classification",
                            }:
                                y = y.float()
                                probs = torch.sigmoid(y_pred)
                                preds = (probs > self.threshold).float()
                            elif (
                                self._downstream_task_detailed
                                == "multiclass_classification"
                            ):
                                y = y.long()
                                probs = torch.softmax(y_pred, dim=1)
                                preds = torch.argmax(probs, dim=1)
                            else:
                                raise ValueError(
                                    f"Unknown classification_type: {self._downstream_task_detailed}"
                                )

                            for pixel_size_i, y_i, pred_i, prob_i in zip(
                                pixel_size, y, preds, probs
                            ):
                                test_summary.append(
                                    {
                                        "img_path": img_path_str,
                                        "patch_idx": img_index,
                                        "patch_size": self.patch_size,
                                        "pixel_size": pixel_size_i.cpu()[0].item(),
                                        "model_trained_on_pixel_size": (
                                            self.trained_pixel_size
                                            if self.trained_pixel_size is not None
                                            else "None"
                                        ),
                                        "stride": (
                                            self.stride
                                            if self.stride is not None
                                            else "None"
                                        ),
                                        "y_true": y_i.cpu().numpy(),
                                        "y_pred": pred_i.cpu().numpy(),
                                        "probabilities_y_pred": prob_i.detach()
                                        .cpu()
                                        .numpy(),
                                    }
                                )
                                img_index += 1

                        except Exception as e:
                            print(
                                f"[WARNING] Batch failed | WSI={img_path_str} | patch_idx={img_index} | {repr(e)}"
                            )
                            continue

                except Exception as e:
                    print(f"[ERROR] Failed to process WSI {img_path_str} | {repr(e)}")
                    continue
                finally:
                    shutil.rmtree(tmp_folder_path)

                if timer.limit_reached:
                    break

        timer.save()

        df_test_summary = pd.DataFrame(test_summary)
        print("Saving ... ", self.output_csv_path)
        df_test_summary.to_csv(self.output_csv_path, index=False)

    def _get_model(self):
        if self.model_type == "resnet18":
            return ResNetClassifier.load_from_checkpoint(
                strict=True,
                checkpoint_path=self.model_checkpoint_path,
                resnet_type="resnet18",
                classification_type=self._downstream_task_detailed,
                image_channels=3,
                num_classes=self._num_classes,
                class_weights_dict=self.class_weights_dict,
                optimizer_init=self.optimizer_init,
                lr_scheduler_init=self.lr_scheduler_init,
                variant="base",
            )
        elif self.model_type == "resnetCN18":
            return ResNetClassifier.load_from_checkpoint(
                strict=True,
                checkpoint_path=self.model_checkpoint_path,
                resnet_type="resnet18",
                classification_type=self._downstream_task_detailed,
                image_channels=3,
                num_classes=self._num_classes,
                class_weights_dict=self.class_weights_dict,
                optimizer_init=self.optimizer_init,
                lr_scheduler_init=self.lr_scheduler_init,
                variant="conditional_normalization",
                num_conditions=self.num_conditions,
            )
        else:
            raise ValueError(f"UNKNOWN MODEL TYPE {self.model_type} IN RESNET EVALUATOR")
