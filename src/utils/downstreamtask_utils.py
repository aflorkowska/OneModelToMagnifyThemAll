from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets.training_dataset import TrainingDataset
    from evaluation.evaluators.resnet_evaluator import ResnetEvaluator
    from evaluation.evaluators.unet_evaluator import UnetEvaluator

import numpy as np
from datasets.single_wsi_dataset.abstract.abstract_single_wsi_dataset import (
    DownstreamTask,
)


class DownstreamTaskUtils:
    @staticmethod
    def set_detailed_downstream_task(
        dataset: ResnetEvaluator | UnetEvaluator | TrainingDataset,
    ):
        dt = dataset.downstream_task
        nc = dataset._num_classes

        if dt == DownstreamTask.SEGMENTATION:
            return "binary_segmentation" if nc == 2 else "multiclass_segmentation"
        if dt == DownstreamTask.NONE:
            return None
        if dt == DownstreamTask.WEAK_CLASSIFICATION:
            return "weak_classification"
        if dt == DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION:
            return "multilabel_classification"
        if dt == DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION:
            return "binary_classification" if nc == 2 else "multiclass_classification"

    @staticmethod
    def calculate_num_classes(
        dataset: ResnetEvaluator | UnetEvaluator | TrainingDataset,
    ):
        dt = dataset.downstream_task

        if dt in (
            DownstreamTask.SEGMENTATION,
            DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
            DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
        ):
            image_gt_all_labels = dataset._loaded_data.iloc[0]["all_labels_in_dataset"]

            if getattr(dataset, "transform_config", None):
                if (
                    dataset.transform_config.apply_mask_mapping
                    and dataset.transform_config.mask_mapping
                ):
                    mapped = set(
                        dataset.transform_config.mask_mapping[x]
                        for x in image_gt_all_labels
                    )
                    image_gt_all_labels = list(dict.fromkeys(mapped))

            if dt in (
                DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
                DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
            ):
                if getattr(
                    dataset, "exclude_background_in_classification_targets", False
                ):
                    image_gt_all_labels = np.delete(
                        image_gt_all_labels, dataset.background_label
                    )

            return len(image_gt_all_labels)

        if dt == DownstreamTask.WEAK_CLASSIFICATION:
            print(
                "WARNING: calculating number of classes for weak classification - assumes weak labels start at 0, are ascending and consecutive (no gaps)."
            )
            all_weak = dataset._loaded_data["weak_label"].tolist()
            unique_weak = np.unique(all_weak)
            return len(unique_weak)
