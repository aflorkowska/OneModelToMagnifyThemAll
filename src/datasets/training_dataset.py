import torch as tc
import numpy as np
from typing import Tuple
from pathlib import Path
from torch.utils.data import Dataset

from datasets.single_wsi_dataset.training_single_wsi_dataset import (
    TrainingSingleWSIDataset,
)
from datasets.single_wsi_dataset.abstract.abstract_single_wsi_dataset import (
    DownstreamTask,
)
from datasets.single_wsi_dataset.histopathology_transform import TransformConfig
from datasets.config import MIN_PIXEL_SIZE, MAX_PIXEL_SIZE
from csv_loaders.training_data_csv_loader import TrainingDataCSVLoader
from utils.downstreamtask_utils import DownstreamTaskUtils


class TrainingDataset(Dataset):
    def __init__(
        self,
        data_dir_path: Path,
        csv_file_path: Path,
        patch_size: Tuple[int, int],
        should_balance_batch: bool,
        transform_config: TransformConfig | None = None,
        pixel_size: Tuple[float, float] | None = None,
        num_patches: int = 10000,
        downstream_task: DownstreamTask = DownstreamTask.NONE,
        exclude_background_in_classification_targets: bool = True,
        background_label: int = 0,
        sample_background_patches: bool = False,
        priority_class: int | None = None,
        min_samples_per_class=1000,
        min_pixel_size: float = MIN_PIXEL_SIZE,
        max_pixel_size: float = MAX_PIXEL_SIZE,
    ) -> None:
        """
        Initializes the dataset for training, capable of providing patches from multiple WSI images as specified in a CSV file.

        Parameters
        ----------
        data_dir_path : Path
            Absolute path to the dir with dataset.
        csv_file_path : Path
            Relative path to the CSV file containing the relative paths to the WSI files and their corresponding masks.
            It has to contain the following columns: image_path, mask_tissue_path, dense_path, weak_label, all_labels_in_dataset.
        pixel_size : Tuple[float, float] | None
            Desired pixel size for the patches.
            Passed pixel size means constant pixel size mode.
            If None, pixel size is randomly selected from range [min_pixel_size, max_pixel_size].
        patch_size : Tuple[int, int]
            Size of the patches to extract.
        should_balance_batch : bool
            Flag. If true, balanced patch samples is active. If false, it is about random sampling all significant pixels.
        stride : Optional[Tuple[int, int]]
            Stride between patches.
            None (default) means that stride is equal to patch_size.
        transform_config: TransformConfig | None = None
            Configuration for the histopathology transformations to apply to the patches. If None, default configuration is used.
        num_patches : int
            Number of patches to extract per epoch.
        downstream_task : DownstreamTask = DownstreamTask.NONE
            Selected downstream task:
                Classification:
                    weak (label for each wsi image)
                    strong (label for each patch in the WSI image)
                Segmentation
                None (interferance).
        min_pixel_size : float
            Minimum pixel size for random selection. Default is MIN_PIXEL_SIZE from config.py.
        max_pixel_size : float
            Maximum pixel size for random selection. Default is MAX_PIXEL_SIZE from config.py.
        """

        self.data_dir_path = data_dir_path
        self.csv_file_path = self.data_dir_path / csv_file_path
        self.patch_size = patch_size
        self.should_balance_batch = should_balance_batch
        self.transform_config = transform_config
        self.pixel_size = pixel_size
        self.num_patches = num_patches
        self.downstream_task = DownstreamTask(downstream_task)
        self.priority_class = priority_class
        self.exclude_background_in_classification_targets = (
            exclude_background_in_classification_targets
        )
        self.background_label = background_label
        self.sample_background_patches = sample_background_patches
        self.min_samples_per_class = min_samples_per_class
        self.min_pixel_size = min_pixel_size
        self.max_pixel_size = max_pixel_size
        self._PATCH_VALIDITY_CONFIG = [
            {"max_pixel_size": 1.0, "min_fraction": 0.2},  # pixel size < 1.0
            {"max_pixel_size": 1.5, "min_fraction": 0.15},  # 1.0 <= pixel size < 1.5
            {"max_pixel_size": float("inf"), "min_fraction": 0.1},  # pixel size >= 1.5
        ]

        loader = TrainingDataCSVLoader()
        self._loaded_data, _ = loader.load_csv(
            csv_file_path=self.csv_file_path,
            data_dir_path=self.data_dir_path,
            downstream_task=self.downstream_task,
            transform_config=self.transform_config,
            should_balance_batch=self.should_balance_batch,
            background_label=self.background_label,
            sample_background_patches=self.sample_background_patches,
            min_samples_per_class=self.min_samples_per_class,
        )
        self._selected_images_dataloaders = {}
        self._num_classes = DownstreamTaskUtils.calculate_num_classes(self)
        self._downstream_task_detailed = (
            DownstreamTaskUtils.set_detailed_downstream_task(self)
        )
        self._class_weights = None
        self._unnormalized_sample_weights = None
        if self.should_balance_batch:
            self._class_weights = self._loaded_data.iloc[0]["dataset_class_weights"]
            self._unnormalized_sample_weights = self._loaded_data[
                "class_probability"
            ].values.astype(float)

    def get_min_pixel_fraction(self, pixel_size: float):
        """
        Returns the minimum pixel fraction for a given pixel_size based on the configuration.

        Args:
            pixel_size (float): Pixel size in μm/pixel.

        Returns:
            float: Minimum fraction of valid pixels for a patch.
        """
        for entry in self._PATCH_VALIDITY_CONFIG:
            if pixel_size < entry["max_pixel_size"]:
                return entry["min_fraction"]

        # This should theoretically never be reached because the last threshold is infinity
        return 0.0

    def get_num_patches(self):
        return self.num_patches

    def get_detailed_downstream_task(self):
        return self._downstream_task_detailed

    def _set_detailed_downstream_task(self):
        if self.downstream_task == DownstreamTask.SEGMENTATION:
            if self._num_classes == 2:
                return "binary_segmentation"
            return "multiclass_segmentation"
        if self.downstream_task == DownstreamTask.NONE:
            return None
        if self.downstream_task == DownstreamTask.WEAK_CLASSIFICATION:
            return "weak_classification"
        if self.downstream_task == DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION:
            return "multilabel_classification"
        if (
            self.downstream_task
            == DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION
        ):
            if self._num_classes == 2:
                return "binary_classification"
            return "multiclass_classification"

    def get_num_class(self):
        return self._num_classes

    def get_class_weights(self):
        return self._class_weights

    def get_unnormalized_sample_weights(self):
        return self._unnormalized_sample_weights

    def get_selected_WSI_dataloader(self):
        return self._selected_images_dataloaders

    def _calculate_num_classes(
        self,
    ):

        if self.downstream_task in (
            DownstreamTask.SEGMENTATION,
            DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
            DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
        ):

            image_gt_all_labels = self._loaded_data.iloc[0]["all_labels_in_dataset"]
            if self.transform_config is not None:
                if (
                    self.transform_config.apply_mask_mapping
                    and self.transform_config.mask_mapping
                ):
                    mapped = set(
                        [
                            self.transform_config.mask_mapping[x]
                            for x in image_gt_all_labels
                        ]
                    )
                    image_gt_all_labels = list(dict.fromkeys(mapped))

            if self.downstream_task in (
                DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
                DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
            ):
                if self.exclude_background_in_classification_targets:
                    image_gt_all_labels = np.delete(
                        image_gt_all_labels, self.background_label
                    )
            return len(image_gt_all_labels)

        if self.downstream_task == DownstreamTask.WEAK_CLASSIFICATION:
            # Warning: the implementation below assumes weak labels are 0-indexed,
            # consecutive (no gaps) and sorted ascending. If your weak labels do
            # not follow this format, consider normalizing them first.
            print(
                "WARNING: calculating number of classes for weak classification - assumes weak labels start at 0, are ascending and consecutive (no gaps)."
            )
            all_weak = self._loaded_data["weak_label"].tolist()
            unique_weak = np.unique(all_weak)
            return len(unique_weak)

        if self.downstream_task == DownstreamTask.NONE:
            return None

    def __len__(self) -> int:
        return len(self._loaded_data)

    def __getitem__(self, idx: int) -> tc.Tensor:

        wsi_idx = min(idx, len(self._loaded_data) - 1)
        expected_label = (
            int(self._loaded_data.iloc[wsi_idx]["splitted_label"])
            if self.should_balance_batch
            else None
        )

        loaded_data_copy = self._loaded_data.copy(deep=True)
        counts = loaded_data_copy["image_path"].value_counts()
        duplicates = counts[counts > 1].index
        loaded_data_copy = loaded_data_copy[
            ~loaded_data_copy["image_path"].isin(duplicates)
        ]

        if self.should_balance_batch:
            valid_indices = loaded_data_copy.index[
                loaded_data_copy["splitted_label"] == expected_label
            ].to_numpy()

            if len(valid_indices) == 0:
                valid_indices = self._loaded_data.index[
                    self._loaded_data["splitted_label"] == expected_label
                ].to_numpy()
        else:
            valid_indices = None

        while True:
            if wsi_idx in self._selected_images_dataloaders:
                single_wsi_dataset = self._selected_images_dataloaders[wsi_idx]
            else:
                single_wsi_dataset = TrainingSingleWSIDataset(
                    img_path=self.data_dir_path
                    / Path(self._loaded_data.iloc[wsi_idx]["image_path"]),
                    patch_size=self.patch_size,
                    should_balance_batch=self.should_balance_batch,
                    bg_mask_path=self.data_dir_path
                    / Path(self._loaded_data.iloc[wsi_idx]["mask_tissue_path"]),
                    transform_config=self.transform_config,
                    downstream_task=self.downstream_task,
                    ground_truth_mask_path=self.data_dir_path
                    / Path(self._loaded_data.iloc[wsi_idx]["dense_label_path"]),
                    image_label=self._loaded_data.iloc[wsi_idx]["weak_label"],
                    image_gt_all_labels=self._loaded_data.iloc[wsi_idx][
                        "all_labels_in_dataset"
                    ],
                    priority_class=self.priority_class,
                    exclude_background_in_classification_targets=self.exclude_background_in_classification_targets,
                    background_label=self.background_label,
                    sample_background_patches=self.sample_background_patches,
                )

                if len(single_wsi_dataset) == 0:
                    if self.should_balance_batch:
                        wsi_idx = np.random.choice(valid_indices)
                        continue
                    else:
                        wsi_idx = np.random.randint(0, len(self._loaded_data))
                        continue

                self._selected_images_dataloaders[wsi_idx] = single_wsi_dataset

            if self.pixel_size is None:
                px = np.random.uniform(
                    self.min_pixel_size, np.nextafter(self.max_pixel_size, float("inf"))
                )
                selected_pixel_sizes = (px, px)
            else:
                selected_pixel_sizes = self.pixel_size

            min_pixel_fraction_for_patch_validity = self.get_min_pixel_fraction(
                selected_pixel_sizes[0]
            )

            if self.should_balance_batch:
                output = single_wsi_dataset[(selected_pixel_sizes, expected_label)]
            else:
                output = single_wsi_dataset[selected_pixel_sizes]

            if self.should_balance_batch:
                expected_label_prep = expected_label
                if self.downstream_task in (
                    DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
                    DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
                ):
                    (
                        patch,
                        pixel_size,
                        label,
                        (dense_labels_original, labels_counts_original),
                    ) = output
                    if label is None:
                        wsi_idx = np.random.choice(valid_indices)
                        continue

                    if self.exclude_background_in_classification_targets:
                        if expected_label > self.background_label:
                            expected_label_prep = expected_label - 1

                    if int(dense_labels_original[expected_label] == 1):
                        pixel_fraction_cls = (
                            labels_counts_original[expected_label].float()
                            / labels_counts_original.sum().float()
                        )
                    else:
                        pixel_fraction_cls = None
                    if (
                        self.downstream_task
                        == DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION
                    ):
                        if (
                            label is not None
                            and pixel_fraction_cls is not None
                            and int(label.item()) == expected_label_prep
                            and pixel_fraction_cls
                            > min_pixel_fraction_for_patch_validity
                        ):
                            break

                    if (
                        self.downstream_task
                        == DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION
                    ):
                        if (
                            label is not None
                            and pixel_fraction_cls is not None
                            and int(label[expected_label_prep]) == 1
                            and pixel_fraction_cls
                            > min_pixel_fraction_for_patch_validity
                        ):
                            break

                if self.downstream_task == DownstreamTask.SEGMENTATION:
                    patch, pixel_size, label = output
                    values, counts = tc.unique(label, return_counts=True)
                    mask = values == expected_label_prep
                    if mask.any():
                        pixel_fraction_seg = counts[mask].item() / counts.sum().item()
                        if pixel_fraction_seg > min_pixel_fraction_for_patch_validity:
                            break

                wsi_idx = np.random.choice(valid_indices)
                continue
            else:
                break

        return output
