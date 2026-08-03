import ast
import torch as tc
import pandas as pd
from typing import Tuple
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from datasets.single_wsi_dataset.abstract.abstract_single_wsi_dataset import (
    DownstreamTask,
)
from datasets.single_wsi_dataset.histopathology_transform import TransformConfig
from datasets.training_dataset import TrainingDataset
from datasets.config import MIN_PIXEL_SIZE, MAX_PIXEL_SIZE


class MagInv_DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir_path: str,
        dataset_name: str,
        img_size: Tuple[int, int] = (256, 256),
        batch_size: int = 2,
        num_workers: int = 4,
        train_data_csv: str = "data.csv",
        val_data_csv: str = "data.csv",
        test_data_csv: str = "data.csv",
        pixel_size: Tuple[float, float] | None = None,
        num_patches: int = 10000,
        downstream_task: DownstreamTask = DownstreamTask.NONE,
        exclude_background_in_classification_targets: bool = True,
        min_pixel_size: float = MIN_PIXEL_SIZE,
        max_pixel_size: float = MAX_PIXEL_SIZE,
        seed: int = 2024,
    ):
        """
        Base Data Module
        :arg
            Dataset: Enter Dataset
            batch_size: Enter batch size
            num_workers: Enter number of workers
            size: Enter resized image
            data_root: Enter root data folder name
            valid_ratio: Enter valid dataset ratio
            seed: Seed for the train/val samplers, so patch sampling order is reproducible
        """
        super().__init__()
        self.data_dir_path = data_dir_path
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_csv = train_data_csv
        self.val_data_csv = val_data_csv
        self.test_data_csv = test_data_csv
        self.pixel_size = pixel_size
        self.num_patches = num_patches
        self.seed = seed
        self.downstream_task = downstream_task
        self.exclude_background_in_classification_targets = (
            exclude_background_in_classification_targets
        )
        self.min_pixel_size = min_pixel_size
        self.max_pixel_size = max_pixel_size
        print(
            f"DataModule initialized with min_pixel_size={self.min_pixel_size} and max_pixel_size={self.max_pixel_size}"
        )
        self._num_classes = None
        self._dataset_class_weights = None
        self._downstream_task_detailed = None
        self._mask_mapping = {
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            4: 2,
            5: 2,
        }  # old label - new label
        self._priority_class = 2  # after mask mapping
        self._sample_background_patches = False
        self._background_label = 0
        self._train_transform_config = self.get_train_transforms()
        self._test_transform_config = self.get_test_transforms()
        self._min_samples_per_class = 1000

    def get_downstream_task_detailed(self):
        if self._downstream_task_detailed == None:
            print("Call method setup('fit') firstly to set the values.")
        return self._downstream_task_detailed

    def get_num_classes(self):
        if self._num_classes == None:
            print("Call method setup('fit') firstly to set the values.")
        return self._num_classes

    def get_class_weights(self):
        if self._dataset_class_weights == None:
            print("Call method setup('fit') firstly to set the values.")
        return self._dataset_class_weights

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.train_ds = TrainingDataset(
                data_dir_path=Path(self.data_dir_path),
                csv_file_path=Path(self.train_data_csv),
                patch_size=self.img_size,
                should_balance_batch=True,
                transform_config=self._train_transform_config,
                pixel_size=self.pixel_size,
                num_patches=self.num_patches,
                downstream_task=self.downstream_task,
                exclude_background_in_classification_targets=self.exclude_background_in_classification_targets,
                background_label=self._background_label,
                priority_class=self._priority_class,
                sample_background_patches=self._sample_background_patches,
                min_samples_per_class=self._min_samples_per_class,
                min_pixel_size=self.min_pixel_size,
                max_pixel_size=self.max_pixel_size,
            )

            self._num_classes = self.train_ds.get_num_class()
            self._dataset_class_weights = self.train_ds.get_class_weights()
            self._downstream_task_detailed = (
                self.train_ds.get_detailed_downstream_task()
            )

            self.valid_ds = TrainingDataset(
                data_dir_path=Path(self.data_dir_path),
                csv_file_path=Path(self.val_data_csv),
                patch_size=self.img_size,
                should_balance_batch=False,
                transform_config=self._test_transform_config,
                pixel_size=self.pixel_size,
                num_patches=self.num_patches // 2,
                downstream_task=self.downstream_task,
                exclude_background_in_classification_targets=self.exclude_background_in_classification_targets,
                priority_class=self._priority_class,
                sample_background_patches=self._sample_background_patches,
                background_label=self._background_label,
                min_samples_per_class=self._min_samples_per_class,
                min_pixel_size=self.min_pixel_size,
                max_pixel_size=self.max_pixel_size,
            )

        elif stage in (None, "test", "predict"):
            self.test_ds = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        generator = tc.Generator().manual_seed(self.seed)
        weights_raw = self.train_ds.get_unnormalized_sample_weights()
        if weights_raw is None:
            print(
                "Info: unnormalized sample weights unavailable — falling back to RandomSampler."
            )
            sampler = RandomSampler(
                self.train_ds,
                replacement=True,
                num_samples=self.train_ds.get_num_patches(),
                generator=generator,
            )
        else:
            weights = tc.as_tensor(weights_raw, dtype=tc.float32)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=self.train_ds.get_num_patches(),
                replacement=True,
                generator=generator,
            )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # We use a random sampler only during training/early validation
        # to quickly and randomly select patches from WSI.
        # For final evaluation/testing, a deterministic loader should be used!

        sampler = RandomSampler(
            self.valid_ds,
            replacement=True,
            num_samples=self.valid_ds.get_num_patches(),
            generator=tc.Generator().manual_seed(self.seed),
        )

        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # Test/inference handled outside Lightning
        return None

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        # Test/inference handled outside Lightning
        return None

    def get_train_transforms(
        self,
    ) -> TransformConfig:
        """
        Returns a TransformConfig instance with typical
        augmentations used for training, including mask mapping.
        """
        train_config = TransformConfig(
            hflip_enable=True,
            hflip_prob=0.5,
            vflip_enable=True,
            vflip_prob=0.5,
            rotation_enable=True,
            z_norm=True,
            apply_mask_mapping=True,
            mask_mapping=self._mask_mapping,
            background_label=self._background_label,
        )
        return train_config

    def get_test_transforms(
        self,
    ) -> TransformConfig:
        """
        Returns a TransformConfig instance with typical
        augmentations used for testing, including mask mapping.
        """
        val_config = TransformConfig(
            hflip_enable=False,
            hflip_prob=0.5,
            vflip_enable=False,
            vflip_prob=0.5,
            rotation_enable=False,
            z_norm=True,
            apply_mask_mapping=True,
            mask_mapping=self._mask_mapping,
            background_label=self._background_label,
        )
        return val_config
