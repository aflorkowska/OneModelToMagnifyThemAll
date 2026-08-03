import os
import sys

######################### WINDOWS
import platform

if platform.system() == "Windows":
    from paths.paths import OPENSLIDE_BIN_DIR

    os.add_dll_directory(OPENSLIDE_BIN_DIR)
import openslide

################################################################
import numpy as np
import scipy.ndimage as nd

from PIL import Image
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Union, List


from datasets.single_wsi_dataset.histopathology_transform import (
    HistopathologyTransform,
    TransformConfig,
)

import torch
from torch.utils.data import Dataset


class DownstreamTask(Enum):
    STRONG_MULTILABEL_CLASSIFICATION = (
        1  # multilabel task - one-hot vector label for each patch
    )
    STRONG_BINARY_MULTICLASS_CLASSIFICATION = (
        2  # binary or multiclass task - single label for each patch
    )
    WEAK_CLASSIFICATION = 3  # label for each WSI image
    SEGMENTATION = 4
    NONE = 5


class AbstractSingleWSIDataset(Dataset, ABC):
    def __init__(
        self,
        img_path: Path,
        patch_size: Tuple[int, int],
        bg_mask_path: Path | None = None,
        transform_config: TransformConfig | None = None,
        pixel_size_tolerance_percent_coeff: Tuple[float, float] = (0.05, 0.05),
        downstream_task: DownstreamTask = DownstreamTask.NONE,
        ground_truth_mask_path: Path | None = None,
        image_label: List[int] | None = None,
        image_gt_all_labels: List[int] | None = None,
        priority_class: int | None = None,
        padding_value: int = 255,
        exclude_background_in_classification_targets: bool = True,
        background_label: int = 0,
        sample_background_patches: bool = False,
    ) -> None:
        """
        Dataloader for single whole-slide image (WSI). It calculates and saves the left, upper corners of the significant patches,
        it rejects empty ones.

        Parameters
        ----------
        img_path: Path
            Absolute path to the WSI, assumed that it exists, and WSI can be loaded properly.
        patch_size : Tuple[int, int]
            Size of the extracted patches.
        bg_mask_path: Path | None = None
            Absolute path to the background mask, suitable for analysed WSI.
            Assumed:
                If path is passed, it exists, and WSI can be loaded properly.
                Data is stored in RGBA format with dimension levels (possible to use read_region method).
        stride : Tuple[int, int] = None
            Stride between patches, calculated from upper, left corner. Stride smaller than patch size means overlap, greater means losing information.
            If None, stride is equal to patch_size.
        transform_config: TransformConfig | None = None,
            Configuration for the histopathology transformations to apply to the patches.
        pixel_size_tolerance_percent_coeff : Tuple[float, float] = (0.05, 0.05)) -> None:
            Threshold of percent pixel size coefficient = ratio of pixel sizes: target and at the suitable downsampling level for the patches extraction.
            If ratio is smaller than pixel_size_tolerance_percent_coeff, interpolation is not necessary.
        downstream_task : DownstreamTask = DownstreamTask.NONE
            Selected mode for downstream task.
        ground_truth_mask_path: Path | None = None
            Absolute path to ground truth mask, assumed that it exists, and GT mask can be loaded properly.
            It can be also treated as dense label.
            Obligatory for SEGMENTATION and STRONG CLASSIFICATION downstream mode, as well as if flag should_balance_batch is True.
        image_label: List[int] | None = None
            Image label means weak label of the image.
            Obligatory for WEAK CLASSIFICATION downstream mode.
        image_gt_all_labels: List[int] | None = None
            All labels (gt classes) appearing in the dataset.
            Obligatory for STRONG CLASSIFICATION downstream mode
        padding_value : int = 255
            Value for background if padding for patch is needed.

        Returns
        -------
        None

        Exceptions print
        ------
        If slide or mask image can not be opened properly.
        If target pixel size is smaller than original pixel size (at level 0), target pixel size is rounded to original pixel size.
        """
        super().__init__()
        self.image_path = img_path
        self.patch_size = patch_size
        self.bg_mask_path = bg_mask_path
        self.transform = HistopathologyTransform(
            config=transform_config, gt_labels=image_gt_all_labels
        )
        self.pixel_size_tolerance_percent_coeff = pixel_size_tolerance_percent_coeff
        self.mask_padding_value = 0
        self.priority_class = priority_class

        self.downstream_task = downstream_task
        self.ground_truth_mask_path = ground_truth_mask_path
        self.image_label = image_label
        self.exclude_background_in_classification_targets = (
            exclude_background_in_classification_targets
        )
        self.background_label = background_label
        self.sample_background_patches = sample_background_patches
        self.image_gt_all_labels = image_gt_all_labels
        self.image_padding_value = padding_value
        image_slide = openslide.open_slide(self.image_path)
        self._level_dimensions = image_slide.level_dimensions
        self._level_downsamples = image_slide.level_downsamples
        self._image_dimension_lvl0 = image_slide.dimensions
        self._pixel_level_0 = (
            float(image_slide.properties["openslide.mpp-x"]),
            float(image_slide.properties["openslide.mpp-y"]),
        )

        image_slide.close()

        if self.downstream_task in (
            DownstreamTask.SEGMENTATION,
            DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
            DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
        ):
            if self.ground_truth_mask_path is None or self.image_gt_all_labels is None:
                raise (
                    f"Image : {self.image_path}. Specified downstream task: SEGMENTATION / CLASSIFICATION, but ground_truth_mask_path or/and image_gt_all_labels is not given."
                )

            if self.transform.config.apply_mask_mapping:
                if not self.transform.config.mask_mapping:
                    raise (
                        f"Image : {self.image_path}. self.transform.config.mask_mapping dict {self.transform.config.mask_mapping} is not given."
                    )

                mapping_keys = set(self.transform.config.mask_mapping.keys())
                gt_labels_set = set(self.image_gt_all_labels)
                if mapping_keys != gt_labels_set:
                    raise (
                        f"Image : {self.image_path}. self.transform.config.mask_mapping keys {mapping_keys} do not match image_gt_all_labels {gt_labels_set}."
                    )

                mapped_values = set(self.transform.config.mask_mapping.values())
                if not all(isinstance(v, int) and v >= 0 for v in mapped_values):
                    raise (
                        f"Image : {self.image_path}. self.transform.config.mask_mapping values must be non-negative integers. Got: {mapped_values}"
                    )
        elif (
            self.downstream_task == DownstreamTask.WEAK_CLASSIFICATION
            and self.image_label is None
        ):
            raise (
                f"Image : {self.image_path}. Specified downstream task: WEAK CLASSIFICATION, but weak label is not given."
            )

        if (
            self.transform.config.apply_mask_mapping
            and self.transform.config.mask_mapping
        ):
            mapped = set(
                [
                    self.transform.config.mask_mapping[x]
                    for x in self.image_gt_all_labels
                ]
            )
            self.image_gt_all_labels = list(dict.fromkeys(mapped))

        self._sensible_mask_pixels_xy = self._find_valid_patch_coords()

    @staticmethod
    def _are_tuple_elements_greater_or_equal(
        input_tuple: Tuple[any], threshold: Tuple[any]
    ) -> bool:
        """
        Check if all elements of input_tuple are greater than or equal to corresponding threshold elements.

        Parameters
        ----------
        input_tuple : Tuple
            Input tuple to compare
        threshold : Tuple
            Threshold tuple for comparison

        Returns
        -------
        bool
            True if all elements of input_tuple >= corresponding threshold elements
        """
        return input_tuple[0] >= threshold[0] and input_tuple[1] >= threshold[1]

    @abstractmethod
    def _find_valid_patch_coords(self):
        """
        Abstract method to find coordinates in the image/slide that are valid for creating patches.

        The exact criteria for validity and the format of the returned data are determined
        by the subclass implementation. The coordinates can represent patch centers, corners,
        or any reference point deemed appropriate.

        Args:
            self: Instance of the class.

        Returns:
            Flexible: A collection, list, generator, or other structure containing
            coordinates of valid patches. Each coordinate should typically be a tuple (x, y).
        """
        pass

    def _process_mask_tiles(
        self,
        bg_mask_slide: openslide.OpenSlide,
        gt_mask_slide: openslide.OpenSlide,
        region_coords,
        region_size,
        level_dimensions,
    ):
        tile_array = np.ones((region_size[1], region_size[0]), dtype=bool)

        if bg_mask_slide is not None:
            bg_level = list(bg_mask_slide.level_dimensions).index(level_dimensions)
            bg_tile = AbstractSingleWSIDataset._read_and_split_bg_mask_patch(
                bg_mask_slide, region_coords, bg_level, region_size
            )
            bg_tile_array = (np.array(bg_tile) > 0).astype(np.uint8)
            tile_array = tile_array * bg_tile_array

        if gt_mask_slide is not None:
            gt_level = list(gt_mask_slide.level_dimensions).index(level_dimensions)
            gt_tile = AbstractSingleWSIDataset._read_and_split_gt_mask_patch(
                gt_mask_slide, region_coords, gt_level, region_size
            )
            gt_tile_transformed = self.transform.apply_mask_mapping_only(mask=gt_tile)
            gt_tile_transformed = gt_tile_transformed.squeeze(0)
            gt_tile_array = np.array(gt_tile_transformed)
            tile_array = tile_array * gt_tile_array
        return tile_array

    def _get_best_dimension_level_for_patches_extraction(
        self, pixel_size: Tuple[float, float]
    ) -> int | None:
        """
        Selects the most appropriate image level for extracting patches
        for a given target pixel size.

        Algorithm:
        - Computes the expected dimensions (width, height) at each level that
            would correspond to the requested `pixel_size` (variable
            `dimension_for_given_pixel_size`). The calculation uses
            `self._pixel_level_0` and `self._image_dimension_lvl0`.
        - Filters available levels in `self._level_dimensions` to those
            whose dimensions are greater than or equal to the required
            dimensions, using `are_tuple_elements_greater_or_equal`.
        - From the matched levels selects the last one (the highest
            resolution that still satisfies the requirement).
        - If no suitable level is found, the function returns `0` (base
            level).

        Parameters
        ----------
        pixel_size : Tuple[float, float]
                Target pixel size (sx, sy).

        Returns
        -------
        int | None
                Index of the selected level within `self._level_dimensions`.
                The implementation returns `0` when no suitable level is found;
                the signature allows `None` but the current code always returns
                an `int`.
        """

        dimension_for_given_pixel_size = tuple(
            int((ps_lvl0 * float(dim_lvl0) / ps_target))
            for ps_lvl0, ps_target, dim_lvl0 in zip(
                self._pixel_level_0, pixel_size, self._image_dimension_lvl0
            )
        )
        interpolation_lvl = filter(
            lambda lvl_dim: self._are_tuple_elements_greater_or_equal(
                lvl_dim, dimension_for_given_pixel_size
            ),
            self._level_dimensions,
        )
        filtered_list = list(interpolation_lvl)
        suitable_dim_lvl = filtered_list[-1] if filtered_list else None
        if suitable_dim_lvl == None:
            return 0
        return list(self._level_dimensions).index(suitable_dim_lvl)

    def _calculate_patchsize_if_interpolation_is_needed(
        self, downsampling_factor, pixel_size
    ) -> None | Tuple[int, int]:

        pixel_size_suitable_level = tuple(
            downsampling_factor * lvl0 for lvl0 in self._pixel_level_0
        )

        pixel_size_coeff = tuple(
            (target / suitable) - 1
            for suitable, target in zip(pixel_size_suitable_level, pixel_size)
        )

        if self._are_tuple_elements_greater_or_equal(
            self.pixel_size_tolerance_percent_coeff, pixel_size_coeff
        ):
            pixel_size = pixel_size_suitable_level
            return pixel_size, None

        calculated_patch_size = tuple(
            int(np.ceil(size * (factor + 1)))
            for size, factor in zip(self.patch_size, pixel_size_coeff)
        )
        return pixel_size, calculated_patch_size

    def _pad_tile(
        self, tile: Image.Image, target_size: tuple[int, int], is_mask: bool = False
    ) -> Image.Image:
        """
        Pad the tile to target_size. Returns a PIL.Image.

        Parameters
        ----------
        tile : PIL.Image
            Input image or mask.
        target_size : tuple[int,int]
            Desired (width, height) after padding.
        is_mask : bool
            If True, use mask_padding_value for mask; else use image_padding_value.
        """
        if is_mask:
            pad_val = self.mask_padding_value
        else:
            pad_val = self.image_padding_value

        if tile.size == target_size:
            return tile

        if tile.mode == "L" or is_mask:
            full_tile = Image.new("L", target_size, pad_val)
        else:
            full_tile = Image.new("RGB", target_size, (pad_val, pad_val, pad_val))

        full_tile.paste(tile, (0, 0))
        return full_tile

    def _resize_tile(
        self, tile: Image.Image, target_size: tuple[int, int]
    ) -> Image.Image:
        """
        Resize the tile to target_size. Chooses resampling depending on mode.
        """
        if tile.size == target_size:
            return tile

        resample_method = Image.NEAREST if tile.mode == "L" else Image.BILINEAR
        return tile.resize(target_size, resample=resample_method)

    def _create_one_hot_vector(self, mask_tensor):

        unique_labels, labels_counts = torch.unique(mask_tensor, return_counts=True)
        num_classes = len(self.image_gt_all_labels)
        one_hot_vector = torch.zeros(
            num_classes, dtype=torch.float32, device=mask_tensor.device
        )
        one_hot_vector[unique_labels] = 1.0
        counts_vector = torch.zeros(
            num_classes, dtype=torch.int32, device=mask_tensor.device
        )
        counts_vector[unique_labels] = labels_counts.to(torch.int32)
        return one_hot_vector, counts_vector

    @staticmethod
    def _one_hot_to_label(one_hot, priority_class=None):
        """
        Convert a one-hot vector (1D) into a single label tensor (float32).

        Description:
            - If a priority class is specified and present in the vector, this class is returned.
            - If the priority class is not present, the function returns the first class that occurs in the one-hot vector.
            - If no class is present (all zeros), it defaults to None.

        Parameters:
            one_hot (tensor): 1D tensor of shape (C,), representing one-hot encoding
            priority_class (int, optional): index of the class that should have priority

        Returns:
            torch.Tensor: single label as float32
        """
        one_hot = one_hot.flatten()

        if priority_class is not None and one_hot[priority_class] == 1:
            single_label = priority_class
        else:
            idx = torch.nonzero(one_hot).flatten()
            if len(idx) > 0:
                single_label = int(idx[0])
            else:
                single_label = None

        return (
            torch.tensor(single_label, dtype=torch.float32)
            if single_label is not None
            else None
        )

    @staticmethod
    def _load_image_patch(
        path: Path,
        region_coords: Tuple[int, int],
        lvl: int,
        region_size: Tuple[int, int],
    ) -> Image.Image:
        image_slide = openslide.open_slide(path)
        tile = image_slide.read_region(region_coords, lvl, region_size).convert("RGB")
        image_slide.close()
        return tile

    @staticmethod
    def _read_and_split_bg_mask_patch(
        bg_mask: openslide.OpenSlide,
        region_coords: Tuple[int, int],
        lvl: int,
        region_size: Tuple[int, int],
    ) -> Image.Image:
        bg_mask = bg_mask.read_region(region_coords, lvl, region_size).convert("L")
        return bg_mask

    @staticmethod
    def _read_and_split_gt_mask_patch(
        ground_truth_mask: openslide.OpenSlide,
        region_coords: Tuple[int, int],
        lvl: int,
        region_size: Tuple[int, int],
    ) -> Image.Image:
        gt_mask = ground_truth_mask.read_region(region_coords, lvl, region_size)
        gt_mask, _, _, _ = gt_mask.split()
        return gt_mask

    @staticmethod
    def _load_gt_mask_patch(
        path: Path,
        region_coords: Tuple[int, int],
        lvl: int,
        region_size: Tuple[int, int],
    ) -> Image.Image:
        ground_truth_mask = openslide.open_slide(path)
        gt_mask = AbstractSingleWSIDataset._read_and_split_gt_mask_patch(
            ground_truth_mask, region_coords, lvl, region_size
        )
        ground_truth_mask.close()
        return gt_mask

    def _load_patch_at(
        self,
        region_coords: Tuple[int, int],
        region_size: Tuple[int, int],
        pixel_size: Tuple[float, float],
        lvl: int,
        patch_size_for_extraction: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[float, float]]:

        pixel_size_tensor = torch.tensor(pixel_size, dtype=torch.float32)
        tile = AbstractSingleWSIDataset._load_image_patch(
            self.image_path, region_coords, lvl, region_size
        )
        padded_tile = self._pad_tile(tile, patch_size_for_extraction)
        resized_tile = self._resize_tile(padded_tile, self.patch_size)

        if self.downstream_task == DownstreamTask.NONE:
            transformed_img_tile, _ = self.transform(resized_tile, None)
            return transformed_img_tile, pixel_size_tensor

        elif self.downstream_task == DownstreamTask.SEGMENTATION:
            gt_mask = AbstractSingleWSIDataset._load_gt_mask_patch(
                self.ground_truth_mask_path, region_coords, lvl, region_size
            )
            padded_mask = self._pad_tile(gt_mask, patch_size_for_extraction)
            resized_mask = self._resize_tile(padded_mask, self.patch_size)
            transformed_img_tile, transformed_mask_tile = self.transform(
                resized_tile, resized_mask
            )

            return transformed_img_tile, pixel_size_tensor, transformed_mask_tile

        elif self.downstream_task == DownstreamTask.WEAK_CLASSIFICATION:
            transformed_img_tile, _ = self.transform(resized_tile, None)
            return transformed_img_tile, pixel_size_tensor, self.image_label

        elif self.downstream_task in (
            DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
            DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
        ):
            gt_mask = AbstractSingleWSIDataset._load_gt_mask_patch(
                self.ground_truth_mask_path, region_coords, lvl, region_size
            )
            padded_mask = self._pad_tile(gt_mask, patch_size_for_extraction)
            resized_mask = self._resize_tile(padded_mask, self.patch_size)
            transformed_img_tile, transformed_mask_tile = self.transform(
                resized_tile, resized_mask
            )
            dense_labels, labels_counts = self._create_one_hot_vector(
                transformed_mask_tile
            )
            dense_labels_original = dense_labels.clone()
            labels_counts_original = labels_counts.clone()

            if self.exclude_background_in_classification_targets:
                mask = torch.ones(dense_labels.size(0), dtype=torch.bool)
                mask[self.background_label] = False
                dense_labels = dense_labels[mask]
                labels_counts = labels_counts[mask]

            if self.downstream_task == DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION:
                return (
                    transformed_img_tile,
                    pixel_size_tensor,
                    dense_labels,
                    (dense_labels_original, labels_counts_original),
                )

            priority_class = self.priority_class
            if (
                self.exclude_background_in_classification_targets
                and priority_class is not None
            ):
                if priority_class == self.background_label:
                    priority_class = None
                elif priority_class > self.background_label:
                    priority_class = priority_class - 1

            single_label = AbstractSingleWSIDataset._one_hot_to_label(
                dense_labels, priority_class
            )

            return (
                transformed_img_tile,
                pixel_size_tensor,
                single_label,
                (dense_labels_original, labels_counts_original),
            )

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx_or_tuple):
        pass
