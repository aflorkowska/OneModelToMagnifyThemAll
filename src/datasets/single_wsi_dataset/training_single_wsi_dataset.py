import os

######################### WINDOWS
import platform

if platform.system() == "Windows":
    from paths.paths import OPENSLIDE_BIN_DIR

    os.add_dll_directory(OPENSLIDE_BIN_DIR)
import openslide

################################################################
import numpy as np
from pathlib import Path
from typing import Tuple, List
from scipy.ndimage import binary_erosion

from datasets.single_wsi_dataset.abstract.abstract_single_wsi_dataset import (
    AbstractSingleWSIDataset,
    DownstreamTask,
)

from datasets.single_wsi_dataset.histopathology_transform import TransformConfig


class TrainingSingleWSIDataset(AbstractSingleWSIDataset):
    def __init__(
        self,
        img_path: Path,
        patch_size: Tuple[int, int],
        should_balance_batch: bool,
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
        Dataloader for single whole-slide image (WSI). Get_item method returns patches and mask in format C,H,W.

        Parameters
        ----------
        img_path: Path
            Absolute path to the WSI, assumed that it exists, and WSI can be loaded properly.
        patch_size : Tuple[int, int]
            Size of the extracted patches.
        should_balance_batch : bool
            Flag. If true, balanced patch samples is active. If false, it is about random sampling all significant pixels.
        bg_mask_path: Path | None = None
            Absolute path to the background mask, suitable for analysed WSI.
            Obligatory if flag should_balance_batch is True.
            Assumed:
                If path is passed, it exists, and WSI can be loaded properly.
                Data is stored in RGBA format with dimension levels (possible to use read_region method).
        stride : Tuple[int, int] = None
            Stride between patches, calculated from upper, left corner. Stride smaller than patch size means overlap, greater means losing information.
            If None, stride is equal to patch_size.
        transform_config: TransformConfig | None = None
            Configuration for the histopathology transformations to apply to the patches. If None, default configuration is used.
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
        self.should_balance_batch = should_balance_batch

        super().__init__(
            img_path=img_path,
            patch_size=patch_size,
            bg_mask_path=bg_mask_path,
            transform_config=transform_config,
            pixel_size_tolerance_percent_coeff=pixel_size_tolerance_percent_coeff,
            downstream_task=downstream_task,
            ground_truth_mask_path=ground_truth_mask_path,
            image_label=image_label,
            image_gt_all_labels=image_gt_all_labels,
            priority_class=priority_class,
            padding_value=padding_value,
            exclude_background_in_classification_targets=exclude_background_in_classification_targets,
            background_label=background_label,
            sample_background_patches=sample_background_patches,
        )

        if self.should_balance_batch and (
            self.bg_mask_path is None or self.ground_truth_mask_path is None
        ):
            raise ValueError(
                f"Image : {self.image_path}. Specified balanced batch generation, but ground_truth_mask_path or/and bg_mask_path is/are not given."
            )

        if not self.should_balance_batch and (
            self.bg_mask_path is None or self.ground_truth_mask_path is None
        ):
            raise ValueError(
                f"Image : {self.image_path}. Specified unbalanced batch generation, but ground_truth_mask_path or/and bg_mask_path is/are not given. Random sampling from all significant pixels will be used."
            )

    def _find_valid_patch_coords(self):

        image_slide = openslide.open_slide(self.image_path)
        bg_mask_slide = (
            openslide.open_slide(self.bg_mask_path) if self.bg_mask_path else None
        )
        gt_mask_slide = (
            openslide.open_slide(self.ground_truth_mask_path)
            if self.ground_truth_mask_path
            else None
        )

        if self.should_balance_batch and bg_mask_slide and gt_mask_slide:
            coordinates = self._find_balanced_class_pixels(bg_mask_slide, gt_mask_slide)
        else:
            coordinates = self._find_mask_pixels(
                bg_mask_slide if bg_mask_slide is not None else image_slide,
                gt_mask_slide if gt_mask_slide is not None else image_slide,
            )

        image_slide.close()
        if bg_mask_slide:
            bg_mask_slide.close()
        if gt_mask_slide:
            gt_mask_slide.close()

        return coordinates

    def _find_balanced_class_pixels(
        self,
        bg_mask_slide: openslide.OpenSlide,
        gt_mask_slide: openslide.OpenSlide,
    ):
        tile_array, downsample_lvl = self._get_mask_tile_array(
            bg_mask_slide, gt_mask_slide
        )

        labels_indexes = {}

        for label in self.image_gt_all_labels:
            if not self.sample_background_patches and label == self.background_label:
                continue

            mask = tile_array == label
            coords = self._extract_and_shuffle_coords(mask, downsample_lvl)

            if coords.shape[1] > 0:
                labels_indexes[label] = coords

        return labels_indexes

    def _find_mask_pixels(
        self,
        bg_mask_slide: openslide.OpenSlide,
        gt_mask_slide: openslide.OpenSlide,
    ):
        tile_array, downsample_lvl = self._get_mask_tile_array(
            bg_mask_slide, gt_mask_slide
        )

        if not self.sample_background_patches:
            mask = tile_array != self.background_label
        else:
            mask = np.ones_like(tile_array, dtype=bool)
        return self._extract_and_shuffle_coords(mask, downsample_lvl)

    def _extract_and_shuffle_coords(
        self,
        mask: np.ndarray,
        downsample_lvl: float,
    ) -> np.ndarray:
        """Extract pixel coordinates from a boolean mask and shuffle them."""

        eroded = binary_erosion(mask, structure=np.ones((5, 5), dtype=bool))
        mask_eroded = eroded.astype(int)
        if np.array_equal(np.unique(mask_eroded), np.array([0])):
            final_mask = mask
        else:
            final_mask = mask_eroded

        rows, cols = np.where(final_mask)

        if rows.size == 0:
            return np.empty((2, 0), dtype=np.uint32)

        coords = np.stack(
            [
                (cols * downsample_lvl).astype(np.uint32),
                (rows * downsample_lvl).astype(np.uint32),
            ]
        )

        indices = np.random.permutation(coords.shape[1])
        return coords[:, indices]

    def _get_mask_tile_array(
        self,
        bg_mask_slide: openslide.OpenSlide,
        gt_mask_slide: openslide.OpenSlide,
    ):
        region_coords = (0, 0)
        lvl_idx = len(gt_mask_slide.level_dimensions) - 1
        region_size = gt_mask_slide.level_dimensions[lvl_idx]

        tile_array = self._process_mask_tiles(
            bg_mask_slide=bg_mask_slide,
            gt_mask_slide=gt_mask_slide,
            region_coords=region_coords,
            region_size=region_size,
            level_dimensions=region_size,
        )

        downsample_lvl = gt_mask_slide.level_downsamples[lvl_idx]
        return tile_array, downsample_lvl

    def __len__(self) -> int:
        if self.should_balance_batch:
            return sum(
                coords.shape[1] for coords in self._sensible_mask_pixels_xy.values()
            )
        else:
            return len(self._sensible_mask_pixels_xy[1])

    def __getitem__(self, idx_or_tuple):
        if self.should_balance_batch:
            pixel_size, label_from_sampler = idx_or_tuple

            label_from_sampler_prep = label_from_sampler
            if label_from_sampler not in self._sensible_mask_pixels_xy.keys():
                label_from_sampler_prep = next(iter(self._sensible_mask_pixels_xy))
                print(
                    f"Warning - selected label {label_from_sampler} does not exist in given image "
                    f"with labels {list(self._sensible_mask_pixels_xy.keys())}. "
                    f"Set to the first label = {label_from_sampler_prep}."
                )

            list_of_valid_coords = self._sensible_mask_pixels_xy[
                label_from_sampler_prep
            ]
            idx = np.random.randint(0, list_of_valid_coords.shape[1])
        else:
            pixel_size = idx_or_tuple
            list_of_valid_coords = self._sensible_mask_pixels_xy
            idx = np.random.randint(0, len(self._sensible_mask_pixels_xy[1]))

        if not self._are_tuple_elements_greater_or_equal(
            pixel_size, self._pixel_level_0
        ):
            print(
                f"""WSI: {self.image_path} - Target pixel size, value: {pixel_size}, 
                    should be greater or equal to the original pixel size, value: {self._pixel_level_0}. 
                    Target pixel size was rounded to original value at level 0."""
            )
            pixel_size = self._pixel_level_0

        slide_level = self._get_best_dimension_level_for_patches_extraction(pixel_size)
        downsampling_factor = self._level_downsamples[slide_level]
        pixel_size, calculated_patch_size = (
            self._calculate_patchsize_if_interpolation_is_needed(
                downsampling_factor, pixel_size
            )
        )
        should_patch_be_resized = True if calculated_patch_size else False

        patch_size_for_extraction = (
            calculated_patch_size if should_patch_be_resized else self.patch_size
        )
        region_coords = (
            int(list_of_valid_coords[0][idx]),
            int(list_of_valid_coords[1][idx]),
        )
        x_left, y_left = max(
            0,
            int(
                region_coords[0]
                - (downsampling_factor * patch_size_for_extraction[0] // 2)
            ),
        ), max(
            0,
            int(
                region_coords[1]
                - (downsampling_factor * patch_size_for_extraction[1] // 2)
            ),
        )

        region_size = tuple(
            (
                calculated
                if calculated * downsampling_factor < int(max_size - coord)
                else int((max_size - coord) / downsampling_factor)
            )
            for calculated, max_size, coord in zip(
                patch_size_for_extraction, self._image_dimension_lvl0, (x_left, y_left)
            )
        )

        return self._load_patch_at(
            (x_left, y_left),
            region_size,
            pixel_size,
            slide_level,
            patch_size_for_extraction,
        )
