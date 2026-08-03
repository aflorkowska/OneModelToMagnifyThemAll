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


from datasets.single_wsi_dataset.abstract.abstract_single_wsi_dataset import (
    AbstractSingleWSIDataset,
    DownstreamTask,
)

from datasets.single_wsi_dataset.histopathology_transform import TransformConfig


class InferenceSingleWSIDataset(AbstractSingleWSIDataset):
    def __init__(
        self,
        img_path: Path,
        pixel_size: Tuple[float, float],
        patch_size: Tuple[int, int],
        bg_mask_path: Path | None = None,
        stride: Tuple[int, int] | None = None,
        transform_config: TransformConfig | None = None,
        downstream_task: DownstreamTask = DownstreamTask.NONE,
        ground_truth_mask_path: Path | None = None,
        image_label: List[int] | None = None,
        image_gt_all_labels: List[int] | None = None,
        exclude_background_in_classification_targets: bool = True,
        background_label: int = 0,
        sample_background_patches: bool = False,
        priority_class: int | None = None,
        pixel_size_tolerance_percent_coeff: Tuple[float, float] = (0.05, 0.05),
        padding_value: int = 255,
    ) -> None:
        """
        Dataloader for single whole-slide image (WSI). It calculates and saves the left, upper corners of the significant patches,
        it rejects empty ones.

        Parameters
        ----------
        img_path: Path
            Absolute path to the WSI, assumed that it exists, and WSI can be loaded properly.
        pixel_size : Tuple[float, float]
            Physical pixel size of the extracted patches [um / pixel].
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
        transform_config: TransformConfig | None = None
            Configuration for the histopathology transformations to apply to the patches. If None, default configuration is used.
        downstream_task : DownstreamTask = DownstreamTask.NONE
            Selected mode for downstream task.
        ground_truth_mask_path: Path | None = None
            Absolute path to ground truth mask, assumed that it exists, and GT mask can be loaded properly.
            It can be also treated as dense label.
            Obligatory for SEGMENTATION and STRONG CLASSIFICATION downstream mode.
        image_label: List[int] | None = None
            Image label means weak label of the image.
            Obligatory for WEAK CLASSIFICATION downstream mode.
        image_gt_all_labels: List[int] | None = None
            All labels (gt classes) appearing in the dataset.
            Obligatory for STRONG CLASSIFICATION downstream mode
        pixel_size_tolerance_percent_coeff : Tuple[float, float] = (0.05, 0.05)) -> None:
            Threshold of percent pixel size coefficient = ratio of pixel sizes: target and at the suitable downsampling level for the patches extraction.
            If ratio is smaller than pixel_size_tolerance_percent_coeff, interpolation is not necessary.
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
        self._WIDTH_INDEX = 0
        self._HEIGHT_INDEX = 1
        self.target_pixel_size = pixel_size
        self.stride = stride

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

    def get_grid_shape(self) -> Tuple[int, int]:
        return self._grid_shape

    def has_patch_at(self, row: int, col: int) -> bool:
        return (row, col) in self._grid_to_dataset_idx

    def get_patch_idx_at(self, row: int, col: int) -> int | None:
        return self._grid_to_dataset_idx.get((row, col))

    def get_patch_position(self, idx):
        return self._dataset_idx_to_grid[idx]

    def _find_valid_patch_coords(self):

        if not self._are_tuple_elements_greater_or_equal(
            self.target_pixel_size, self._pixel_level_0
        ):
            print(
                f"""WSI: {self.image_path} - Target pixel size, value: {self.target_pixel_size}, 
                    should be greater or equal to the original pixel size, value: {self._pixel_level_0}. 
                    Target pixel size was rounded to original value at level 0."""
            )
            self.target_pixel_size = self._pixel_level_0

        self._slide_level = self._get_best_dimension_level_for_patches_extraction(
            self.target_pixel_size
        )
        self._downsampling_factor = self._level_downsamples[self._slide_level]
        _, calculated_patch_size = self._calculate_patchsize_if_interpolation_is_needed(
            self._downsampling_factor, self.target_pixel_size
        )
        should_patch_be_resized = True if calculated_patch_size else False
        self._patch_size_for_extraction = (
            calculated_patch_size if should_patch_be_resized else self.patch_size
        )
        self._increment_step = self._calculate_increment_step()

        self._grid_shape = (
            int(np.ceil(self._image_dimension_lvl0[1] / self._increment_step[1])),
            int(np.ceil(self._image_dimension_lvl0[0] / self._increment_step[0])),
        )

        mask_slide = (
            openslide.open_slide(self.bg_mask_path) if self.bg_mask_path else None
        )
        gt_mask_slide = (
            openslide.open_slide(self.ground_truth_mask_path)
            if self.ground_truth_mask_path
            else None
        )
        coordinates = self._calculate_left_upper_corners(
            mask_slide,
            gt_mask_slide,
        )
        if mask_slide:
            mask_slide.close()
        if gt_mask_slide:
            gt_mask_slide.close()

        return coordinates

    def get_patch_size_for_extraction(self):
        return self._patch_size_for_extraction

    def _calculate_increment_step(self) -> Tuple[int, int]:

        stride = self.patch_size if self.stride == None else self.stride
        increment_step = tuple(
            int((s / final_size) * self._downsampling_factor * extracted)
            for s, final_size, extracted in zip(
                stride, self.patch_size, self._patch_size_for_extraction
            )
        )
        return increment_step

    def _calculate_left_upper_corners(
        self,
        bg_mask_slide: openslide.OpenSlide | None,
        gt_mask_slide: openslide.OpenSlide | None,
    ) -> List[Tuple[int, int]]:

        step_width = self._increment_step[self._WIDTH_INDEX]
        step_height = self._increment_step[self._HEIGHT_INDEX]
        image_lvl0_width = self._image_dimension_lvl0[self._WIDTH_INDEX]
        image_lvl0_height = self._image_dimension_lvl0[self._HEIGHT_INDEX]

        upper_left_corners = []
        for y in range(0, image_lvl0_height, step_height):
            for x in range(0, image_lvl0_width, step_width):
                region_coords = np.array([int(x), int(y)])
                if bg_mask_slide is None and gt_mask_slide is None:
                    upper_left_corners.append(region_coords)
                else:
                    if self.sample_background_patches:
                        upper_left_corners.append(region_coords)
                        continue
                    region_size = tuple(
                        int(min(calculated, max_size - coord))
                        for calculated, max_size, coord in zip(
                            self._patch_size_for_extraction,
                            self._image_dimension_lvl0,
                            region_coords,
                        )
                    )
                    lvl_dim = self._level_dimensions[self._slide_level]
                    tile_array = self._process_mask_tiles(
                        bg_mask_slide=bg_mask_slide,
                        gt_mask_slide=gt_mask_slide,
                        region_coords=region_coords,
                        region_size=region_size,
                        level_dimensions=lvl_dim,
                    )
                    if np.any(tile_array):
                        upper_left_corners.append(region_coords)

        upper_left_corners_np = np.array(upper_left_corners)

        self._grid_to_dataset_idx = {}
        for idx, (x, y) in enumerate(upper_left_corners_np):
            row = y // self._increment_step[1]
            col = x // self._increment_step[0]
            self._grid_to_dataset_idx[(row, col)] = idx

        self._dataset_idx_to_grid = {
            idx: (row, col) for (row, col), idx in self._grid_to_dataset_idx.items()
        }

        return upper_left_corners_np

    def __len__(self) -> int:
        return len(self._sensible_mask_pixels_xy)

    def __getitem__(self, idx: int):
        idx = np.clip(idx, 0, len(self._sensible_mask_pixels_xy) - 1)
        region_coords = self._sensible_mask_pixels_xy[idx]
        region_size = tuple(
            (
                calculated
                if calculated * self._downsampling_factor < int(max_size - coord)
                else int((max_size - coord) / self._downsampling_factor)
            )
            for calculated, max_size, coord in zip(
                self._patch_size_for_extraction,
                self._image_dimension_lvl0,
                region_coords,
            )
        )

        return self._load_patch_at(
            region_coords,
            region_size,
            self.target_pixel_size,
            self._slide_level,
            self._patch_size_for_extraction,
        )
