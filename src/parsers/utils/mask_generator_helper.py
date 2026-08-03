import os
import sys

######################### WINDOWS
import platform

if platform.system() == "Windows":
    from paths.paths import OPENSLIDE_BIN_DIR

    os.add_dll_directory(OPENSLIDE_BIN_DIR)
import openslide

################################################################
import copy
import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

from parsers.config import WSI_EXTs, ANNOTATION_EXTs, AnnotationDataType

from utils.wsi_loading_utils import WSILoadingUtils

from parsers.utils.file_system_utils import FileSystemUtils
from parsers.utils.annotation_parser import AnnotationParser
from parsers.utils.mask_processor import MaskProcessor


class MaskGenerationHelper:
    """
    Helper class for mask generation and annotation processing.

    Provides static methods for converting XML annotations to masks,
    calculating class distributions, and analyzing annotations.
    """

    @staticmethod
    def generate_mask_as_tiff(
        image_path: Path,
        annotation_path: Path,
        output_path: Path,
        group_to_value: Dict[str, int],
        xml_type: AnnotationDataType,
    ) -> Tuple[List[int], List[int]]:
        """
        Generate mask from xml annotations, and then save it as pyramidal tiff file.
        It produces levels with downsampling factor only equal to 2.

        Parameters
        ----------
        image_path : Path
            Image path - to copy metadata.

        annotation_path : Path
            Path to xml with annotations

        output_path : Path
            Path to output file, with filename ending with extension ".tiff"

        group_to_value : Dict[str, int]
            Dictionary mapping group names to unique integer identifiers.

        xml_type : AnnotationDataType
            Information about data saved in xml, wheather coordinates are saved in pixels at level 0 resolution, or in um.

        Returns
        -------
        Tuple[list[int], list[int]]

        """
        image = WSILoadingUtils.load_image(image_path)
        if image == None:
            return

        metadata = MaskProcessor.prepare_metadata(image)
        pixel_size_um = WSILoadingUtils.get_pixel_size_scalling_factor(image_path)

        if xml_type == AnnotationDataType.PIXELS_LVL_0:
            factor = (1.0, 1.0)
        elif xml_type == AnnotationDataType.UM_LVL_0:
            factor = pixel_size_um

        coords_polys_lvl0 = AnnotationParser.parse_annotations(annotation_path, factor)
        mask, labels, counts = MaskProcessor.create_array_from_coordinates(
            (image.width, image.height), coords_polys_lvl0, group_to_value
        )
        MaskProcessor.save_mask_as_tiff(metadata, mask, output_path)

        return labels, counts

    @staticmethod
    def calculate_class_distribution_from_gt_masks(
        dataset_config: pd.core.series.Series,
        images_classes_stats_output_path: Path,
        data_dir: Path,
    ):
        """
        Calculate class distribution based on ground truth masks.

        Parameters
        ----------
        dataset_config : pd.core.series.Series
            Dataset configuration series with information

        images_classes_stats_output_path : Path
            Path to save csv with calculated class counts - how many pixels of each class were found in each image from the dataset.
            IMPORTANT: If bg_path is specified, region of interest (tissue from background mask) is searched. If not, the whole image.

        data_dir : Path
            Path to main dir with images.
        Returns
        -------

        """

        gt_masks_dir = (
            Path(dataset_config.gt_masks_dir)
            if not pd.isna(dataset_config.gt_masks_dir)
            else None
        )

        if gt_masks_dir is None or not gt_masks_dir.exists():
            return

        files_summary_csv = (
            Path(dataset_config.files_summary)
            if not pd.isna(dataset_config.files_summary)
            else None
        )
        df_file_summary = (
            pd.read_csv(files_summary_csv) if files_summary_csv != None else None
        )

        all_labels_csv_radboud = (
            Path(dataset_config.all_classes_radboud)
            if not pd.isna(dataset_config.all_classes_radboud)
            else None
        )
        df_all_labels_radboud = (
            pd.read_csv(all_labels_csv_radboud)
            if all_labels_csv_radboud != None
            else None
        )
        classes_radboud = (
            df_all_labels_radboud["Value"].to_list()
            if isinstance(df_all_labels_radboud, pd.DataFrame)
            else np.nan
        )
        base_classes_image_radboud = {element: 0 for element in classes_radboud}

        all_labels_csv_karolinska = (
            Path(dataset_config.all_classes_karolinska)
            if not pd.isna(dataset_config.all_classes_karolinska)
            else None
        )
        df_all_labels_karolinska = (
            pd.read_csv(all_labels_csv_karolinska)
            if all_labels_csv_karolinska != None
            else None
        )
        classes_karolinska = (
            df_all_labels_karolinska["Value"].to_list()
            if isinstance(df_all_labels_karolinska, pd.DataFrame)
            else np.nan
        )
        base_classes_image_karolinska = {element: 0 for element in classes_karolinska}

        gt_masks_paths = FileSystemUtils.find_files_with_extension(
            gt_masks_dir, WSI_EXTs
        )

        bg_masks_dir = (
            Path(dataset_config.bg_masks_dir)
            if not pd.isna(dataset_config.bg_masks_dir)
            else None
        )
        if bg_masks_dir is not None and bg_masks_dir.exists():
            bg_masks_paths = FileSystemUtils.find_files_with_extension(
                bg_masks_dir, WSI_EXTs
            )
        else:
            bg_masks_paths = None

        if not gt_masks_paths:
            raise FileNotFoundError(f"Input list of paths is empty.")

        all_images_class_stats = []
        TEMP_MASK_VALUE = 100
        for path in tqdm.tqdm(gt_masks_paths, desc="Calculating class distribution..."):
            try:
                print(path)
                image_id = path.stem.removesuffix("_mask")
                data_provider = (
                    df_file_summary[df_file_summary["image_id"] == image_id][
                        "data_provider"
                    ].values[0]
                    if isinstance(df_file_summary, pd.DataFrame)
                    else None
                )
                if data_provider == "radboud":
                    image_labels = copy.deepcopy(base_classes_image_radboud)
                elif data_provider == "karolinska":
                    image_labels = copy.deepcopy(base_classes_image_karolinska)
                else:
                    print(f"Unknown data provider for image {image_id}. Skipping...")
                    continue

                mask_slide = openslide.open_slide(path)
                dim_last_lvl = mask_slide.level_dimensions[-1]
                last_lvl_num = len(mask_slide.level_dimensions) - 1
                tile = mask_slide.read_region((0, 0), last_lvl_num, dim_last_lvl)
                tile_array = np.array(tile)
                tile_array = tile_array[:, :, 0]

                if bg_masks_paths is not None and bg_masks_paths:
                    bg_path = FileSystemUtils.find_matching_file(path, bg_masks_paths)
                    if bg_path is not None:
                        is_bg_mask_took_into_account = True
                        bg_slide = openslide.open_slide(bg_path)
                        bg_level = list(bg_slide.level_dimensions).index(dim_last_lvl)
                        bg_tile = bg_slide.read_region(
                            (0, 0), bg_level, dim_last_lvl
                        ).convert("L")
                        bg_tile_array = np.array(bg_tile)
                        final_bg_tile_array = bg_tile_array > 0
                        tile_array[tile_array == 0] = TEMP_MASK_VALUE
                        tile_array = tile_array * final_bg_tile_array
                        weak_labels, weak_labels_counts = np.unique(
                            tile_array, return_counts=True
                        )
                        weak_labels = np.delete(
                            weak_labels, 0
                        )  # remove background counts outside the mask area
                        weak_labels_counts = np.delete(
                            weak_labels_counts, 0
                        )  # remove background counts outside the mask area
                        weak_labels[weak_labels == TEMP_MASK_VALUE] = (
                            0  # rename background counts inside the mask area
                        )
                    else:
                        is_bg_mask_took_into_account = False
                        weak_labels, weak_labels_counts = np.unique(
                            tile_array, return_counts=True
                        )
                else:
                    is_bg_mask_took_into_account = False
                    weak_labels, weak_labels_counts = np.unique(
                        tile_array, return_counts=True
                    )

                for i in range(len(weak_labels)):
                    image_labels[weak_labels[i]] = weak_labels_counts[i]

                all_images_class_stats.append(
                    {
                        "mask_tissue_path": os.path.relpath(bg_path, data_dir),
                        "dense_label_path": os.path.relpath(path, data_dir),
                        "image_class_distribution": image_labels,
                        "image_classes": list(weak_labels),
                        "bg_mask_took_into_account_when_image_class_distribution_calculation": is_bg_mask_took_into_account,
                    }
                )

            except Exception as e:
                print(f"An unexpected error occurred: {e} while proccesing {path} file")
                continue

        df_all_images_class_weights = pd.DataFrame(all_images_class_stats)
        df_all_images_class_weights.to_csv(
            images_classes_stats_output_path, index=False
        )

    @staticmethod
    def generate_masks_from_annotations(
        annotations_dir: Path,
        images_dir: Path,
        output_mask_dir: Path,
        data_dir: Path,
        xml_type: AnnotationDataType,
    ):
        """
        Generate mask from xml annotations, and then save it as pyramidal tiff file.
        It produces levels with downsampling factor only equal to 2.

        Parameters
        ----------
        annotations_dir : Path
            Dir with annotations.

        images_dir : Path
            Dir with images.

        output_mask_dir : Path
            Output dir for masks.

        data_dir : Path
            Path to main data directory.

        xml_type : AnnotationDataType
            Information about data saved in xml, wheather coordinates are saved in pixels at level 0 resolution, or in um.

        Returns
        -------

        """
        classes = AnnotationParser.find_unique_part_of_groups(annotations_dir)
        path_to_save_labels_explanation = output_mask_dir / Path(
            "labels_explanation.csv"
        )
        MaskProcessor.save_labels_explanation_to_csv(
            classes, path_to_save_labels_explanation
        )

        img_paths = FileSystemUtils.find_files_with_extension(images_dir, WSI_EXTs)
        annotations_paths = FileSystemUtils.find_files_with_extension(
            annotations_dir, ANNOTATION_EXTs
        )
        data = FileSystemUtils.map_images_to_annotations(img_paths, annotations_paths)

        if not data:
            raise FileNotFoundError(
                f"Input list of paths is empty. Cannot find corresponding pairs: image - annotation."
            )

        path_to_save_weak_labels_all_images = output_mask_dir / Path(
            "weak_labels_all_images.csv"
        )

        rows = []
        all_labels = {}
        for record in tqdm.tqdm(data, desc="Generating masks"):
            try:
                annotation_path = record["annotation"]
                image_path = record["image"]
                filename = image_path.stem + "_mask"
                output_mask_path = output_mask_dir / filename
                img_relative_path = os.path.relpath(image_path, data_dir)
                weak_labels, _ = MaskGenerationHelper.generate_mask_as_tiff(
                    image_path, annotation_path, output_mask_path, classes, xml_type
                )

                rows.append(
                    {"image_path": img_relative_path, "weak_labels": weak_labels}
                )
            except Exception as e:
                print(
                    f"An unexpected error occurred: {e} while proccesing {record['image']} file"
                )
                continue

        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(path_to_save_weak_labels_all_images, index=False)

    @staticmethod
    def analyze_background(dataset_config: pd.core.series.Series, data_dir: Path):
        """
        Analyzing annotations related to background and tissue.

        Parameters
        ----------
        dataset_config : pd.core.series.Series
            Dataset configuration series with information

        data_dir : Path
            Path to main data directory.

        Returns
        -------

        """
        images_dir = (
            Path(dataset_config.images_dir)
            if not pd.isna(dataset_config.images_dir)
            else None
        )
        bg_annotations_dir = (
            Path(dataset_config.bg_annotations_dir)
            if not pd.isna(dataset_config.bg_annotations_dir)
            else None
        )
        bg_annotations_type = (
            dataset_config.bg_annotations_type
            if not pd.isna(dataset_config.bg_annotations_type)
            else None
        )
        bg_masks_dir = (
            Path(dataset_config.bg_masks_dir)
            if not pd.isna(dataset_config.bg_masks_dir)
            else None
        )

        if bg_masks_dir != None and not bg_masks_dir.exists():
            bg_masks_dir.mkdir(parents=True)

        if (
            images_dir != None
            and images_dir.exists()
            and bg_annotations_dir != None
            and bg_annotations_dir.exists()
            and bg_masks_dir != None
            and bg_masks_dir.exists()
            and bg_annotations_type != None
        ):
            print("\nProcessing background annotations...")
            MaskGenerationHelper.generate_masks_from_annotations(
                bg_annotations_dir,
                images_dir,
                bg_masks_dir,
                data_dir,
                bg_annotations_type,
            )

    @staticmethod
    def analyze_ground_truth(dataset_config: pd.core.series.Series, data_dir: Path):
        """
        Analyzing annotations related to ground truths.

        Parameters
        ----------
        dataset_config : pd.core.series.Series
            Dataset configuration series with information

        data_dir : Path
            Path to main data directory.

        Returns
        -------
        """
        images_dir = (
            Path(dataset_config.images_dir)
            if not pd.isna(dataset_config.images_dir)
            else None
        )
        gt_annotations_dir = (
            Path(dataset_config.gt_annotations_dir)
            if not pd.isna(dataset_config.gt_annotations_dir)
            else None
        )
        gt_annotations_type = (
            dataset_config.gt_annotations_type
            if not pd.isna(dataset_config.gt_annotations_type)
            else None
        )
        gt_masks_dir = (
            Path(dataset_config.gt_masks_dir)
            if not pd.isna(dataset_config.gt_masks_dir)
            else None
        )

        if gt_masks_dir != None and not gt_masks_dir.exists():
            gt_masks_dir.mkdir(parents=True)

        if (
            images_dir != None
            and images_dir.exists()
            and gt_annotations_dir != None
            and gt_annotations_dir.exists()
            and gt_masks_dir != None
            and gt_masks_dir.exists()
            and gt_annotations_type != None
        ):
            print("\nProcessing ground truth annotations...")
            MaskGenerationHelper.generate_masks_from_annotations(
                gt_annotations_dir,
                images_dir,
                gt_masks_dir,
                data_dir,
                gt_annotations_type,
            )
