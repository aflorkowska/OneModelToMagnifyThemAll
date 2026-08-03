"""
Abstract base class for dataset parsers.

This module provides an abstract parser that handles common operations for dataset parsing,
including k-fold splitting, train/val split CSV generation, and dataset summary creation.
"""

import os
import sys
import ast
import math
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from paths.paths import DATA_DIR, LIBHAA_DIR

sys.path.append(os.path.join(os.path.dirname(__file__), LIBHAA_DIR))

from libhaa.scripts.segment import main as segmentation_main
from parsers.config import WSI_EXTs
from parsers.utils.mask_generator_helper import MaskGenerationHelper
from parsers.utils.file_system_utils import FileSystemUtils
from parsers.utils.annotation_parser import AnnotationParser
from utils.wsi_loading_utils import WSILoadingUtils


class AbstractParser(ABC):
    """
    Abstract base class for dataset parsers.

    This class provides common functionality for parsing different histopathology datasets,
    including dataset CSV creation, k-fold splitting, and train/val split generation.
    """

    ALL_DATASETS_SUMMARIES_CSV = Path(DATA_DIR) / Path(r"wsi_data.csv")

    def __init__(
        self,
        dataset_filename: Path,
        kfold: int = 5,
        data_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        """
        Initialize the parser.

        Parameters
        ----------
        dataset_filename : Path
            Name of the dataset folder
        kfold : int, optional
            Number of k-fold splits (default: 5)
        data_dir : Path, optional
            Data directory path (default: DATA_DIR from paths.paths)
        seed : int, optional
            Random seed used for k-fold splitting and train/val/test shuffling (default: 42)
        """
        self.dataset_filename = dataset_filename
        self.kfold = kfold
        self.data_dir = data_dir if data_dir is not None else Path(DATA_DIR)
        self.dataset_dir = self.data_dir / dataset_filename
        self.seed = seed
        self.logger = None

    def setup_logger(self, logger_filename: str) -> logging.Logger:
        """
        Setup logger for the parser.

        Parameters
        ----------
        logger_filename : str
            Name of the log file

        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        logger_path = self.dataset_dir / logger_filename
        logging.basicConfig(
            filename=logger_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        return self.logger

    @abstractmethod
    def prepare_dataset(self) -> None:
        """
        Prepare dataset-specific structure and files.

        This method should be implemented by subclasses to handle
        dataset-specific preparation tasks (e.g., renaming directories,
        creating weak labels, saving class definitions).
        """
        pass

    @abstractmethod
    def create_paths_summary(self) -> pd.DataFrame:
        """
        Create a DataFrame with dataset configuration and paths.

        Returns
        -------
        pd.DataFrame
            DataFrame with a single row containing dataset configuration.
            Must include columns: dataset_dir, images_dir, gt_masks_dir, files_summary, etc.
        """
        pass

    def create_datasets_csv(
        self, dataset_config: pd.Series
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Create dataset CSV files for different data providers.

        This method scans the dataset directories, matches images with masks and labels,
        and creates CSV files for each data provider.

        Parameters
        ----------
        dataset_config : pd.Series
            Series containing dataset configuration

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            Relative paths to the created dataset CSV files
        """
        images_dir = (
            Path(dataset_config.images_dir)
            if not pd.isna(dataset_config.images_dir)
            else None
        )
        bg_masks_dir = (
            Path(dataset_config.bg_masks_dir)
            if not pd.isna(dataset_config.bg_masks_dir)
            else None
        )
        gt_masks_dir = (
            Path(dataset_config.gt_masks_dir)
            if not pd.isna(dataset_config.gt_masks_dir)
            else None
        )
        files_summary_csv = (
            Path(dataset_config.files_summary)
            if not pd.isna(dataset_config.files_summary)
            else None
        )
        all_labels_csv_list = self._get_all_labels_csvs(dataset_config)
        images_class_distribiution_csv = (
            Path(dataset_config.image_class_distribution)
            if not pd.isna(dataset_config.image_class_distribution)
            else None
        )

        FileSystemUtils.check_if_required_data_for_dataset_summary_generation_exist(
            images_dir,
            bg_masks_dir,
            gt_masks_dir,
            files_summary_csv,
            all_labels_csv_list,
            images_class_distribiution_csv,
        )

        img_paths = (
            FileSystemUtils.find_files_with_extension(images_dir, WSI_EXTs)
            if images_dir is not None
            else []
        )
        bg_mask_paths = (
            FileSystemUtils.find_files_with_extension(bg_masks_dir, WSI_EXTs)
            if bg_masks_dir is not None
            else []
        )
        gt_mask_paths = (
            FileSystemUtils.find_files_with_extension(gt_masks_dir, WSI_EXTs)
            if gt_masks_dir is not None
            else None
        )
        df_file_summary = (
            pd.read_csv(files_summary_csv) if files_summary_csv is not None else None
        )
        df_images_class_dist = (
            pd.read_csv(images_class_distribiution_csv)
            if images_class_distribiution_csv is not None
            else None
        )

        if not img_paths or len(img_paths) != len(bg_mask_paths):
            print(
                f"\nCheck content of the following dirs! images: {images_dir} - {len(img_paths)} imgs,\n"
                f"bg masks: {bg_masks_dir} - {len(bg_mask_paths)} masks "
            )

        all_datasets_data = self._process_images(
            img_paths,
            bg_mask_paths,
            gt_mask_paths,
            df_file_summary,
            df_images_class_dist,
            dataset_config,
        )

        dataset_paths = self._save_datasets_by_provider(
            all_datasets_data, dataset_config
        )
        return dataset_paths

    def _get_all_labels_csvs(self, dataset_config: pd.Series) -> List[Optional[Path]]:
        """
        Extract all label CSV paths from dataset configuration.

        This method can be overridden by subclasses to specify
        which label CSVs are relevant for their dataset.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        List[Optional[Path]]
            List of label CSV paths
        """
        labels_csvs = []
        for col in dataset_config.index:
            if "classes" in col.lower() and not pd.isna(dataset_config[col]):
                labels_csvs.append(Path(dataset_config[col]))
        return labels_csvs if labels_csvs else [None]

    def _process_images(
        self,
        img_paths: List[Path],
        bg_mask_paths: List[Path],
        gt_mask_paths: Optional[List[Path]],
        df_file_summary: Optional[pd.DataFrame],
        df_images_class_dist: Optional[pd.DataFrame],
        dataset_config: pd.Series,
    ) -> Dict[str, List[Dict]]:
        """
        Process images and create dataset records.

        Parameters
        ----------
        img_paths : List[Path]
            List of image paths
        bg_mask_paths : List[Path]
            List of background mask paths
        gt_mask_paths : Optional[List[Path]]
            List of ground truth mask paths
        df_file_summary : Optional[pd.DataFrame]
            DataFrame with file summary information
        df_images_class_dist : Optional[pd.DataFrame]
            DataFrame with image class distribution
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary mapping provider names to lists of image data dictionaries
        """
        import tqdm

        all_datasets_data = self._init_datasets_dict(dataset_config)

        for img_path in tqdm.tqdm(img_paths, desc="Processing images"):
            bg_mask_path = FileSystemUtils.find_matching_file(img_path, bg_mask_paths)
            gt_mask_path = (
                FileSystemUtils.find_matching_file(img_path, gt_mask_paths)
                if gt_mask_paths is not None
                else None
            )
            weak_label = (
                AnnotationParser.find_weak_label(df_file_summary, img_path)
                if isinstance(df_file_summary, pd.DataFrame)
                else np.nan
            )

            img_path_processed = (
                np.nan
                if not WSILoadingUtils.can_open_slide(img_path)
                else Path(os.path.relpath(img_path, self.data_dir))
            )
            if bg_mask_path is not None:
                bg_mask_path = (
                    np.nan
                    if not WSILoadingUtils.can_open_slide(bg_mask_path)
                    else os.path.relpath(bg_mask_path, self.data_dir)
                )
            else:
                bg_mask_path = np.nan

            image_metadata = {
                "image_path": img_path_processed,
                "mask_tissue_path": bg_mask_path,
                "weak_label": weak_label,
            }

            if gt_mask_path is not None:
                gt_mask_path_processed = (
                    np.nan
                    if not WSILoadingUtils.can_open_slide(gt_mask_path)
                    else os.path.relpath(gt_mask_path, self.data_dir)
                )
                filtered_row = df_images_class_dist[
                    df_images_class_dist["dense_label_path"] == gt_mask_path_processed
                ]
                image_metadata.update(
                    {
                        "dense_label_path": gt_mask_path_processed,
                        "image_class_distribiution": (
                            filtered_row["image_class_distribution"].values[0]
                            if not filtered_row.empty
                            else np.nan
                        ),
                        "image_classes": (
                            filtered_row["image_classes"].values[0]
                            if not filtered_row.empty
                            else np.nan
                        ),
                        "bg_mask_took_into_account_when_image_class_distribution_calculation": (
                            filtered_row[
                                "bg_mask_took_into_account_when_image_class_distribution_calculation"
                            ].values[0]
                            if not filtered_row.empty
                            else np.nan
                        ),
                    }
                )
            else:
                image_metadata.update(
                    {
                        "dense_label_path": np.nan,
                        "image_class_distribiution": np.nan,
                        "image_classes": np.nan,
                        "bg_mask_took_into_account_when_image_class_distribution_calculation": np.nan,
                    }
                )

            data_provider = (
                df_file_summary[df_file_summary["image_id"] == img_path_processed.stem][
                    "data_provider"
                ].values[0]
                if isinstance(df_file_summary, pd.DataFrame)
                else None
            )

            self._add_image_to_provider(
                image_metadata, data_provider, all_datasets_data, dataset_config
            )

        return all_datasets_data

    def _init_datasets_dict(self, dataset_config: pd.Series) -> Dict[str, List[Dict]]:
        """
        Initialize datasets dictionary for different providers.

        This method can be overridden by subclasses to define
        which data providers should be tracked.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary with provider names as keys and empty lists as values
        """
        return {"default": []}

    def _add_image_to_provider(
        self,
        image_metadata: Dict,
        data_provider: Optional[str],
        all_datasets_data: Dict[str, List[Dict]],
        dataset_config: pd.Series,
    ) -> None:
        """
        Add image metadata to appropriate provider dataset.

        This method can be overridden by subclasses for custom provider handling.

        Parameters
        ----------
        image_metadata : Dict
            Image metadata dictionary
        data_provider : Optional[str]
            Name of the data provider
        all_datasets_data : Dict[str, List[Dict]]
            Accumulator for all datasets
        dataset_config : pd.Series
            Dataset configuration
        """
        if data_provider and data_provider in all_datasets_data:
            image_metadata["all_labels_in_dataset"] = self._get_labels_for_provider(
                data_provider, dataset_config
            )
            all_datasets_data[data_provider].append(image_metadata)
        else:
            image_metadata["all_labels_in_dataset"] = self._get_labels_for_provider(
                "default", dataset_config
            )
            all_datasets_data["default"].append(image_metadata)

    def _get_labels_for_provider(
        self, provider: str, dataset_config: pd.Series
    ) -> Optional[List]:
        """
        Get class labels for a specific provider.

        This method should be overridden by subclasses to return
        provider-specific labels.

        Parameters
        ----------
        provider : str
            Provider name
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        Optional[List]
            List of labels or None
        """
        return None

    def _save_datasets_by_provider(
        self, all_datasets_data: Dict[str, List[Dict]], dataset_config: pd.Series
    ) -> Tuple[Optional[str], ...]:
        """
        Save dataset CSVs for each provider.

        This method can be overridden by subclasses for custom saving logic.

        Parameters
        ----------
        all_datasets_data : Dict[str, List[Dict]]
            Dictionary mapping providers to image data
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        Tuple[Optional[str], ...]
            Tuple of relative paths to saved CSV files
        """
        paths = []
        for provider, data_list in all_datasets_data.items():
            if data_list:
                df_data = pd.DataFrame(data_list)
                save_dir = dataset_config.dataset_dir / Path(f"{provider}_provider")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / Path("dataset_summary.csv")
                df_data.to_csv(save_path, index=False)
                paths.append(os.path.relpath(save_path, self.data_dir))
            else:
                paths.append(None)
        return tuple(paths) if paths else (None,)

    def run_dataset_setup(self) -> None:
        """
        Execute the complete dataset setup pipeline.

        This method orchestrates the entire parsing process:
        1. Prepare dataset
        2. Create paths summary
        3. Generate background and ground truth masks (optional)
        4. Calculate class distribution
        5. Create dataset CSVs
        6. Split into k-fold subsets
        7. Split into train/val/test subsets
        8. Save metadata to master CSV
        """
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

        self.logger.info(f"Starting dataset setup for {self.dataset_filename}")

        self.prepare_dataset()
        self.logger.info("Dataset prepared")

        df = self.create_paths_summary()
        self.logger.info("Paths summary created")

        dataset_config = df.iloc[0].copy()

        self._generate_masks_if_needed(dataset_config)
        self.logger.info("Mask generation completed")

        self._calculate_class_distribution_if_needed(dataset_config)
        self.logger.info("Class distribution calculated")

        dataset_paths = self.create_datasets_csv(dataset_config)
        self.logger.info("Dataset CSVs created")

        if all(p is None for p in dataset_paths):
            self.logger.error("No dataset paths generated. Aborting.")
            return

        kfold_paths = self._create_kfold_splits(dataset_config, dataset_paths)
        self.logger.info("K-fold splits created")

        train_val_test_paths = self._create_train_val_test_splits(
            dataset_config, dataset_paths
        )
        self.logger.info("Train/val/test splits created")

        self._update_master_csv(dataset_paths, kfold_paths, train_val_test_paths)
        self.logger.info("Master CSV updated")

    def _generate_masks_if_needed(self, dataset_config: pd.Series) -> None:
        """
        Generate background and ground truth masks if needed.

        This method orchestrates the mask generation process:
        1. Generate background XML annotations (if segmentation_lvl is specified)
        2. Generate background masks from XML annotations
        3. Generate ground truth masks from XML annotations

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration
        """
        if not pd.isna(dataset_config.get("images_dir")) and not pd.isna(
            dataset_config.get("segmentation_lvl")
        ):
            self._generate_background_annotations(dataset_config)

        if not pd.isna(dataset_config.get("images_dir")) and not pd.isna(
            dataset_config.get("bg_annotations_dir")
        ):
            self._generate_background_masks(dataset_config)

        if not pd.isna(dataset_config.get("images_dir")) and not pd.isna(
            dataset_config.get("gt_annotations_dir")
        ):
            self._generate_ground_truth_masks(dataset_config)

    def _generate_background_annotations(self, dataset_config: pd.Series) -> None:
        """
        Generate background XML annotations using segmentation model.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration with images_dir and segmentation_lvl
        """
        try:
            bg_annotations_dir = Path(dataset_config["dataset_dir"]) / Path(
                r"bg_annotations"
            )
            if not bg_annotations_dir.exists():
                bg_annotations_dir.mkdir(parents=True, exist_ok=True)

            segmentation_lvl = dataset_config["segmentation_lvl"]
            weights_path = Path(self.data_dir) / Path(r"weights_v11.07.2023.tar")

            if self.logger:
                self.logger.info("Generating background XML annotations")
            print("Generating bg xml annotations")

            segmentation_main(
                Path(dataset_config["images_dir"]),
                weights_path,
                bg_annotations_dir,
                segmentation_lvl,
            )
            dataset_config["bg_annotations_dir"] = bg_annotations_dir

            if self.logger:
                self.logger.info(
                    f"Background annotations saved to {bg_annotations_dir}"
                )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error generating background annotations: {str(e)}",
                    exc_info=True,
                )
            print(f"Error generating background annotations: {str(e)}")

    def _generate_background_masks(self, dataset_config: pd.Series) -> None:
        """
        Generate background masks from XML annotations.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration with bg_annotations_dir
        """
        try:
            bg_masks_dir = Path(dataset_config["dataset_dir"]) / Path(r"bg_masks")
            dataset_config["bg_masks_dir"] = bg_masks_dir

            if self.logger:
                self.logger.info("Generating background masks from annotations")
            print("Generating bg masks")

            MaskGenerationHelper.analyze_background(dataset_config, self.data_dir)

            if self.logger:
                self.logger.info(f"Background masks saved to {bg_masks_dir}")
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error generating background masks: {str(e)}", exc_info=True
                )
            print(f"Error generating background masks: {str(e)}")

    def _generate_ground_truth_masks(self, dataset_config: pd.Series) -> None:
        """
        Generate ground truth masks from XML annotations.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration with gt_annotations_dir
        """
        try:
            gt_masks_dir = Path(dataset_config["dataset_dir"]) / Path(r"gt_masks")
            dataset_config["gt_masks_dir"] = gt_masks_dir

            if self.logger:
                self.logger.info("Generating ground truth masks from annotations")
            print("Generating gt masks")

            MaskGenerationHelper.analyze_ground_truth(dataset_config, self.data_dir)

            if self.logger:
                self.logger.info(f"Ground truth masks saved to {gt_masks_dir}")
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error generating ground truth masks: {str(e)}", exc_info=True
                )
            print(f"Error generating ground truth masks: {str(e)}")

    def _calculate_class_distribution_if_needed(
        self, dataset_config: pd.Series
    ) -> None:
        """
        Calculate class distribution from ground truth masks if they exist.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration with gt_masks_dir
        """
        try:
            if pd.isna(dataset_config.get("gt_masks_dir")):
                return

            gt_masks_dir = Path(dataset_config["dataset_dir"]) / Path(r"gt_masks")
            path_to_save_images_class_distribution = gt_masks_dir / Path(
                r"images_class_distribution.csv"
            )
            dataset_config["image_class_distribution"] = (
                path_to_save_images_class_distribution
            )

            if self.logger:
                self.logger.info(
                    "Calculating class distribution based on ground truth masks"
                )
            print("Calculating class distribiution based on gt masks")

            MaskGenerationHelper.calculate_class_distribution_from_gt_masks(
                dataset_config, path_to_save_images_class_distribution, self.data_dir
            )

            if self.logger:
                self.logger.info(
                    f"Class distribution saved to {path_to_save_images_class_distribution}"
                )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error calculating class distribution: {str(e)}", exc_info=True
                )
            print(f"Error calculating class distribution: {str(e)}")

    def _create_kfold_splits(
        self, dataset_config: pd.Series, dataset_paths: Tuple
    ) -> List[str]:
        """
        Create k-fold splits for each dataset provider.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration
        dataset_paths : Tuple
            Tuple of paths to dataset CSV files

        Returns
        -------
        List[str]
            List of paths to k-fold summary CSV files
        """
        kfold_paths = []
        providers = self._get_providers_for_kfold(dataset_config)

        for idx, dataset_path in enumerate(dataset_paths):
            if dataset_path is None:
                kfold_paths.append(None)
                continue

            provider = providers[idx] if idx < len(providers) else "default"
            provider_dir = dataset_config.dataset_dir / Path(f"{provider}_provider")

            kfold_summary_path = self.split_dataset_csv_into_kfold_subsets(
                dataset_csv_path=self.data_dir / dataset_path,
                dataset_dir=provider_dir,
                unique_name=f"_{provider}" if provider != "default" else "",
            )
            kfold_paths.append(kfold_summary_path)

        return kfold_paths

    def _get_providers_for_kfold(self, dataset_config: pd.Series) -> List[str]:
        """
        Get list of providers for k-fold splitting.

        This method can be overridden by subclasses.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        List[str]
            List of provider names
        """
        return ["default"]

    def split_dataset_csv_into_kfold_subsets(
        self,
        dataset_csv_path: Path,
        dataset_dir: Path,
        unique_name: str = "",
    ) -> str:
        """
        Split dataset CSV into k-fold subsets with stratification by class.

        Creates train and validation CSVs for each fold using StratifiedGroupKFold
        based on unique image paths to prevent data leakage while maintaining
        class balance across folds.

        Parameters
        ----------
        dataset_csv_path : Path
            Path to the dataset CSV file
        dataset_dir : Path
            Directory to save k-fold subsets
        unique_name : str, optional
            Suffix for the output directory name

        Returns
        -------
        str
            Relative path to the k-fold summary CSV
        """
        base_name = "kfold_datasubsets" + unique_name
        output_dir_for_kfold_datasets = dataset_dir / Path(base_name)
        output_dir_for_kfold_datasets.mkdir(parents=True, exist_ok=True)

        summary_filename = base_name + "_summary.csv"
        output_datasubsets_summary_csv_path = dataset_dir / Path(summary_filename)
        fold_paths = []

        df = pd.read_csv(dataset_csv_path)

        df_with_classes = df[df["image_classes"].notna()].copy()
        if df_with_classes.empty:
            if self.logger:
                self.logger.warning(
                    f"No rows with image_classes found in {dataset_csv_path}. Using simple GroupKFold."
                )
            groups = df["image_path"].values
            kf = GroupKFold(n_splits=self.kfold)
            use_stratified = False
        else:
            df_with_classes["image_classes"] = df_with_classes["image_classes"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            df_with_classes["class_combo"] = df_with_classes["image_classes"].apply(
                lambda x: str(tuple(sorted(x)))
            )

            combo_counts = df_with_classes["class_combo"].value_counts()
            min_samples = combo_counts.min()

            if min_samples < self.kfold:
                if self.logger:
                    self.logger.warning(
                        f"Some class combinations have fewer than {self.kfold} samples "
                        f"(minimum: {min_samples}). Falling back to GroupKFold."
                    )
                groups = df["image_path"].values
                kf = GroupKFold(n_splits=self.kfold)
                use_stratified = False
            else:
                groups = df_with_classes["image_path"].values
                strata = df_with_classes["class_combo"].values
                kf = StratifiedGroupKFold(
                    n_splits=self.kfold, random_state=self.seed, shuffle=True
                )
                use_stratified = True
                df = df_with_classes.reset_index(drop=True)

        if use_stratified:
            split_iterator = kf.split(df, strata, groups)
        else:
            split_iterator = kf.split(df, [0] * len(df), groups)

        for fold, (train_idx, val_idx) in enumerate(split_iterator):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            train_csv_path = output_dir_for_kfold_datasets / Path(
                f"train_fold_{fold}.csv"
            )
            val_csv_path = output_dir_for_kfold_datasets / Path(f"val_fold_{fold}.csv")
            train_df.to_csv(train_csv_path, index=False)
            val_df.to_csv(val_csv_path, index=False)
            train_csv_relpath = os.path.relpath(train_csv_path, self.data_dir)
            val_csv_relpath = os.path.relpath(val_csv_path, self.data_dir)

            fold_paths.append(
                {"fold": fold, "train": train_csv_relpath, "val": val_csv_relpath}
            )

        fold_paths_df = pd.DataFrame(fold_paths)
        fold_paths_df.to_csv(output_datasubsets_summary_csv_path, index=False)
        output_datasubsets_summary_csv_relative_path = os.path.relpath(
            output_datasubsets_summary_csv_path, self.data_dir
        )
        return output_datasubsets_summary_csv_relative_path

    def _create_train_val_test_splits(
        self, dataset_config: pd.Series, dataset_paths: Tuple
    ) -> List[Optional[str]]:
        """
        Create train/val/test splits for each dataset provider.

        Parameters
        ----------
        dataset_config : pd.Series
            Dataset configuration
        dataset_paths : Tuple
            Tuple of paths to dataset CSV files

        Returns
        -------
        List[Optional[str]]
            List of paths to train/val/test split summary CSV files
        """
        train_val_test_paths = []
        providers = self._get_providers_for_kfold(dataset_config)

        for idx, dataset_path in enumerate(dataset_paths):
            if dataset_path is None:
                train_val_test_paths.append(None)
                continue

            provider = providers[idx] if idx < len(providers) else "default"
            provider_dir = dataset_config.dataset_dir / Path(f"{provider}_provider")

            if self.logger:
                self.logger.info(
                    f"Creating train/val/test split for {provider} provider"
                )
            print(f"\n=== Creating train/val/test split for {provider} provider ===")

            split_summary_path = self.split_dataset_csv_into_train_val_test(
                dataset_csv_path=self.data_dir / dataset_path,
                dataset_dir=provider_dir,
                unique_name=f"_{provider}" if provider != "default" else "",
            )

            if split_summary_path is None:
                if self.logger:
                    self.logger.warning(
                        f"Failed to create train/val/test split for {provider} provider"
                    )
                print(f"Failed to create train/val/test split for {provider} provider")
            else:
                if self.logger:
                    self.logger.info(
                        f"Train/val/test split created successfully for {provider}: {split_summary_path}"
                    )
                print(f"Train/val/test split created: {split_summary_path}")

            train_val_test_paths.append(split_summary_path)

        return train_val_test_paths

    def split_dataset_csv_into_train_val_test(
        self,
        dataset_csv_path: Path,
        dataset_dir: Path,
        test_ratio: float = 0.2,
        test_max: int = 600,
        val_ratio: float = 0.1,
        unique_name: str = "",
    ) -> Optional[str]:
        """
        Split dataset CSV into train/val/test subsets maintaining class balance.

        Strategy:
        - Test = min(20% of total, 600 images)
        - Val = 10% of remaining (train) images
        - Train = remaining images

        Uses class combinations to stratify the split and ensure balanced distribution.

        Parameters
        ----------
        dataset_csv_path : Path
            Path to the dataset CSV file
        dataset_dir : Path
            Directory to save splits
        test_ratio : float, optional
            Ratio of data to allocate for testing (default: 0.2)
        test_max : int, optional
            Maximum number of test samples (default: 600)
        val_ratio : float, optional
            Ratio of train data to allocate for validation (default: 0.1)
        unique_name : str, optional
            Suffix for the output directory name

        Returns
        -------
        Optional[str]
            Relative path to the train/val/test split summary CSV, or None if split failed
        """
        try:
            if self.logger:
                self.logger.info(
                    f"Starting train/val/test split for {dataset_csv_path}"
                )

            base_name = "train_val_split" + unique_name
            output_dir = dataset_dir / Path(base_name)
            output_dir.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(dataset_csv_path)
            print(f"  Loaded {len(df)} total samples from {dataset_csv_path.name}")
            print(f"  Columns: {list(df.columns)}")

            if self.logger:
                self.logger.info(f"Loaded dataset with {len(df)} total samples")

            df_with_classes = df[df["image_classes"].notna()].copy()
            print(f"  Found {len(df_with_classes)} samples with image_classes")

            if self.logger:
                self.logger.info(
                    f"Found {len(df_with_classes)} samples with image_classes"
                )

            if df_with_classes.empty:
                if self.logger:
                    self.logger.warning(
                        f"No rows with image_classes found in {dataset_csv_path}. Skipping train/val/test split."
                    )
                print(
                    f"WARNING: No image_classes found in {dataset_csv_path} - skipping train/val/test split"
                )
                return None

            df_with_classes["image_classes"] = df_with_classes["image_classes"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            df_with_classes["class_combo"] = df_with_classes["image_classes"].apply(
                lambda x: str(tuple(sorted(x)))
            )

            total_samples = len(df_with_classes)
            print(f"Total samples with classes: {total_samples}")

            test_size = min(int(total_samples * test_ratio), test_max)

            if test_size == 0 and total_samples > 2:
                test_size = 1
                if self.logger:
                    self.logger.info(
                        f"Dataset has only {total_samples} samples. Using minimum test_size=1"
                    )
                print(f"Using minimum test_size=1 for small dataset")

            if test_size == 0:
                if self.logger:
                    self.logger.warning(
                        f"Dataset too small ({total_samples} samples). Skipping split."
                    )
                print(
                    f"Dataset too small ({total_samples} samples) - need at least 3 samples"
                )
                return None

            print(f"Test size: {test_size}")

            combo_counts = df_with_classes["class_combo"].value_counts()
            combo_ratios = combo_counts / combo_counts.sum()
            test_counts = (combo_ratios * test_size).astype(int)

            train_rows, val_rows, test_rows = [], [], []

            for combo in combo_counts.index:
                subset = (
                    df_with_classes[df_with_classes["class_combo"] == combo]
                    .sample(frac=1, random_state=self.seed)
                    .reset_index(drop=True)
                )
                n = len(subset)
                n_test = test_counts.get(combo, 0)

                if n <= 2:
                    if n == 1:
                        train_rows.append(subset)
                    elif n == 2:
                        train_rows.append(subset.iloc[[0]])
                        test_rows.append(subset.iloc[[1]])
                    continue

                test_part = subset.iloc[:n_test]
                remaining = subset.iloc[n_test:].reset_index(drop=True)

                val_size = math.ceil(len(remaining) * val_ratio)
                if val_size == 0:
                    train_rows.append(remaining)
                else:
                    val_part = remaining.iloc[:val_size]
                    train_part = remaining.iloc[val_size:]
                    val_rows.append(val_part)
                    train_rows.append(train_part)

                test_rows.append(test_part)

            train_df = (
                pd.concat([d for d in train_rows if not d.empty]).reset_index(drop=True)
                if train_rows
                else pd.DataFrame()
            )
            val_df = (
                pd.concat([d for d in val_rows if not d.empty]).reset_index(drop=True)
                if val_rows
                else pd.DataFrame()
            )
            test_df = (
                pd.concat([d for d in test_rows if not d.empty]).reset_index(drop=True)
                if test_rows
                else pd.DataFrame()
            )

            if train_df.empty or val_df.empty or test_df.empty:
                if self.logger:
                    self.logger.error(
                        f"One of train/val/test subsets is empty for {dataset_csv_path}. "
                        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}. Skipping split."
                    )
                print(
                    f"Empty subset(s): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
                )
                print(
                    f"Dataset too small to split into train/val/test - use k-fold instead"
                )
                return None

            print(
                f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
            )

            train_df = train_df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )
            val_df = val_df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )
            test_df = test_df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

            train_csv_path = output_dir / Path("train.csv")
            val_csv_path = output_dir / Path("val.csv")
            test_csv_path = output_dir / Path("test.csv")

            train_df.to_csv(train_csv_path, index=False)
            val_df.to_csv(val_csv_path, index=False)
            test_df.to_csv(test_csv_path, index=False)

            split_paths_df = pd.DataFrame(
                {
                    "train": [os.path.relpath(train_csv_path, self.data_dir)],
                    "val": [os.path.relpath(val_csv_path, self.data_dir)],
                    "test": [os.path.relpath(test_csv_path, self.data_dir)],
                }
            )

            split_summary_csv_path = output_dir / Path("train_val_test_split.csv")
            split_paths_df.to_csv(split_summary_csv_path, index=False)

            if self.logger:
                self.logger.info(
                    f"Train/val/test split created: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
                )

            return os.path.relpath(split_summary_csv_path, self.data_dir)

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error during train/val/test split for {dataset_csv_path}: {str(e)}",
                    exc_info=True,
                )
            return None

    def _update_master_csv(
        self,
        dataset_paths: Tuple,
        kfold_paths: List[str],
        train_val_test_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Update the master datasets CSV file.

        Parameters
        ----------
        dataset_paths : Tuple
            Tuple of relative paths to dataset CSV files
        kfold_paths : List[str]
            List of relative paths to k-fold summary CSVs
        train_val_test_paths : Optional[List[str]]
            List of relative paths to train/val/test split summary CSVs
        """
        new_datasets = []
        for i in range(len(dataset_paths)):
            if dataset_paths[i] is not None:
                new_row = {
                    "dataset_path": dataset_paths[i],
                    "kfold_datasubsets_path": kfold_paths[i],
                }
                if train_val_test_paths and i < len(train_val_test_paths):
                    new_row["train_val_test_split"] = train_val_test_paths[i]
                new_datasets.append(new_row)

        if os.path.exists(self.ALL_DATASETS_SUMMARIES_CSV):
            df_wsi_data = pd.read_csv(self.ALL_DATASETS_SUMMARIES_CSV)
            new_row = pd.DataFrame(new_datasets)
            df_wsi_data = pd.concat([df_wsi_data, new_row], ignore_index=True)
        else:
            df_wsi_data = pd.DataFrame(new_datasets)

        df_wsi_data.to_csv(self.ALL_DATASETS_SUMMARIES_CSV, index=False)
