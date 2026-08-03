"""
Parser for PANDA (Prostate cANcer graDe Assessment) dataset.

This module implements PANDAParser, a concrete implementation of AbstractParser
for parsing the PANDA dataset with its specific structure and requirements.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from parsers.config import AnnotationDataType
from parsers.utils.mask_processor import MaskProcessor
from parsers.abstract.abstract_parser import AbstractParser


class PANDAParser(AbstractParser):
    """
    Parser for the PANDA (Prostate cANcer graDe Assessment) dataset.

    The PANDA dataset consists of:
    - Radboud data provider: 6 classes (background, stroma, healthy, Gleason 3-5)
    - Karolinska data provider: 3 classes (background, benign tissue, cancerous tissue)

    This parser handles the specific structure and requirements of the PANDA dataset,
    including the two different label schemes for different data providers.
    """

    RADBOUD_CLASSES = {
        "background": 0,
        "stroma (connective tissue, non-epithelium tissue)": 1,
        "healthy (benign)": 2,
        "cancerous epithelium (Gleason 3)": 3,
        "cancerous epithelium (Gleason 4)": 4,
        "cancerous epithelium (Gleason 5)": 5,
    }

    KAROLINSKA_CLASSES = {
        "background (non tissue) or unknown": 0,
        "benign tissue (stroma and epithelium combined)": 1,
        "cancerous tissue (stroma and epithelium combined)": 2,
    }

    def __init__(
        self,
        dataset_filename: Path = Path(r"PANDAChallenge"),
        kfold: int = 5,
        data_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        """
        Initialize the PANDA parser.

        Parameters
        ----------
        dataset_filename : Path, optional
            Name of the dataset folder (default: "PANDAChallenge")
        kfold : int, optional
            Number of k-fold splits (default: 5)
        data_dir : Path, optional
            Data directory path (default: DATA_DIR from paths.paths)
        seed : int, optional
            Random seed used for k-fold splitting and train/val/test shuffling (default: 42)
        """
        super().__init__(dataset_filename, kfold, data_dir, seed)

    def prepare_dataset(self) -> None:
        """
        Prepare PANDA dataset structure.

        This method:
        1. Initializes or creates the gt_masks directory
        2. Adds weak_labels column to train.csv if it doesn't exist
        3. Creates class definition CSVs for both Radboud and Karolinska providers
        """
        gt_masks_dir = self.dataset_dir / Path(r"gt_masks")
        gt_masks_dir.mkdir(parents=True, exist_ok=True)

        train_files_summary_csv = self.dataset_dir / Path(r"train.csv")
        if train_files_summary_csv.exists():
            df = pd.read_csv(train_files_summary_csv)
            if "weak_labels" not in df.columns:
                df["weak_labels"] = [[] for _ in range(len(df))]
                df.to_csv(train_files_summary_csv, index=False)
                if self.logger:
                    self.logger.info("Added weak_labels column to train.csv")

        all_classes_csv_path_radboud = gt_masks_dir / Path(
            r"radboud_labels_explanation.csv"
        )
        MaskProcessor.save_labels_explanation_to_csv(
            self.RADBOUD_CLASSES, all_classes_csv_path_radboud
        )
        if self.logger:
            self.logger.info(
                f"Saved Radboud class definitions to {all_classes_csv_path_radboud}"
            )

        all_classes_csv_path_karolinska = gt_masks_dir / Path(
            r"karolinska_labels_explanation.csv"
        )
        MaskProcessor.save_labels_explanation_to_csv(
            self.KAROLINSKA_CLASSES, all_classes_csv_path_karolinska
        )
        if self.logger:
            self.logger.info(
                f"Saved Karolinska class definitions to {all_classes_csv_path_karolinska}"
            )

    def create_paths_summary(self) -> pd.DataFrame:
        """
        Create PANDA dataset configuration summary.

        Returns
        -------
        pd.DataFrame
            DataFrame with single row containing PANDA-specific paths and configuration.
            Includes separate class definitions for both Radboud and Karolinska providers.
        """
        data = {
            "dataset_dir": self.dataset_dir,
            "images_dir": self.dataset_dir / Path(r"images"),
            "bg_annotations_dir": np.nan,
            "bg_annotations_type": AnnotationDataType.PIXELS_LVL_0,
            "bg_masks_dir": np.nan,
            "gt_annotations_dir": np.nan,
            "gt_annotations_type": np.nan,
            "gt_masks_dir": self.dataset_dir / Path(r"gt_masks"),
            "segmentation_lvl": 2,
            "files_summary": self.dataset_dir / Path(r"train.csv"),
            "all_classes_radboud": self.dataset_dir
            / Path(r"gt_masks")
            / Path(r"radboud_labels_explanation.csv"),
            "all_classes_karolinska": self.dataset_dir
            / Path(r"gt_masks")
            / Path(r"karolinska_labels_explanation.csv"),
            "image_class_distribiution": np.nan,
        }

        df = pd.DataFrame(data, index=[0])
        return df

    def _init_datasets_dict(self, dataset_config: pd.Series) -> Dict[str, List[Dict]]:
        """
        Initialize datasets dictionary for both PANDA providers.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary with 'radboud' and 'karolinska' keys, each mapping to empty list
        """
        return {"radboud": [], "karolinska": []}

    def _get_all_labels_csvs(self, dataset_config: pd.Series) -> List[Optional[Path]]:
        """
        Extract PANDA-specific label CSV paths.

        Returns
        -------
        List[Optional[Path]]
            List containing Radboud and Karolinska label CSV paths
        """
        radboud_csv = (
            Path(dataset_config.all_classes_radboud)
            if not pd.isna(dataset_config.all_classes_radboud)
            else None
        )
        karolinska_csv = (
            Path(dataset_config.all_classes_karolinska)
            if not pd.isna(dataset_config.all_classes_karolinska)
            else None
        )
        return [radboud_csv, karolinska_csv]

    def _get_labels_for_provider(
        self, provider: str, dataset_config: pd.Series
    ) -> Optional[List[int]]:
        """
        Get class labels specific to a PANDA provider.

        Parameters
        ----------
        provider : str
            Provider name ('radboud', 'karolinska', or 'default')
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        Optional[List[int]]
            List of class labels for the provider
        """
        if provider == "radboud":
            return list(self.RADBOUD_CLASSES.values())
        elif provider == "karolinska":
            return list(self.KAROLINSKA_CLASSES.values())
        else:
            return None

    def _add_image_to_provider(
        self,
        image_metadata: Dict,
        data_provider: Optional[str],
        all_datasets_data: Dict[str, List[Dict]],
        dataset_config: pd.Series,
    ) -> None:
        """
        Add image metadata to appropriate PANDA provider dataset.

        Handles provider-specific label assignment.

        Parameters
        ----------
        image_metadata : Dict
            Image metadata dictionary
        data_provider : Optional[str]
            Name of the data provider ('radboud' or 'karolinska')
        all_datasets_data : Dict[str, List[Dict]]
            Accumulator for all datasets
        dataset_config : pd.Series
            Dataset configuration
        """
        if data_provider == "radboud":
            image_metadata["all_labels_in_dataset"] = list(
                self.RADBOUD_CLASSES.values()
            )
            all_datasets_data["radboud"].append(image_metadata)
        elif data_provider == "karolinska":
            image_metadata["all_labels_in_dataset"] = list(
                self.KAROLINSKA_CLASSES.values()
            )
            all_datasets_data["karolinska"].append(image_metadata)
        else:
            if self.logger:
                self.logger.warning(
                    f"Unknown data provider: {data_provider}. Skipping image: {image_metadata.get('image_path')}"
                )

    def _save_datasets_by_provider(
        self, all_datasets_data: Dict[str, List[Dict]], dataset_config: pd.Series
    ) -> Tuple[Optional[str], ...]:
        """
        Save PANDA dataset CSVs for each provider (Radboud and Karolinska).

        Parameters
        ----------
        all_datasets_data : Dict[str, List[Dict]]
            Dictionary mapping providers to image data lists
        dataset_config : pd.Series
            Dataset configuration

        Returns
        -------
        Tuple[Optional[str], Optional[str]]
            Relative paths to Radboud and Karolinska dataset CSVs
        """
        providers_order = ["radboud", "karolinska"]
        paths = []

        for provider in providers_order:
            data_list = all_datasets_data.get(provider, [])
            if data_list:
                df_data = pd.DataFrame(data_list)
                save_dir = dataset_config.dataset_dir / Path(f"{provider}_provider")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / Path("dataset_summary.csv")
                df_data.to_csv(save_path, index=False)
                paths.append(os.path.relpath(save_path, self.data_dir))
                if self.logger:
                    self.logger.info(
                        f"Saved {provider} dataset CSV: {save_path} ({len(data_list)} images)"
                    )
            else:
                paths.append(None)
                if self.logger:
                    self.logger.warning(f"No data found for provider: {provider}")

        return tuple(paths)

    def _get_providers_for_kfold(self, dataset_config: pd.Series) -> List[str]:
        """
        Get PANDA-specific providers for k-fold splitting.

        Returns
        -------
        List[str]
            List containing ['radboud', 'karolinska']
        """
        return ["radboud", "karolinska"]
