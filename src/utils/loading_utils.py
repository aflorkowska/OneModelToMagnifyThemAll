import pandas as pd
from pathlib import Path


class LoadingUtils:
    """Utility class for loading paths from CSV files."""

    @staticmethod
    def get_train_val_paths_from_kfold_subsets_summary_csv(
        summary_csv: Path, fold: int
    ):
        """
        Extract train and validation paths for a specific fold from a k-fold summary CSV.

        Args:
            summary_csv: Path to the CSV file containing fold information
            fold: Fold number to retrieve

        Returns:
            Tuple of (train_path, val_path) as Path objects, or (None, None) if fold not found
        """
        df = pd.read_csv(summary_csv)
        if fold not in df["fold"].values:
            print(
                f"Given folds number: {fold} not found in given csv. Folds number set to 0."
            )
            fold = 0

        row = df[df["fold"] == fold]
        if row.empty:
            print(f"Fold {fold} not found in the CSV file.")
            return None, None

        train_path = row["train"].values[0]
        val_path = row["val"].values[0]
        return Path(train_path), Path(val_path)

    @staticmethod
    def get_train_val_test_paths_from_trainvaltest_subsets_summary_csv(
        summary_csv: Path,
    ):
        """
        Extract train, validation, and test paths from a train/val/test summary CSV.

        Args:
            summary_csv: Path to the CSV file containing subset information

        Returns:
            Tuple of (train_path, val_path, test_path) as Path objects
        """
        df = pd.read_csv(summary_csv).iloc[0]
        train_path = df["train"]
        val_path = df["val"]
        test_path = df["test"]
        return Path(train_path), Path(val_path), Path(test_path)
