import pandas as pd
from pathlib import Path
from csv_loaders.data_csv_loader import DataCSVLoader


class TrainingDataCSVLoader(DataCSVLoader):
    """
    CSV loader for training dataset:
    - Batch balancing optional via should_balance_batch parameter
    """

    def load_csv(
        self,
        csv_file_path: Path,
        data_dir_path: Path,
        downstream_task,
        transform_config=None,
        should_balance_batch: bool = True,
        background_label: int | None = None,
        sample_background_patches: bool = True,
        min_samples_per_class: int = 1000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return super().load_csv(
            csv_file_path,
            data_dir_path,
            downstream_task,
            transform_config,
            should_balance_batch=should_balance_batch,
            background_label=background_label,
            sample_background_patches=sample_background_patches,
            min_samples_per_class=min_samples_per_class,
        )
