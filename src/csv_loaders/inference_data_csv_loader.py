import pandas as pd
from pathlib import Path
from csv_loaders.data_csv_loader import DataCSVLoader


class InferenceDataCSVLoader(DataCSVLoader):
    """
    CSV loader for inference/validation dataset:
    - Always disables batch balancing
    """

    def load_csv(
        self,
        csv_file_path: Path,
        data_dir_path: Path,
        downstream_task,
        transform_config=None,
        background_label: int | None = None,
        sample_background_patches: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return super().load_csv(
            csv_file_path,
            data_dir_path,
            downstream_task,
            transform_config,
            should_balance_batch=False,
            background_label=background_label,
            sample_background_patches=sample_background_patches,
            min_samples_per_class=None,
        )
