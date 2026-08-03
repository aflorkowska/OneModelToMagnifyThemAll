import ast
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from datasets.single_wsi_dataset.abstract.abstract_single_wsi_dataset import (
    DownstreamTask,
)
from datasets.single_wsi_dataset.histopathology_transform import TransformConfig


class DataCSVLoader:
    """
    CSV data loader and validator.

    Pipeline:
        load_csv -> parse_df -> clean_df -> postprocess
    """

    def load_csv(
        self,
        csv_file_path: Path,
        data_dir_path: Path,
        downstream_task: DownstreamTask,
        transform_config: TransformConfig | None = None,
        should_balance_batch: bool = False,
        background_label: int | None = None,
        sample_background_patches: bool = True,
        min_samples_per_class: int | None = 1000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        data_dir_path = Path(data_dir_path)

        if not data_dir_path.exists():
            raise FileNotFoundError(f"Data dir {data_dir_path} does not exist")

        if not Path(csv_file_path).exists():
            raise FileNotFoundError(f"CSV file {csv_file_path} does not exist")

        df = pd.read_csv(csv_file_path)
        df = self.parse_df(df)
        cleaned_df, dropped_df = self.clean_df(df, downstream_task, data_dir_path)

        if transform_config and transform_config.apply_mask_mapping:

            cleaned_df["image_class_distribiution"] = cleaned_df[
                "image_class_distribiution"
            ].apply(lambda d: self._map_and_sum(d, transform_config.mask_mapping))

            if downstream_task in (
                DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
                DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
                DownstreamTask.SEGMENTATION,
            ):
                cleaned_df["image_classes"] = cleaned_df["image_classes"].apply(
                    lambda lst: sorted(
                        set(transform_config.mask_mapping.get(x, x) for x in lst)
                    )
                )

        if min_samples_per_class is not None:
            cleaned_df = self._filter_insufficient_samples(
                cleaned_df,
                background_label,
                sample_background_patches,
                min_samples_per_class=min_samples_per_class,
            )

        if should_balance_batch:
            cleaned_df = self._balance_batch(
                cleaned_df, background_label, sample_background_patches
            )

        return cleaned_df, dropped_df

    @staticmethod
    def parse_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        path_columns = [
            "image_path",
            "mask_tissue_path",
            "dense_label_path",
        ]

        for col in path_columns:
            df[col] = df[col].apply(
                lambda v: v if isinstance(v, str) and len(v) > 0 else np.nan
            )

        literal_columns = [
            "weak_label",
            "image_classes",
            "all_labels_in_dataset",
            "image_class_distribiution",
        ]

        for col in literal_columns:
            df[col] = df[col].apply(lambda v: ast.literal_eval(v) if pd.notna(v) else v)

        return df

    def clean_df(
        self,
        df: pd.DataFrame,
        downstream_task,
        data_dir_path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        df = df.copy()
        drop_reason = pd.Series(None, index=df.index, dtype="object")
        valid_mask = pd.Series(True, index=df.index)

        valid_mask &= self._path_exists_mask(df, "image_path", data_dir_path)
        drop_reason.loc[~valid_mask] = "missing or non-existing image_path"

        next_mask = self._path_exists_mask(df, "mask_tissue_path", data_dir_path)
        drop_reason.loc[valid_mask & ~next_mask] = (
            "missing or non-existing mask_tissue_path"
        )
        valid_mask &= next_mask

        distrib_mask = df["image_class_distribiution"].notna()
        drop_reason.loc[valid_mask & ~distrib_mask] = (
            "missing image_class_distribiution"
        )
        valid_mask &= distrib_mask

        if downstream_task == DownstreamTask.WEAK_CLASSIFICATION:
            weak_mask = self._weak_label_mask(df)
            drop_reason.loc[valid_mask & ~weak_mask] = "missing or empty weak_label"
            valid_mask &= weak_mask

        elif downstream_task in (
            DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION,
            DownstreamTask.STRONG_MULTILABEL_CLASSIFICATION,
            DownstreamTask.SEGMENTATION,
        ):
            dense_mask = self._path_exists_mask(df, "dense_label_path", data_dir_path)
            drop_reason.loc[valid_mask & ~dense_mask] = (
                "missing or non-existing dense_label_path"
            )
            valid_mask &= dense_mask

            required_cols_mask = self._required_columns_mask(
                df, ["image_classes", "all_labels_in_dataset"]
            )
            drop_reason.loc[valid_mask & ~required_cols_mask] = (
                "missing image_classes or all_labels_in_dataset"
            )
            valid_mask &= required_cols_mask

            all_labels_mask = self._all_labels_mask(df)
            drop_reason.loc[valid_mask & ~all_labels_mask] = (
                "all_labels_in_dataset must have at least 2 labels"
            )
            valid_mask &= all_labels_mask

            img_classes_mask = self._image_classes_mask(df)
            drop_reason.loc[valid_mask & ~img_classes_mask] = (
                "empty or background-only image_classes"
            )
            valid_mask &= img_classes_mask

        cleaned_df = df[valid_mask].copy()
        dropped_df = df[~valid_mask].copy()
        dropped_df["drop_reason"] = drop_reason.loc[~valid_mask]

        return cleaned_df, dropped_df

    @staticmethod
    def _filter_insufficient_samples(
        df: pd.DataFrame,
        background_label: int | None,
        sample_background_patches: bool,
        min_samples_per_class: int = 1000,
    ) -> pd.DataFrame:

        def has_enough_samples(d: dict) -> bool:
            for k, v in d.items():
                if not sample_background_patches and k == background_label:
                    continue
                if not (v >= min_samples_per_class or v == 0):
                    return False
            return True

        return df[df["image_class_distribiution"].apply(has_enough_samples)]

    @staticmethod
    def _balance_batch(
        df: pd.DataFrame,
        background_label: int,
        sample_background_patches: bool,
    ) -> pd.DataFrame:
        df["splitted_label"] = df["image_classes"]
        df = df.explode("splitted_label").reset_index(drop=True)

        if not sample_background_patches:
            df = df[df["splitted_label"] != background_label].reset_index(drop=True)

        class_weights = DataCSVLoader._calculate_class_weights(df)
        df["dataset_class_weights"] = [class_weights] * len(df)

        df["class_probability"] = df["splitted_label"].apply(
            lambda lbl: class_weights.get(lbl, 0)
        )

        return df

    @staticmethod
    def _path_exists_mask(df, column, base_path: Path):
        return df[column].apply(
            lambda p: pd.notna(p) and (base_path / Path(p)).exists()
        )

    @staticmethod
    def _required_columns_mask(df, columns: list[str]):
        mask = pd.Series(True, index=df.index)
        for col in columns:
            mask &= df[col].notna()
        return mask

    @staticmethod
    def _weak_label_mask(df):
        return df["weak_label"].apply(lambda x: pd.notna(x) and bool(x))

    @staticmethod
    def _image_classes_mask(df):
        return df["image_classes"].apply(
            lambda x: (isinstance(x, (list, tuple)) and len(x) > 0 and x != [0])
        )

    @staticmethod
    def _all_labels_mask(df):
        return df["all_labels_in_dataset"].apply(
            lambda x: (isinstance(x, (list, tuple)) and len(set(x)) >= 2)
        )

    @staticmethod
    def _map_and_sum(d: dict, mapping: dict) -> dict:
        new_counts = defaultdict(int)
        for k, v in d.items():
            if k in mapping:
                new_counts[mapping[k]] += v
        return dict(new_counts)

    @staticmethod
    def _calculate_class_weights(df: pd.DataFrame) -> dict:
        labels = df["splitted_label"].tolist()
        counts = Counter(labels)
        return {cls: 1 / cnt for cls, cnt in counts.items()}
