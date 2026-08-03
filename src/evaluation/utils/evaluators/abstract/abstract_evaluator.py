import torch
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from utils.downstreamtask_utils import DownstreamTaskUtils
from datasets.single_wsi_dataset.histopathology_transform import TransformConfig
from csv_loaders.inference_data_csv_loader import InferenceDataCSVLoader
from datasets.single_wsi_dataset.training_single_wsi_dataset import DownstreamTask


class AbstractEvaluator(ABC):
    def __init__(
        self,
        data_dir_path: Path,
        output_dir_path: Path,
        model_checkpoint_path: Path,
        model_type: str,
        unique_name: str,
        csv_file_path: Path,
        patch_size: Tuple[int, int],
        transform_config: TransformConfig | None = None,
        tested_pixel_size: Tuple[float, float] | None = None,
        trained_pixel_size: Tuple[float, float] | None = None,
        downstream_task: DownstreamTask = DownstreamTask.NONE,
        exclude_background_in_classification_targets: bool = True,
        background_label: int = 0,
        sample_background_patches: bool = False,
        priority_class: int | None = None,
        pixel_size_tolerance_percent_coeff: Tuple[float, float] = (0.05, 0.05),
        padding_value: int = 255,
    ) -> None:

        self.data_dir_path = data_dir_path
        self.output_csv_path = self._create_output_csv_path(
            output_dir_path,
            model_type,
            trained_pixel_size,
            tested_pixel_size,
            unique_name,
        )
        self.model_checkpoint_path = model_checkpoint_path
        if not self.model_checkpoint_path.exists():
            raise FileNotFoundError("Given model checkpoint path does not exist")

        self.model_type = model_type
        self.csv_file_path = self.data_dir_path / csv_file_path
        self.patch_size = patch_size
        self.transform_config = transform_config
        self.tested_pixel_size = tested_pixel_size
        self.trained_pixel_size = trained_pixel_size
        self.downstream_task = DownstreamTask(downstream_task)
        self.priority_class = priority_class
        self.exclude_background_in_classification_targets = (
            exclude_background_in_classification_targets
        )
        self.background_label = background_label
        self.sample_background_patches = sample_background_patches
        self.pixel_size_tolerance_percent_coeff = pixel_size_tolerance_percent_coeff
        self.padding_value = padding_value

        loader = InferenceDataCSVLoader()
        self._loaded_data, _ = loader.load_csv(
            csv_file_path=self.csv_file_path,
            data_dir_path=self.data_dir_path,
            downstream_task=self.downstream_task,
            transform_config=self.transform_config,
            background_label=self.background_label,
            sample_background_patches=self.sample_background_patches,
        )
        self._num_classes = DownstreamTaskUtils.calculate_num_classes(self)
        self._downstream_task_detailed = (
            DownstreamTaskUtils.set_detailed_downstream_task(self)
        )

        self.model = self._get_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pixel_size_to_str(pixel_size):
        return (
            f"{str(pixel_size).replace('.', 'p')}" if pixel_size is not None else "None"
        )

    def _create_output_csv_path(
        self,
        output_dir_path: Path,
        model_type,
        trained_pixel_size,
        tested_pixel_size,
        unique_name,
    ):

        trained_pixel_size_str = self._pixel_size_to_str(trained_pixel_size)
        tested_pixel_size_str = self._pixel_size_to_str(tested_pixel_size)
        filename = f"{str(model_type)}_trained_PS_{trained_pixel_size_str}_test_PS_{tested_pixel_size_str}_{unique_name}.csv"
        return output_dir_path / filename

    @abstractmethod
    def run_evaluation(self):
        pass

    @abstractmethod
    def _get_model(self):
        pass
