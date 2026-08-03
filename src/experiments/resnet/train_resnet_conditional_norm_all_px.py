import os
import sys
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from training.timing_callback import TimingCallback

from networks.resnet.resnet_classifier import ResNetClassifier
from networks.resnet.config_resnet import config
from datasets.single_wsi_dataset.inference_single_wsi_dataset import DownstreamTask
from datasets.data_module import MagInv_DataModule
from training.train import train_kfold_dataset
from utils.loading_utils import LoadingUtils
from utils.seed_utils import (
    add_reproducibility_args,
    resolve_deterministic_flag,
    set_global_seed,
)

from paths.paths import DATA_DIR, OUTPUTS_MODELS


def update_data_config(
    config: dict,
    data_dir: Path,
    fold_number: int,
    arg_removeBG,
    min_pixel_size: float = None,
    max_pixel_size: float = None,
    num_workers: int = 9,
    seed: int = 2024,
):

    wsi_data_csv_path = data_dir / Path(r"wsi_data.csv")
    wsi_data_csv = pd.read_csv(wsi_data_csv_path)
    datasubsets_summary_path = data_dir / Path(
        wsi_data_csv["train_val_test_split"].iloc[1]
    )
    train_csv_path, val_csv_path, _ = (
        LoadingUtils.get_train_val_test_paths_from_trainvaltest_subsets_summary_csv(
            datasubsets_summary_path
        )
    )
    config["data"]["data_dir_path"] = data_dir
    config["data"]["img_size"] = (224, 224)
    config["data"]["batch_size"] = 128
    config["data"]["num_workers"] = num_workers
    config["data"]["seed"] = seed
    config["data"]["train_data_csv"] = data_dir / train_csv_path
    config["data"]["val_data_csv"] = data_dir / val_csv_path
    config["data"]["pixel_size"] = None
    config["data"]["num_patches"] = 10000
    config["data"][
        "downstream_task"
    ] = DownstreamTask.STRONG_BINARY_MULTICLASS_CLASSIFICATION.value
    config["data"]["exclude_background_in_classification_targets"] = (
        True if arg_removeBG == 1 else False
    )
    if min_pixel_size is not None:
        config["data"]["min_pixel_size"] = min_pixel_size
    if max_pixel_size is not None:
        config["data"]["max_pixel_size"] = max_pixel_size


def update_model_config(config: dict, data: LightningDataModule):

    config["model"][
        "class_weights_dict"
    ] = None  # Set only in case of weighted cost function - data.get_class_weights()  else None
    config["model"]["num_classes"] = data.get_num_classes()
    config["model"]["classification_type"] = data.get_downstream_task_detailed()
    config["model"]["resnet_type"] = "resnet18"
    config["model"]["variant"] = "conditional_normalization"
    config["model"]["num_conditions"] = 2
    config["model"]["optimizer_init"]["init_args"]["lr"] = 0.001


def update_trainer_config(config: dict, output_data_dir: Path, folder_name: str):

    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", mode="min", patience=100, min_delta=0.0001, verbose=True
    )
    checkpoint_callback_vallos = ModelCheckpoint(
        save_top_k=1,
        filename="best-valid_loss",
        monitor="valid_loss",
        mode="min",
        save_last=True,
    )
    checkpoint_callback_f1 = ModelCheckpoint(
        save_top_k=1,
        filename="best-valid_f1",
        monitor="valid_f1_macro",
        mode="max",
        save_last=True,
    )
    checkpoint_callback_balancedAcc = ModelCheckpoint(
        save_top_k=1,
        filename="best-valid_accuracy",
        monitor="valid_accuracy",
        mode="max",
        save_last=True,
    )
    output_logs_dir = output_data_dir / folder_name
    config["trainer"]["default_root_dir"] = output_logs_dir
    config["trainer"]["callbacks"] = [
        early_stopping_callback,
        checkpoint_callback_vallos,
        checkpoint_callback_f1,
        checkpoint_callback_balancedAcc,
        TimingCallback(output_logs_dir),
    ]
    config["trainer"]["log_every_n_steps"] = (
        config["data"]["num_patches"] // config["data"]["batch_size"]
    )


def update_config(
    data_dir,
    output_models,
    args_fold,
    args_checkpoint_path,
    arg_unique_name,
    arg_removeBG,
    args_min_pixel_size=None,
    args_max_pixel_size=None,
    args_num_workers=9,
    args_seed=2024,
):

    update_data_config(
        config,
        data_dir,
        int(args_fold),
        arg_removeBG,
        args_min_pixel_size,
        args_max_pixel_size,
        num_workers=args_num_workers,
        seed=args_seed,
    )
    checkpoint_path = (
        Path(args_checkpoint_path)
        if isinstance(args_checkpoint_path, str)
        and len(args_checkpoint_path) > 0
        and Path(args_checkpoint_path).exists()
        else None
    )
    update_trainer_config(
        config, output_models, f"resnetCN18_fold_{args_fold}_{arg_unique_name}"
    )
    data = MagInv_DataModule(**config["data"])
    data.setup()
    update_model_config(config, data)
    return data, checkpoint_path, config


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fold", type=str, required=True, help="Number of fold.")
    parser.add_argument(
        "--unique_name",
        type=str,
        required=True,
        help="Dir for saving outputs, logs from torch lighting. Pass unique text to add to dirname: resnetCN18_fold_{args.fold}_{GIVEN_DIR_NAME}",
    )
    parser.add_argument(
        "--remove_BG",
        type=int,
        required=True,
        help="Int flag - removing background class for strong classification",
    )
    parser.add_argument(
        "--min_pixel_size",
        type=float,
        default=None,
        help="Minimum pixel size for random selection. If not provided, uses default from config.",
    )
    parser.add_argument(
        "--max_pixel_size",
        type=float,
        default=None,
        help="Maximum pixel size for random selection. If not provided, uses default from config.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Checkpoint model path."
    )
    parser = add_reproducibility_args(parser)
    args = parser.parse_args()

    set_global_seed(args.seed)

    data, checkpoint_path, _ = update_config(
        Path(DATA_DIR),
        Path(OUTPUTS_MODELS),
        args.fold,
        args.checkpoint_path,
        args.unique_name,
        args.remove_BG,
        args.min_pixel_size,
        args.max_pixel_size,
        args_num_workers=args.num_workers,
        args_seed=args.seed,
    )
    config["trainer"]["default_root_dir"].mkdir(parents=True, exist_ok=True)
    config["trainer"]["deterministic"] = resolve_deterministic_flag(args.deterministic)
    model = ResNetClassifier(**config["model"])
    train_kfold_dataset(model, config, data, checkpoint_path)
