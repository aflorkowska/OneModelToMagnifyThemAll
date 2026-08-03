import os
import sys
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from training.timing_callback import TimingCallback

from networks.unet.unet import UNet
from networks.unet.config_unet import config
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
    pixel_size: float,
    fold_number: int,
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
    config["data"]["batch_size"] = 128  # 64 / 128
    config["data"]["num_workers"] = num_workers
    config["data"]["seed"] = seed
    config["data"]["train_data_csv"] = data_dir / train_csv_path
    config["data"]["val_data_csv"] = data_dir / val_csv_path
    config["data"]["pixel_size"] = (pixel_size, pixel_size)
    config["data"]["num_patches"] = 10000
    config["data"]["downstream_task"] = DownstreamTask.SEGMENTATION.value
    config["data"]["exclude_background_in_classification_targets"] = False
    if min_pixel_size is not None:
        config["data"]["min_pixel_size"] = min_pixel_size
    if max_pixel_size is not None:
        config["data"]["max_pixel_size"] = max_pixel_size


def update_model_config(config: dict, data: LightningDataModule):

    config["model"][
        "class_weights_dict"
    ] = None  # Set only in case of weighted cost function - data.get_class_weights()  else None
    config["model"]["n_classes"] = data.get_num_classes()
    config["model"]["variant"] = "base"
    config["model"]["segmentation_type"] = data.get_downstream_task_detailed()
    config["model"]["features_start"] = 32
    config["model"]["bilinear"] = False
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
    checkpoint_callback_dice = ModelCheckpoint(
        save_top_k=1,
        filename="best-valid_dice",
        monitor="valid_dice",
        mode="max",
        save_last=True,
    )
    checkpoint_callback_iou = ModelCheckpoint(
        save_top_k=1,
        filename="best-valid_iou",
        monitor="valid_iou",
        mode="max",
        save_last=True,
    )
    output_logs_dir = output_data_dir / folder_name
    config["trainer"]["default_root_dir"] = output_logs_dir
    config["trainer"]["callbacks"] = [
        early_stopping_callback,
        checkpoint_callback_vallos,
        checkpoint_callback_dice,
        checkpoint_callback_iou,
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
    args_pixel_size,
    arg_unique_name,
    args_min_pixel_size=None,
    args_max_pixel_size=None,
    args_num_workers=9,
    args_seed=2024,
):

    pixel_size = float(args_pixel_size)
    pixel_size_str = str(args_pixel_size).replace(".", "_")
    update_data_config(
        config,
        data_dir,
        pixel_size,
        int(args_fold),
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
        config,
        output_models,
        f"unet_fold_{args_fold}_pixel_size_{pixel_size_str}_{arg_unique_name}",
    )
    data = MagInv_DataModule(**config["data"])
    data.setup()
    update_model_config(config, data)
    return data, checkpoint_path, config


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Number of fold.")
    parser.add_argument(
        "--pixel_size", type=float, required=True, help="Pixel size (pixel is squared)."
    )
    parser.add_argument(
        "--unique_name",
        type=str,
        required=True,
        help="Dir for saving outputs, logs from torch lighting. Pass unique text to add to dirname: unet2D_fold_{args.fold}_pixel_size_{pixel_size_str}_{GIVEN_DIR_NAME}",
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
        args.pixel_size,
        args.unique_name,
        args.min_pixel_size,
        args.max_pixel_size,
        args_num_workers=args.num_workers,
        args_seed=args.seed,
    )
    config["trainer"]["default_root_dir"].mkdir(parents=True, exist_ok=True)
    config["trainer"]["deterministic"] = resolve_deterministic_flag(args.deterministic)
    model = UNet(**config["model"])
    train_kfold_dataset(model, config, data, checkpoint_path)
