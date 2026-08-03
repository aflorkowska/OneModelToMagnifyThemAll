import torch
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from paths.paths import DATA_DIR, OUTPUTS_MODELS, OUTPUTS_EVALUATION
from utils.loading_utils import LoadingUtils

from experiments.unet.train_unet_base import (
    update_config as update_config_unet2D_pxDiff,
)
from experiments.unet.train_unet_conditional_norm_all_px import (
    update_config as update_config_unetCN2D_pxNone,
)
from evaluation.utils.evaluators.unet_evaluator import UnetEvaluator
from evaluation.utils.evaluators.abstract.abstract_evaluator import AbstractEvaluator
from training.evaluation_timer import EvaluationTimer

BENCHMARK_BATCH_SIZE = 128
BENCHMARK_MEASURE_BATCHES = 50


def prepare_training_dataloader(
    mode: str,
    checkpoint_path: str,
    pixel_size: float,
    unique_name: str,
):
    fold = 0
    if mode == "unet_px":
        data, checkpoint_path, config = update_config_unet2D_pxDiff(
            Path(DATA_DIR),
            Path(OUTPUTS_MODELS),
            fold,
            checkpoint_path,
            pixel_size,
            unique_name,
        )
    if mode == "unetCN":
        data, checkpoint_path, config = update_config_unetCN2D_pxNone(
            Path(DATA_DIR), Path(OUTPUTS_MODELS), fold, checkpoint_path, unique_name
        )
    data.setup()
    return data, config


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_checkpoint_path", type=str, required=True)
    parser.add_argument("--trained_pixel_size", type=str, required=True)
    parser.add_argument("--tested_pixel_size", type=str, required=True)
    parser.add_argument("--unique_name", type=str, required=True)
    args = parser.parse_args()

    trained_pixel_size = (
        float(args.trained_pixel_size) if args.trained_pixel_size != "None" else None
    )
    tested_pixel_size = (
        float(args.tested_pixel_size) if args.tested_pixel_size != "None" else None
    )

    checkpoint_path = Path(args.model_checkpoint_path)

    if args.model_type == "unetCN":
        data_module, config = prepare_training_dataloader(
            "unetCN", args.model_checkpoint_path, None, args.unique_name
        )
    else:
        data_module, config = prepare_training_dataloader(
            "unet_px", args.model_checkpoint_path, trained_pixel_size, args.unique_name
        )

    patch_size = config["data"]["img_size"]
    stride = None
    class_weights_dict = config["model"]["class_weights_dict"]
    optimizer_init = config["model"]["optimizer_init"]
    lr_scheduler_init = config["model"]["lr_scheduler_init"]
    num_conditions = (
        config["model"]["num_conditions"] if args.model_type in ["unetCN"] else None
    )
    downstream_task = config["data"]["downstream_task"]

    transform_config = data_module.get_train_transforms()
    exclude_background_in_classification_targets = (
        data_module.exclude_background_in_classification_targets
    )
    background_label = data_module._background_label
    sample_background_patches = data_module._sample_background_patches
    priority_class = data_module._priority_class

    data_dir_path = Path(DATA_DIR)
    wsi_data_csv_path = data_dir_path / Path(r"wsi_data.csv")
    wsi_data_csv = pd.read_csv(wsi_data_csv_path)
    datasubsets_summary_path = data_dir_path / Path(
        wsi_data_csv["train_val_test_split"].iloc[1]
    )
    _, _, test_csv_path = (
        LoadingUtils.get_train_val_test_paths_from_trainvaltest_subsets_summary_csv(
            datasubsets_summary_path
        )
    )

    output_dir_path = (
        Path(OUTPUTS_EVALUATION)
        / Path("BENCHMARK")
        / Path(
            f"{args.model_type}_trained_PS_{AbstractEvaluator._pixel_size_to_str(trained_pixel_size)}_{args.unique_name}_bs{BENCHMARK_BATCH_SIZE}"
        )
    )
    print(f"Benchmark timing outputs will be saved to: {output_dir_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    evaluator = UnetEvaluator(
        data_dir_path=data_dir_path,
        output_dir_path=output_dir_path,
        output_mask_path=Path(OUTPUTS_EVALUATION),
        model_type=args.model_type,
        unique_name=args.unique_name,
        model_checkpoint_path=checkpoint_path,
        csv_file_path=test_csv_path,
        patch_size=patch_size,
        stride=stride,
        batch_size=BENCHMARK_BATCH_SIZE,
        transform_config=transform_config,
        tested_pixel_size=tested_pixel_size,
        trained_pixel_size=trained_pixel_size,
        downstream_task=downstream_task,
        exclude_background_in_classification_targets=exclude_background_in_classification_targets,
        background_label=background_label,
        sample_background_patches=sample_background_patches,
        priority_class=priority_class,
        n_channels=config["model"]["n_channels"],
        features_start=config["model"]["features_start"],
        bilinear=config["model"]["bilinear"],
        image_channels=config["model"]["n_channels"],
        class_weights_dict=class_weights_dict,
        optimizer_init=optimizer_init,
        lr_scheduler_init=lr_scheduler_init,
        num_conditions=num_conditions,
    )

    H, W = patch_size
    x_dummy = torch.randn(BENCHMARK_BATCH_SIZE, 3, H, W, device=evaluator.device)
    pixel_size_dummy = torch.full(
        (BENCHMARK_BATCH_SIZE, 2),
        fill_value=trained_pixel_size if trained_pixel_size is not None else 1.0,
        device=evaluator.device,
    )

    timer = EvaluationTimer(output_dir_path, measure_batches=BENCHMARK_MEASURE_BATCHES)
    timer.on_eval_start()

    with torch.no_grad():
        num_warmup = EvaluationTimer._WARMUP_BATCHES
        for _ in range(BENCHMARK_MEASURE_BATCHES + num_warmup):
            if timer.limit_reached:
                break

            try:
                timer.on_batch_data_ready(BENCHMARK_BATCH_SIZE)

                if evaluator.model_type == "unet":
                    _ = evaluator.model(x_dummy)
                elif evaluator.model_type in ["unetCN"]:
                    _ = evaluator.model(x_dummy, pixel_size_dummy)
                else:
                    raise ValueError(
                        f"UNKNOWN MODEL TYPE {evaluator.model_type} IN UNET BENCHMARK"
                    )

                timer.on_batch_end()
            except Exception as e:
                print(f"[WARNING] Batch failed | {repr(e)}")

    timer.save()
    print(f"Benchmark complete. Results saved to: {output_dir_path}")
