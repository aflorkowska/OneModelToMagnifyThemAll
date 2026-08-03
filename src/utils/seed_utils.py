"""
Central reproducibility helpers shared by all training entry points.

Seeding must happen before any dataset/model/sampler objects are constructed,
since DataLoader workers, WeightedRandomSampler and weight initialization all
draw from the global RNG state at construction/iteration time.
"""

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import seed_everything as pl_seed_everything

DETERMINISTIC_CHOICES = {"true": True, "false": False, "warn": "warn"}


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl_seed_everything(seed, workers=True)


def add_reproducibility_args(parser: ArgumentParser) -> ArgumentParser:
    """Add --seed, --num_workers and --deterministic flags shared by all training scripts."""
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed applied to python/numpy/torch and DataLoader workers.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=9,
        help="Number of DataLoader worker processes (increase for GPU training to keep the GPU fed).",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="true",
        choices=list(DETERMINISTIC_CHOICES),
        help="pl.Trainer determinism mode: 'true' (default) forces deterministic CUDA ops "
        "(errors on any op with no deterministic implementation), "
        "'warn' uses deterministic ops where available and warns otherwise, "
        "'false' disables it for maximum throughput.",
    )
    return parser


def resolve_deterministic_flag(value: str) -> bool | str:
    return DETERMINISTIC_CHOICES[value]
