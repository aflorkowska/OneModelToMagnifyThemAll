import csv
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch


class TimingCallback(pl.Callback):

    def __init__(self, output_dir: Path):
        super().__init__()
        self._output_dir = Path(output_dir)
        self._batches_csv = self._output_dir / "timing_batches.csv"
        self._epochs_csv = self._output_dir / "timing_epochs.csv"

        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._start_event = None
            self._end_event = None

        self._epoch_start: float = 0.0
        self._compute_start: float = 0.0
        self._prev_end: float = 0.0

        self._batch_rows: list[dict] = []
        self._current_epoch: int = 0
        self._headers_written: bool = False

    def _ensure_headers(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if not self._batches_csv.exists():
            with open(self._batches_csv, "w", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=["epoch", "batch_idx", "batch_size", "dataloader_ms", "compute_ms", "compute_ms_per_image", "total_ms", "warmup"],
                ).writeheader()
        if not self._epochs_csv.exists():
            with open(self._epochs_csv, "w", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "total_epoch_s",
                        "num_batches",
                        "warmup_batches_excluded",
                        "dataloader_mean_ms",
                        "dataloader_std_ms",
                        "dataloader_median_ms",
                        "dataloader_iqr_ms",
                        "dataloader_min_ms",
                        "dataloader_max_ms",
                        "dataloader_total_ms",
                        "compute_mean_ms",
                        "compute_std_ms",
                        "compute_median_ms",
                        "compute_iqr_ms",
                        "compute_min_ms",
                        "compute_max_ms",
                        "compute_total_ms",
                        "compute_per_image_mean_ms",
                        "compute_per_image_median_ms",
                        "compute_per_image_iqr_ms",
                    ],
                ).writeheader()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self._headers_written:
            self._ensure_headers()
            self._headers_written = True
        self._current_epoch = trainer.current_epoch
        if self._use_cuda:
            torch.cuda.synchronize()
        self._epoch_start = time.perf_counter()
        self._prev_end = self._epoch_start
        self._batch_rows = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Sync ensures H2D transfers from data prep are done before starting
        # the dataloader clock, and that no prior GPU tail bleeds into it.
        if self._use_cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        dataloader_ms = (now - self._prev_end) * 1000.0

        is_warmup = self._current_epoch == 0

        if self._use_cuda:
            self._start_event.record()
        else:
            self._compute_start = now

        batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]

        self._batch_rows.append(
            {
                "epoch": self._current_epoch,
                "batch_idx": batch_idx,
                "batch_size": batch_size,
                "dataloader_ms": dataloader_ms,
                "warmup": is_warmup,
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            compute_ms = self._start_event.elapsed_time(self._end_event)
        else:
            now = time.perf_counter()
            compute_ms = (now - self._compute_start) * 1000.0

        self._prev_end = time.perf_counter()

        row = self._batch_rows[-1]
        row["compute_ms"] = compute_ms
        row["compute_ms_per_image"] = compute_ms / row["batch_size"]
        row["total_ms"] = row["dataloader_ms"] + compute_ms

        with open(self._batches_csv, "a", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=["epoch", "batch_idx", "batch_size", "dataloader_ms", "compute_ms", "compute_ms_per_image", "total_ms", "warmup"],
            ).writerow(row)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._use_cuda:
            torch.cuda.synchronize()
        total_epoch_s = time.perf_counter() - self._epoch_start

        measured = [r for r in self._batch_rows if not r.get("warmup", False)]
        n = len(measured)
        warmup_excluded = len(self._batch_rows) - n

        if n == 0:
            return

        dl_arr = np.array([r["dataloader_ms"] for r in measured])
        compute_arr = np.array([r["compute_ms"] for r in measured])
        per_image_arr = np.array([r["compute_ms_per_image"] for r in measured])
        dl_q25, dl_q75 = np.percentile(dl_arr, [25, 75])
        comp_q25, comp_q75 = np.percentile(compute_arr, [25, 75])
        pi_q25, pi_q75 = np.percentile(per_image_arr, [25, 75])

        epoch_row = {
            "epoch": self._current_epoch,
            "total_epoch_s": round(total_epoch_s, 3),
            "num_batches": n,
            "warmup_batches_excluded": warmup_excluded,
            "dataloader_mean_ms": round(float(np.mean(dl_arr)), 3),
            "dataloader_std_ms": round(float(np.std(dl_arr)), 3),
            "dataloader_median_ms": round(float(np.median(dl_arr)), 3),
            "dataloader_iqr_ms": round(float(dl_q75 - dl_q25), 3),
            "dataloader_min_ms": round(float(np.min(dl_arr)), 3),
            "dataloader_max_ms": round(float(np.max(dl_arr)), 3),
            "dataloader_total_ms": round(float(np.sum(dl_arr)), 3),
            "compute_mean_ms": round(float(np.mean(compute_arr)), 3),
            "compute_std_ms": round(float(np.std(compute_arr)), 3),
            "compute_median_ms": round(float(np.median(compute_arr)), 3),
            "compute_iqr_ms": round(float(comp_q75 - comp_q25), 3),
            "compute_min_ms": round(float(np.min(compute_arr)), 3),
            "compute_max_ms": round(float(np.max(compute_arr)), 3),
            "compute_total_ms": round(float(np.sum(compute_arr)), 3),
            "compute_per_image_mean_ms": round(float(np.mean(per_image_arr)), 4),
            "compute_per_image_median_ms": round(float(np.median(per_image_arr)), 4),
            "compute_per_image_iqr_ms": round(float(pi_q75 - pi_q25), 4),
        }

        with open(self._epochs_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(epoch_row.keys())).writerow(epoch_row)
