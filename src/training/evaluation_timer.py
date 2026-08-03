import csv
import time
from pathlib import Path

import numpy as np
import torch


class EvaluationTimer:
    """Mirrors TimingCallback but for evaluation loops (no PL dependency).

    Skips the first _WARMUP_BATCHES to allow CUDA context init and cuDNN
    algorithm selection to complete. Uses CUDA Events for accurate GPU-side
    compute timing when a GPU is available.
    """

    _BATCH_FIELDNAMES = ["batch_idx", "batch_size", "dataloader_ms", "compute_ms", "total_ms", "compute_ms_per_image"]
    _WARMUP_BATCHES = 5
    _MEASURE_BATCHES = 50

    def __init__(self, output_dir: Path, measure_batches: int | None = None):
        self._output_dir = Path(output_dir)
        self._batches_csv = self._output_dir / "timing_eval_batches.csv"
        self._summary_csv = self._output_dir / "timing_eval_summary.csv"
        self._use_cuda = torch.cuda.is_available()
        if measure_batches is not None:
            self._MEASURE_BATCHES = measure_batches

        self._batch_rows: list[dict] = []
        self._prev_end: float = 0.0
        self._compute_start: float = 0.0
        self._batch_idx: int = 0
        self.limit_reached: bool = False

        if self._use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._start_event = None
            self._end_event = None

        self._output_dir.mkdir(parents=True, exist_ok=True)

    def on_eval_start(self):
        if self._use_cuda:
            torch.cuda.synchronize()
        self._batch_idx = 0
        self._batch_rows = []
        self.limit_reached = False
        self._prev_end = time.perf_counter()

        with open(self._batches_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._BATCH_FIELDNAMES).writeheader()

    def on_batch_data_ready(self, batch_size: int):
        """Call after .to(device) transfers and before model forward.

        Syncs CUDA so that H2D transfer time is NOT counted in compute_ms,
        and any tail from the previous batch's GPU work is NOT counted in
        dataloader_ms.
        """
        if self._use_cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        dataloader_ms = (now - self._prev_end) * 1000.0

        if self._use_cuda:
            self._start_event.record()
        else:
            self._compute_start = now

        if self._batch_idx >= self._WARMUP_BATCHES:
            self._batch_rows.append(
                {
                    "batch_idx": self._batch_idx - self._WARMUP_BATCHES,
                    "batch_size": batch_size,
                    "dataloader_ms": dataloader_ms,
                }
            )

    def on_batch_end(self):
        """Call right after the model forward pass."""
        if self._use_cuda:
            self._end_event.record()
            torch.cuda.synchronize()
            compute_ms = self._start_event.elapsed_time(self._end_event)
        else:
            now = time.perf_counter()
            compute_ms = (now - self._compute_start) * 1000.0

        self._prev_end = time.perf_counter()

        if self._batch_idx >= self._WARMUP_BATCHES and self._batch_rows:
            row = self._batch_rows[-1]
            row["compute_ms"] = compute_ms
            row["total_ms"] = row["dataloader_ms"] + compute_ms
            row["compute_ms_per_image"] = compute_ms / row["batch_size"]

            with open(self._batches_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._BATCH_FIELDNAMES).writerow(row)

        self._batch_idx += 1
        measured = self._batch_idx - self._WARMUP_BATCHES
        if measured == self._MEASURE_BATCHES:
            self.limit_reached = True
            self.save()

    def save(self):
        rows = [r for r in self._batch_rows if "compute_ms" in r]
        if not rows:
            return

        dl_arr = np.array([r["dataloader_ms"] for r in rows])
        compute_arr = np.array([r["compute_ms"] for r in rows])
        n = len(rows)

        def _stats(arr):
            q25, q75 = np.percentile(arr, [25, 75])
            return {
                "mean": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "median": round(float(np.median(arr)), 3),
                "iqr": round(float(q75 - q25), 3),
                "q25": round(float(q25), 3),
                "q75": round(float(q75), 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
            }

        dl = _stats(dl_arr)
        comp = _stats(compute_arr)

        summary = {
            "num_batches": n,
            "warmup_batches_skipped": self._WARMUP_BATCHES,
            "dataloader_mean_ms": dl["mean"],
            "dataloader_std_ms": dl["std"],
            "dataloader_median_ms": dl["median"],
            "dataloader_iqr_ms": dl["iqr"],
            "dataloader_q25_ms": dl["q25"],
            "dataloader_q75_ms": dl["q75"],
            "dataloader_min_ms": dl["min"],
            "dataloader_max_ms": dl["max"],
            "compute_mean_ms": comp["mean"],
            "compute_std_ms": comp["std"],
            "compute_median_ms": comp["median"],
            "compute_iqr_ms": comp["iqr"],
            "compute_q25_ms": comp["q25"],
            "compute_q75_ms": comp["q75"],
            "compute_min_ms": comp["min"],
            "compute_max_ms": comp["max"],
        }

        with open(self._summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        self._save_table_csv(summary, n)
        self._print_table(summary, n)

    def _save_table_csv(self, s: dict, n: int):
        table_csv = self._output_dir / "timing_eval_table.csv"
        rows = [
            ("num_batches",            n),
            ("warmup_batches_skipped", self._WARMUP_BATCHES),
            ("dataloader_mean",        s["dataloader_mean_ms"]),
            ("dataloader_std",         s["dataloader_std_ms"]),
            ("dataloader_median",      s["dataloader_median_ms"]),
            ("dataloader_iqr",         s["dataloader_iqr_ms"]),
            ("dataloader_q25",         s["dataloader_q25_ms"]),
            ("dataloader_q75",         s["dataloader_q75_ms"]),
            ("dataloader_min",         s["dataloader_min_ms"]),
            ("dataloader_max",         s["dataloader_max_ms"]),
            ("compute_mean",           s["compute_mean_ms"]),
            ("compute_std",            s["compute_std_ms"]),
            ("compute_median",         s["compute_median_ms"]),
            ("compute_iqr",            s["compute_iqr_ms"]),
            ("compute_q25",            s["compute_q25_ms"]),
            ("compute_q75",            s["compute_q75_ms"]),
            ("compute_min",            s["compute_min_ms"]),
            ("compute_max",            s["compute_max_ms"]),
        ]
        with open(table_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value_ms"])
            writer.writerows(rows)

    def _print_table(self, s: dict, n: int):
        header = f"{'Metric':<30} {'Value (ms)':>12}"
        sep = "-" * 44
        print(f"\n{'='*44}")
        print(f"  Evaluation Timing Summary  ({n} batches, {self._WARMUP_BATCHES} warmup skipped)")
        print(f"{'='*44}")
        print(header)
        print(sep)
        rows = [
            ("dataloader mean",    s["dataloader_mean_ms"]),
            ("dataloader std",     s["dataloader_std_ms"]),
            ("dataloader median",  s["dataloader_median_ms"]),
            ("dataloader IQR",     s["dataloader_iqr_ms"]),
            ("dataloader min",     s["dataloader_min_ms"]),
            ("dataloader max",     s["dataloader_max_ms"]),
            ("compute mean",       s["compute_mean_ms"]),
            ("compute std",        s["compute_std_ms"]),
            ("compute median",     s["compute_median_ms"]),
            ("compute IQR",        s["compute_iqr_ms"]),
            ("compute min",        s["compute_min_ms"]),
            ("compute max",        s["compute_max_ms"]),
        ]
        for label, val in rows:
            print(f"  {label:<28} {val:>12.3f}")
        print(f"{'='*44}\n")
