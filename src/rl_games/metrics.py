"""Training metrics: in-memory history plus a CSV log next to the save file.

Contents:
  - EpisodeRecord : one row per finished training episode
  - TrainingLog   : collects records, reports moving averages, writes CSV
  - moving_average(): simple trailing mean used by the CLI and plots
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    steps: int  # environment steps in this episode
    total_steps: int  # cumulative environment steps
    reward: float
    epsilon: float
    loss: float  # mean training loss in the episode (nan if no update)


class TrainingLog:
    """Append-only record of a training run."""

    def __init__(self) -> None:
        self.records: list[EpisodeRecord] = []

    def add(self, record: EpisodeRecord) -> None:
        self.records.append(record)

    @property
    def rewards(self) -> list[float]:
        return [r.reward for r in self.records]

    def mean_reward(self, last: int) -> float:
        if not self.records:
            return float("nan")
        return float(np.mean(self.rewards[-last:]))

    def to_csv(self, path: Path, *, append: bool = True) -> None:
        """Write (or append) the records; a header is written for new files."""
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or not append
        with path.open("a" if append else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(EpisodeRecord)])
            if new_file:
                writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))

    @staticmethod
    def read_csv(path: Path) -> list[EpisodeRecord]:
        with path.open(newline="") as f:
            return [
                EpisodeRecord(
                    episode=int(row["episode"]),
                    steps=int(row["steps"]),
                    total_steps=int(row["total_steps"]),
                    reward=float(row["reward"]),
                    epsilon=float(row["epsilon"]),
                    loss=float(row["loss"]),
                )
                for row in csv.DictReader(f)
            ]


def moving_average(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing mean; the first points average over what is available."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    idx = np.arange(1, arr.size + 1)
    start = np.maximum(0, idx - window)
    return (csum[idx] - csum[start]) / (idx - start)
