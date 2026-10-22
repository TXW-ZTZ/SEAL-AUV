"""Create publication-style training-reward comparisons for SEAL, SAC, and SAC-E.

The latest available runs use one seed per method.  Consequently, the shaded
region is the standard deviation of the per-episode return within a trailing
rolling window; it is not a cross-seed confidence interval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from config import PathConfig


PALETTE = {
  "seal": "#0F4D92",  # Proposed method: publication palette anchor blue.
  "sac": "#B64342",
  "sace": "#42949E",
}

TRAJECTORIES = {
  "circle": "Circle",
  "figure8": "Lemniscate",
  "sinusoid": "Sine",
}


@dataclass(frozen=True)
class RunSpec:
  label: str
  train_tag: str
  seed: int
  color: str


RUNS = (
  RunSpec("SAC", "single_traj_curriculum_sac", 666, PALETTE["sac"]),
  RunSpec("SAC-E", "single_traj_curriculum_sacr_weak", 1, PALETTE["sace"]),
  RunSpec("SEAL", "csac_sinusoid_no_current_ep2200_eval100_seed666", 666, PALETTE["seal"]),
)

MAX_EPISODE = 2000
ROLLING_WINDOW = 50
OUTPUT_SUBDIR = "algorithm_compare/training_returns_publication/seal_sac_sace_latest_ep2000"


def apply_publication_style() -> None:
  """Apply the compact line-figure style used for paper-ready comparisons."""
  plt.rcParams.update(
    {
      "font.family": "serif",
      "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
      "font.size": 14,
      "axes.labelsize": 15,
      "axes.titlesize": 16,
      "xtick.labelsize": 13,
      "ytick.labelsize": 13,
      "legend.fontsize": 12,
      "axes.linewidth": 1.4,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "legend.frameon": False,
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
      "svg.fonttype": "none",
      "axes.unicode_minus": False,
    }
  )


def log_path(paths: PathConfig, run: RunSpec, trajectory: str) -> Path:
  return (
    paths.log_dir
    / run.train_tag
    / f"{trajectory}_seed{run.seed}"
    / f"episode_costs_seed{run.seed}.jsonl"
  )


def load_episode_returns(path: Path, max_episode: int) -> tuple[np.ndarray, np.ndarray]:
  if not path.exists():
    raise FileNotFoundError(f"Training log does not exist: {path}")

  episodes: list[int] = []
  returns: list[float] = []
  with path.open("r", encoding="utf-8") as handle:
    for row_index, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue
      record = json.loads(line)
      episode = int(record.get("episode", row_index))
      if episode > max_episode:
        continue
      episodes.append(episode)
      returns.append(float(record["episode_return"]))

  if not episodes:
    raise ValueError(f"No episode-return records were found in {path}")
  return np.asarray(episodes, dtype=np.int32), np.asarray(returns, dtype=np.float64)


def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
  """Trailing rolling mean and population standard deviation, including the start."""
  if values.ndim != 1 or values.size == 0:
    raise ValueError("values must be a non-empty one-dimensional array")
  if window < 1:
    raise ValueError("window must be positive")

  count = np.minimum(np.arange(1, values.size + 1), window).astype(np.float64)
  starts = np.maximum(np.arange(values.size) - window + 1, 0)
  cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
  cumulative_sq = np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64)))
  sums = cumulative[1:] - cumulative[starts]
  sums_sq = cumulative_sq[1:] - cumulative_sq[starts]
  mean = sums / count
  variance = np.maximum(sums_sq / count - mean * mean, 0.0)
  return mean, np.sqrt(variance)


def plot_trajectory(paths: PathConfig, trajectory: str, output_dir: Path) -> list[Path]:
  title = TRAJECTORIES[trajectory]
  fig, ax = plt.subplots(figsize=(7.2, 4.8))
  lower_bounds: list[np.ndarray] = []
  upper_bounds: list[np.ndarray] = []

  for run_index, run in enumerate(RUNS):
    source = log_path(paths, run, trajectory)
    episodes, returns = load_episode_returns(source, MAX_EPISODE)
    mean, std = rolling_mean_std(returns, ROLLING_WINDOW)
    lower_bounds.append(mean - std)
    upper_bounds.append(mean + std)

    ax.fill_between(
      episodes,
      mean - std,
      mean + std,
      color=run.color,
      alpha=0.16,
      linewidth=0,
      zorder=1,
    )
    ax.plot(
      episodes,
      mean,
      color=run.color,
      linewidth=2.4,
      label=run.label,
      zorder=2 + run_index,
    )
    print(f"[{title}] {run.label}: {source}")

  all_lower = np.concatenate(lower_bounds)
  all_upper = np.concatenate(upper_bounds)
  data_min = float(np.min(all_lower))
  data_max = float(np.max(all_upper))
  data_range = max(data_max - data_min, 1.0)
  ax.set_ylim(data_min - 0.06 * data_range, data_max + 0.06 * data_range)
  ax.set_xlim(1, MAX_EPISODE)
  ax.set_xlabel("Training Episode")
  ax.set_ylabel("Episode Return")
  ax.set_title(f"Training Reward: {title}", pad=10)
  ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
  ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
  ax.grid(axis="y", color="#CFCECE", linewidth=0.65, alpha=0.8)
  ax.tick_params(direction="out", width=1.2, length=4.5)
  ax.legend(loc="upper left", handlelength=2.6)
  fig.tight_layout(pad=0.8)

  output_dir.mkdir(parents=True, exist_ok=True)
  base_path = output_dir / f"training_reward_{title.lower()}_seal_sac_sace_ep0001-2000_stdshade"
  png_path = base_path.with_suffix(".png")
  pdf_path = base_path.with_suffix(".pdf")
  fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
  fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
  plt.close(fig)
  return [png_path, pdf_path]


def main() -> None:
  apply_publication_style()
  paths = PathConfig()
  output_dir = paths.figure_dir / OUTPUT_SUBDIR

  saved: list[Path] = []
  for trajectory in TRAJECTORIES:
    saved.extend(plot_trajectory(paths, trajectory, output_dir))

  print("Saved publication figures:")
  for path in saved:
    print(path)


if __name__ == "__main__":
  main()
