"""
Plot episode-return curves during training for CSAC / SAC / SACr.

Data source:
  logs/<train_tag>/<trajectory>_seed<seed>/episode_costs_seed<seed>.jsonl

The plotted field is ``episode_return``. By default all algorithms share
``--seed``; use per-algorithm seed/tag options when comparing newer runs that
were trained with different seeds or tags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import AlgorithmKind, PathConfig, TrajectoryKind

TRAIN_TAG_BY_ALGO: dict[AlgorithmKind, str] = {
  "csac": "single_traj_curriculum",
  "sac": "single_traj_curriculum_sac",
  "sacr": "single_traj_curriculum_sacr",
}

ALGO_LABELS: dict[AlgorithmKind, str] = {
  "csac": "CSAC",
  "sac": "SAC",
  "sacr": "SACr",
}

ALGO_COLORS: dict[AlgorithmKind, str] = {
  "csac": "#1f77b4",
  "sac": "#ff7f0e",
  "sacr": "#2ca02c",
}

COMPARE_FIGURE_SUBDIR = "algorithm_compare"
TRAINING_RETURNS_SUBDIR = "training_returns"

SAC_BASELINE_TAGS = frozenset(
  {
    "sac",
    "sacr",
    "single_traj_curriculum_sac",
    "single_traj_curriculum_sacr",
    "single_traj_curriculum_sacr_weak",
  }
)


def _stage_log_dir(paths: PathConfig, train_tag: str, trajectory: TrajectoryKind, seed: int) -> Path:
  return paths.log_dir / train_tag / f"{trajectory}_seed{seed}"


def _is_csac_log_dir(log_dir: Path) -> bool:
  train_tag = log_dir.parent.name.lower()
  if train_tag in SAC_BASELINE_TAGS:
    return False
  if train_tag.endswith("_sac") or train_tag.endswith("_sacr") or "_sacr" in train_tag:
    return False
  return True


def resolve_latest_csac_log_dir(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  *,
  csac_train_tag: str | None = None,
) -> Path:
  if csac_train_tag:
    log_dir = _stage_log_dir(paths, csac_train_tag, trajectory, seed)
    log_path = log_dir / f"episode_costs_seed{seed}.jsonl"
    if not log_path.exists():
      raise FileNotFoundError(f"Missing CSAC log with --csac-train-tag={csac_train_tag}: {log_path}")
    return log_dir

  stage_name = f"{trajectory}_seed{seed}"
  candidates: list[tuple[float, Path]] = []
  for log_path in paths.log_dir.rglob(f"{stage_name}/episode_costs_seed{seed}.jsonl"):
    log_dir = log_path.parent
    if not _is_csac_log_dir(log_dir):
      continue
    candidates.append((log_path.stat().st_mtime, log_dir))

  if not candidates:
    fallback = _stage_log_dir(paths, TRAIN_TAG_BY_ALGO["csac"], trajectory, seed)
    log_path = fallback / f"episode_costs_seed{seed}.jsonl"
    if log_path.exists():
      return fallback
    raise FileNotFoundError(
      f"No CSAC training log found for {trajectory} / seed {seed} under {paths.log_dir}"
    )

  candidates.sort(key=lambda item: item[0], reverse=True)
  return candidates[0][1]


def resolve_log_dir(
  paths: PathConfig,
  algorithm: AlgorithmKind,
  trajectory: TrajectoryKind,
  seed: int,
  *,
  train_tag: str | None = None,
) -> Path:
  if algorithm == "csac":
    return resolve_latest_csac_log_dir(paths, trajectory, seed, csac_train_tag=train_tag)
  selected_train_tag = train_tag or TRAIN_TAG_BY_ALGO[algorithm]
  return _stage_log_dir(paths, selected_train_tag, trajectory, seed)


def load_training_return_curve(
  paths: PathConfig,
  algorithm: AlgorithmKind,
  trajectory: TrajectoryKind,
  seed: int,
  *,
  train_tag: str | None = None,
  max_episode: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Path]:
  log_dir = resolve_log_dir(paths, algorithm, trajectory, seed, train_tag=train_tag)
  log_path = log_dir / f"episode_costs_seed{seed}.jsonl"
  if not log_path.exists():
    raise FileNotFoundError(
      f"Missing training log for {ALGO_LABELS[algorithm]} / {trajectory} / seed {seed}: {log_path}"
    )

  episodes: list[int] = []
  returns: list[float] = []
  with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      episode = int(rec.get("episode", i + 1))
      if max_episode is not None and episode > max_episode:
        continue
      episodes.append(episode)
      returns.append(float(rec["episode_return"]))

  if not episodes:
    raise ValueError(f"No training records in {log_path}")
  return np.asarray(episodes, dtype=np.int32), np.asarray(returns, dtype=np.float64), log_path


def _moving_average(episodes: np.ndarray, values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
  if window <= 1 or len(values) < window:
    return episodes, values
  kernel = np.ones(window, dtype=np.float64) / window
  smooth = np.convolve(values, kernel, mode="valid")
  return episodes[window - 1 :], smooth


def _seed_label(seed_by_algo: dict[AlgorithmKind, int], algorithms: tuple[AlgorithmKind, ...]) -> str:
  seeds = [seed_by_algo[algorithm] for algorithm in algorithms]
  if len(set(seeds)) == 1:
    return f"seed{seeds[0]}"
  return "_".join(f"{algorithm}{seed_by_algo[algorithm]}" for algorithm in algorithms)


def _seed_title(seed_by_algo: dict[AlgorithmKind, int], algorithms: tuple[AlgorithmKind, ...]) -> str:
  seeds = [seed_by_algo[algorithm] for algorithm in algorithms]
  if len(set(seeds)) == 1:
    return f"seed {seeds[0]}"
  return ", ".join(f"{ALGO_LABELS[algorithm]} seed {seed_by_algo[algorithm]}" for algorithm in algorithms)


def compare_output_dir(paths: PathConfig, seed_label: str) -> Path:
  out = paths.figure_dir / COMPARE_FIGURE_SUBDIR / TRAINING_RETURNS_SUBDIR / seed_label
  out.mkdir(parents=True, exist_ok=True)
  return out


def plot_trajectory_compare(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  smooth_window: int = 50,
  show_raw: bool = True,
  csac_train_tag: str | None = None,
  sac_train_tag: str | None = None,
  sacr_train_tag: str | None = None,
  seed_by_algo: dict[AlgorithmKind, int] | None = None,
  hide_seed_in_title: bool = False,
  simple_legend: bool = False,
  max_episode: int | None = None,
  algorithms: tuple[AlgorithmKind, ...] = ("csac", "sac", "sacr"),
) -> Path:
  fig, ax = plt.subplots(figsize=(10.5, 4.8))
  used_logs: dict[str, str] = {}
  train_tag_by_algo: dict[AlgorithmKind, str | None] = {
    "csac": csac_train_tag,
    "sac": sac_train_tag,
    "sacr": sacr_train_tag,
  }
  if seed_by_algo is None:
    seed_by_algo = {algorithm: seed for algorithm in algorithms}
  mixed_seeds = len({seed_by_algo[algorithm] for algorithm in algorithms}) > 1

  for algorithm in algorithms:
    algo_seed = seed_by_algo[algorithm]
    episodes, returns, log_path = load_training_return_curve(
      paths,
      algorithm,
      trajectory,
      algo_seed,
      train_tag=train_tag_by_algo[algorithm],
      max_episode=max_episode,
    )
    used_logs[ALGO_LABELS[algorithm]] = str(log_path)
    if show_raw:
      ax.plot(episodes, returns, color=ALGO_COLORS[algorithm], alpha=0.18, linewidth=0.8)

    window = min(smooth_window, max(5, len(returns) // 20))
    smooth_x, smooth_y = _moving_average(episodes, returns, window)
    seed_part = f", seed={algo_seed}" if mixed_seeds else ""
    label = ALGO_LABELS[algorithm] if simple_legend else (
      f"{ALGO_LABELS[algorithm]} (avg {window}, n={len(returns)}{seed_part})"
    )
    ax.plot(
      smooth_x,
      smooth_y,
      color=ALGO_COLORS[algorithm],
      linewidth=2.0,
      label=label,
    )

  ax.set_xlabel("Training episode")
  ax.set_ylabel("Episode return (task reward)")
  title_suffix = f", first {max_episode} episodes" if max_episode is not None else ""
  if hide_seed_in_title:
    ax.set_title(f"Training return - {trajectory} (CSAC vs SAC vs SACr{title_suffix})")
  else:
    seed_title = _seed_title(seed_by_algo, algorithms)
    ax.set_title(f"Training return - {trajectory} ({seed_title}, CSAC vs SAC vs SACr{title_suffix})")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize=9)
  fig.tight_layout()

  output_seed_label = _seed_label(seed_by_algo, algorithms)
  out_dir = compare_output_dir(paths, output_seed_label)
  suffix = f"_ep0001-{max_episode:04d}" if max_episode is not None else ""
  if simple_legend:
    suffix = f"{suffix}_simplelegend"
  out_path = out_dir / f"compare_training_return_{trajectory}_{output_seed_label}_three_algo{suffix}.png"
  fig.savefig(out_path, dpi=150)
  plt.close(fig)

  print(f"  [{trajectory}] CSAC log: {used_logs['CSAC']}")
  print(f"  [{trajectory}] SAC  log: {used_logs['SAC']}")
  print(f"  [{trajectory}] SACr log: {used_logs['SACr']}")
  return out_path


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plot CSAC/SAC/SACr training episode-return curves."
  )
  parser.add_argument("--seed", type=int, default=666, help="Default training random seed.")
  parser.add_argument(
    "--trajectory",
    type=str,
    default="all",
    choices=["circle", "sinusoid", "figure8", "all"],
    help="Trajectory to plot; all writes one figure per trajectory.",
  )
  parser.add_argument("--smooth-window", type=int, default=50, help="Moving-average window in episodes.")
  parser.add_argument("--no-raw", action="store_true", help="Plot only the moving-average curves.")
  parser.add_argument("--csac-train-tag", type=str, default=TRAIN_TAG_BY_ALGO["csac"], help="CSAC log tag.")
  parser.add_argument("--sac-train-tag", type=str, default=TRAIN_TAG_BY_ALGO["sac"], help="SAC log tag.")
  parser.add_argument("--sacr-train-tag", type=str, default=TRAIN_TAG_BY_ALGO["sacr"], help="SACr log tag.")
  parser.add_argument("--csac-seed", type=int, default=None, help="CSAC seed; defaults to --seed.")
  parser.add_argument("--sac-seed", type=int, default=None, help="SAC seed; defaults to --seed.")
  parser.add_argument("--sacr-seed", type=int, default=None, help="SACr seed; defaults to --seed.")
  parser.add_argument("--hide-seed-in-title", action="store_true", help="Do not show seed values in figure titles.")
  parser.add_argument("--simple-legend", action="store_true", help="Only show algorithm names in the legend.")
  parser.add_argument(
    "--max-episode",
    type=int,
    default=0,
    help="Only plot records with episode <= this value; 0 disables truncation.",
  )
  return parser


def main(argv: list[str] | None = None) -> None:
  args = build_arg_parser().parse_args(argv)
  paths = PathConfig()
  paths.ensure()

  train_tags: dict[AlgorithmKind, str] = {
    "csac": args.csac_train_tag.strip() or TRAIN_TAG_BY_ALGO["csac"],
    "sac": args.sac_train_tag.strip() or TRAIN_TAG_BY_ALGO["sac"],
    "sacr": args.sacr_train_tag.strip() or TRAIN_TAG_BY_ALGO["sacr"],
  }
  seed_by_algo: dict[AlgorithmKind, int] = {
    "csac": args.csac_seed if args.csac_seed is not None else args.seed,
    "sac": args.sac_seed if args.sac_seed is not None else args.seed,
    "sacr": args.sacr_seed if args.sacr_seed is not None else args.seed,
  }
  max_episode = int(args.max_episode) if int(args.max_episode) > 0 else None

  trajectories: list[TrajectoryKind] = (
    ["circle", "figure8", "sinusoid"] if args.trajectory == "all" else [args.trajectory]  # type: ignore[list-item]
  )

  for traj in trajectories:
    out_path = plot_trajectory_compare(
      paths,
      traj,
      args.seed,
      smooth_window=args.smooth_window,
      show_raw=not args.no_raw,
      csac_train_tag=train_tags["csac"],
      sac_train_tag=train_tags["sac"],
      sacr_train_tag=train_tags["sacr"],
      seed_by_algo=seed_by_algo,
      hide_seed_in_title=args.hide_seed_in_title,
      simple_legend=args.simple_legend,
      max_episode=max_episode,
    )
    print(f"Saved: {out_path}")

  output_seed_label = _seed_label(seed_by_algo, ("csac", "sac", "sacr"))
  print(f"\nAll compare figures under: {compare_output_dir(paths, output_seed_label)}")


if __name__ == "__main__":
  main()
