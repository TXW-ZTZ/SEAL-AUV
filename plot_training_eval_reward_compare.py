"""
Plot eval reward curves collected during training for CSAC / SAC / SACr.

Data source:
  logs/<train_tag>/<trajectory>_seed<seed>/eval_during_train_seed<seed>.jsonl

The plotted field is ``reward_mean`` from the requested trajectory entry when
available, otherwise ``overall.reward_mean``.
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
TRAINING_EVAL_REWARDS_SUBDIR = "training_eval_rewards"
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
    log_path = log_dir / f"eval_during_train_seed{seed}.jsonl"
    if not log_path.exists():
      raise FileNotFoundError(f"Missing CSAC eval log with --csac-train-tag={csac_train_tag}: {log_path}")
    return log_dir

  stage_name = f"{trajectory}_seed{seed}"
  candidates: list[tuple[float, Path]] = []
  for log_path in paths.log_dir.rglob(f"{stage_name}/eval_during_train_seed{seed}.jsonl"):
    log_dir = log_path.parent
    if not _is_csac_log_dir(log_dir):
      continue
    candidates.append((log_path.stat().st_mtime, log_dir))

  if not candidates:
    fallback = _stage_log_dir(paths, TRAIN_TAG_BY_ALGO["csac"], trajectory, seed)
    log_path = fallback / f"eval_during_train_seed{seed}.jsonl"
    if log_path.exists():
      return fallback
    raise FileNotFoundError(
      f"No CSAC eval-during-train log found for {trajectory} / seed {seed} under {paths.log_dir}"
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


def load_eval_reward_curve(
  paths: PathConfig,
  algorithm: AlgorithmKind,
  trajectory: TrajectoryKind,
  seed: int,
  *,
  train_tag: str | None = None,
  max_episode: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Path]:
  log_dir = resolve_log_dir(paths, algorithm, trajectory, seed, train_tag=train_tag)
  log_path = log_dir / f"eval_during_train_seed{seed}.jsonl"
  if not log_path.exists():
    raise FileNotFoundError(
      f"Missing eval-during-train log for {ALGO_LABELS[algorithm]} / {trajectory} / seed {seed}: {log_path}"
    )

  episodes: list[int] = []
  rewards: list[float] = []
  with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      episode = int(rec["episode"])
      if max_episode is not None and episode > max_episode:
        continue
      by_traj = rec.get("by_trajectory", {})
      if trajectory in by_traj:
        reward = float(by_traj[trajectory]["reward_mean"])
      else:
        reward = float(rec["overall"]["reward_mean"])
      episodes.append(episode)
      rewards.append(reward)

  if not episodes:
    raise ValueError(f"No eval records in {log_path}")
  return np.asarray(episodes, dtype=np.int32), np.asarray(rewards, dtype=np.float64), log_path


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
  out = paths.figure_dir / COMPARE_FIGURE_SUBDIR / TRAINING_EVAL_REWARDS_SUBDIR / seed_label
  out.mkdir(parents=True, exist_ok=True)
  return out


def plot_trajectory_compare(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  *,
  csac_train_tag: str | None = None,
  sac_train_tag: str | None = None,
  sacr_train_tag: str | None = None,
  seed_by_algo: dict[AlgorithmKind, int] | None = None,
  hide_seed_in_title: bool = False,
  simple_legend: bool = False,
  fill_sac_start_from_csac: bool = False,
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
  csac_start: tuple[int, float] | None = None

  for algorithm in algorithms:
    algo_seed = seed_by_algo[algorithm]
    episodes, rewards, log_path = load_eval_reward_curve(
      paths,
      algorithm,
      trajectory,
      algo_seed,
      train_tag=train_tag_by_algo[algorithm],
      max_episode=max_episode,
    )
    used_logs[ALGO_LABELS[algorithm]] = str(log_path)
    if algorithm == "csac" and len(episodes) > 0:
      csac_start = (int(episodes[0]), float(rewards[0]))
    if (
      fill_sac_start_from_csac
      and algorithm == "sac"
      and csac_start is not None
      and len(episodes) > 0
      and int(episodes[0]) > csac_start[0]
    ):
      episodes = np.insert(episodes, 0, csac_start[0])
      rewards = np.insert(rewards, 0, csac_start[1])
    seed_part = f", seed={algo_seed}" if mixed_seeds else ""
    label = ALGO_LABELS[algorithm] if simple_legend else f"{ALGO_LABELS[algorithm]} (n={len(rewards)}{seed_part})"
    ax.plot(
      episodes,
      rewards,
      marker="o",
      linewidth=1.8,
      markersize=4.5,
      color=ALGO_COLORS[algorithm],
      label=label,
    )

  ax.set_xlabel("Training episode")
  ax.set_ylabel("Eval reward (mean)")
  title_suffix = f", first {max_episode} episodes" if max_episode is not None else ""
  if hide_seed_in_title:
    ax.set_title(f"Eval reward during training - {trajectory} (CSAC vs SAC vs SACr{title_suffix})")
  else:
    seed_title = _seed_title(seed_by_algo, algorithms)
    ax.set_title(f"Eval reward during training - {trajectory} ({seed_title}, CSAC vs SAC vs SACr{title_suffix})")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize=9)
  fig.tight_layout()

  output_seed_label = _seed_label(seed_by_algo, algorithms)
  out_dir = compare_output_dir(paths, output_seed_label)
  suffix = f"_ep0001-{max_episode:04d}" if max_episode is not None else ""
  if simple_legend:
    suffix = f"{suffix}_simplelegend"
  if fill_sac_start_from_csac:
    suffix = f"{suffix}_sacstartfilled"
  out_path = out_dir / f"compare_training_eval_reward_{trajectory}_{output_seed_label}_three_algo{suffix}.png"
  fig.savefig(out_path, dpi=150)
  plt.close(fig)

  print(f"  [{trajectory}] CSAC eval log: {used_logs['CSAC']}")
  print(f"  [{trajectory}] SAC  eval log: {used_logs['SAC']}")
  print(f"  [{trajectory}] SACr eval log: {used_logs['SACr']}")
  return out_path


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plot CSAC/SAC/SACr eval reward curves collected during training."
  )
  parser.add_argument("--seed", type=int, default=666, help="Default training random seed.")
  parser.add_argument(
    "--trajectory",
    type=str,
    default="all",
    choices=["circle", "sinusoid", "figure8", "all"],
    help="Trajectory to plot; all writes one figure per trajectory.",
  )
  parser.add_argument("--csac-train-tag", type=str, default=TRAIN_TAG_BY_ALGO["csac"], help="CSAC log tag.")
  parser.add_argument("--sac-train-tag", type=str, default=TRAIN_TAG_BY_ALGO["sac"], help="SAC log tag.")
  parser.add_argument("--sacr-train-tag", type=str, default=TRAIN_TAG_BY_ALGO["sacr"], help="SACr log tag.")
  parser.add_argument("--csac-seed", type=int, default=None, help="CSAC seed; defaults to --seed.")
  parser.add_argument("--sac-seed", type=int, default=None, help="SAC seed; defaults to --seed.")
  parser.add_argument("--sacr-seed", type=int, default=None, help="SACr seed; defaults to --seed.")
  parser.add_argument("--hide-seed-in-title", action="store_true", help="Do not show seed values in figure titles.")
  parser.add_argument("--simple-legend", action="store_true", help="Only show algorithm names in the legend.")
  parser.add_argument(
    "--fill-sac-start-from-csac",
    action="store_true",
    help="If SAC starts later than CSAC, prepend a display-only SAC point using CSAC's first eval value.",
  )
  parser.add_argument(
    "--max-episode",
    type=int,
    default=0,
    help="Only plot eval records with episode <= this value; 0 disables truncation.",
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
      csac_train_tag=train_tags["csac"],
      sac_train_tag=train_tags["sac"],
      sacr_train_tag=train_tags["sacr"],
      seed_by_algo=seed_by_algo,
      hide_seed_in_title=args.hide_seed_in_title,
      simple_legend=args.simple_legend,
      fill_sac_start_from_csac=args.fill_sac_start_from_csac,
      max_episode=max_episode,
    )
    print(f"Saved: {out_path}")

  output_seed_label = _seed_label(seed_by_algo, ("csac", "sac", "sacr"))
  print(f"\nAll compare figures under: {compare_output_dir(paths, output_seed_label)}")


if __name__ == "__main__":
  main()
