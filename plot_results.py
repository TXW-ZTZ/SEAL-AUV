"""
训练/评估结果独立绘图。

用法:
  python plot_results.py --trajectory circle --seed 1
  python plot_results.py --trajectory all --seed 1
  python plot_sac_results.py --trajectory all --seed 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig, TrajectoryKind, CSACConfig, SimConfig, AlgorithmKind
from trajectory import make_trajectory

TRAIN_TAG_DEFAULT = "single_traj_curriculum"
TRAIN_TAG_DEFAULTS: dict[AlgorithmKind, str] = {
  "csac": TRAIN_TAG_DEFAULT,
  "sac": "single_traj_curriculum_sac",
  "sacr": "single_traj_curriculum_sacr",
}


def _set_equal_3d_aspect(ax, arrays: list[np.ndarray]):
  """Use equal 3D scale for faithful geometry perception."""
  valid = [np.asarray(a, dtype=np.float64) for a in arrays if np.asarray(a).size > 0]
  if not valid:
    return
  pts = np.vstack(valid)
  xyz_min = np.min(pts, axis=0)
  xyz_max = np.max(pts, axis=0)
  center = 0.5 * (xyz_min + xyz_max)
  spans = np.maximum(xyz_max - xyz_min, 1e-6)
  half = 0.5 * float(np.max(spans))
  ax.set_xlim(center[0] - half, center[0] + half)
  ax.set_ylim(center[1] - half, center[1] + half)
  ax.set_zlim(center[2] - half, center[2] + half)
  if hasattr(ax, "set_box_aspect"):
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _train_dirs(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  *,
  train_tag: str,
  current_label: str | None,
  curriculum: bool,
) -> tuple[Path, Path]:
  if train_tag:
    ckpt_dir = paths.checkpoint_dir / train_tag
    log_dir = paths.log_dir / train_tag
  else:
    ckpt_dir = paths.traj_checkpoint_dir(trajectory)
    log_dir = paths.log_dir / trajectory
  if curriculum:
    ckpt_dir = ckpt_dir / "curriculum"
    log_dir = log_dir / "curriculum"
  elif current_label:
    ckpt_dir = ckpt_dir / current_label
    log_dir = log_dir / current_label
  return ckpt_dir, log_dir


def _load_eval_rollouts(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  algorithm: AlgorithmKind = "csac",
) -> list[dict]:
  p = paths.traj_eval_log_dir(trajectory, algorithm) / f"eval_rollouts_seed{seed}.json"
  if not p.exists():
    eval_scripts = {
      "sac": "evaluate_sac.py",
      "sacr": "evaluate_sacr.py",
      "csac": "evaluate.py",
    }
    eval_script = eval_scripts.get(algorithm, "evaluate.py")
    raise FileNotFoundError(
      f"Missing {p}. Run: python {eval_script} --trajectory {trajectory} --seed {seed}"
    )
  return json.loads(p.read_text(encoding="utf-8"))


def _load_cross_eval_rollouts(
  paths: PathConfig,
  test_trajectory: TrajectoryKind,
  train_trajectory: TrajectoryKind,
  seed: int,
) -> list[dict]:
  p = paths.cross_eval_log_dir(train_trajectory, seed) / test_trajectory / "eval_rollouts.json"
  if not p.exists():
    raise FileNotFoundError(
      f"Missing {p}. Run: python test_cross_trajectory.py --seed {seed}"
    )
  return json.loads(p.read_text(encoding="utf-8"))


def _avoidance_cost(min_surface: float) -> float:
  """Lower is better; penalize surface penetration and near-misses."""
  if not np.isfinite(min_surface):
    return 100.0
  if min_surface < 0.0:
    return 10.0 + abs(min_surface) * 5.0
  return 1.0 / (min_surface + 0.2)


def _episode_rank_key(r: dict, episode_steps: int) -> tuple:
  """Lexicographic rank for eval episode selection (lower is better)."""
  success = bool(r.get("success", False))
  collision = bool(r.get("collision", False))
  reason = str(r.get("reason", ""))
  steps = int(r.get("steps", 0))
  clean_full = steps >= episode_steps and not collision and reason != "deviate"

  rmse = float(r.get("rmse_3d", np.inf))
  if not np.isfinite(rmse):
    rmse = np.inf
  min_surface = float(r.get("min_obstacle_surface", -np.inf))
  cum_err = float(r.get("cumulative_error", np.inf))
  if not np.isfinite(cum_err):
    cum_err = np.inf
  warnings = int(r.get("collision_warning_count", 0))
  d_end = float(r.get("d_endpoint", np.inf))
  if not np.isfinite(d_end):
    d_end = np.inf

  combined = 0.5 * rmse + 0.5 * _avoidance_cost(min_surface)
  return (
    0 if success else 1,
    1 if collision else 0,
    0 if clean_full else 1,
    combined,
    warnings,
    cum_err,
    d_end,
  )


def _pick_best_episode_indexed(rollouts: list[dict]) -> tuple[int, dict]:
  if not rollouts:
    raise ValueError("rollouts is empty")
  episode_steps = SimConfig().episode_steps
  best_idx = min(range(len(rollouts)), key=lambda i: _episode_rank_key(rollouts[i], episode_steps))
  return best_idx, rollouts[best_idx]


def _pick_best_episode(rollouts: list[dict]) -> dict:
  _, best = _pick_best_episode_indexed(rollouts)
  return best


def _format_best_episode_caption(ep: dict) -> str:
  success = bool(ep.get("success", False))
  collision = bool(ep.get("collision", False))
  rmse = float(ep.get("rmse_3d", np.nan))
  min_surface = float(ep.get("min_obstacle_surface", np.nan))
  warnings = int(ep.get("collision_warning_count", 0))
  return (
    f"RMSE={rmse:.3f} m, min clearance={min_surface:.3f} m, "
    f"success={success}, collision={collision}, warnings={warnings}"
  )


def _log_best_episode_selection(trajectory: str, rollouts: list[dict]) -> dict:
  idx, best = _pick_best_episode_indexed(rollouts)
  print(
    f"[{trajectory}] plot episode #{idx + 1}/{len(rollouts)} — "
    f"{_format_best_episode_caption(best)}"
  )
  return best


def plot_3d_trajectory(ep: dict, out_dir: Path, trajectory: TrajectoryKind):
  pos = np.asarray(ep["positions"])
  ref = np.asarray(ep["ref_positions"])
  obs = np.asarray(ep["obstacle_centers"])
  r = ep["obstacle_radius"]

  fig = plt.figure(figsize=(9, 7))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "r--", linewidth=1.5, label="reference")
  ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=1.2, label="AUV")
  if len(ref) > 0:
    ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color="#2ca02c", s=36, marker="o", label="ref start")
    ax.scatter(ref[-1, 0], ref[-1, 1], ref[-1, 2], color="#d62728", s=36, marker="o", label="ref end")
  if len(pos) > 0:
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="#17becf", s=30, marker="^", label="AUV start")
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="#9467bd", s=30, marker="^", label="AUV end")
  for c in obs:
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 12)
    xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="green", alpha=0.25, linewidth=0)
  ax.set_xlabel("N (m)")
  ax.set_ylabel("E (m)")
  ax.set_zlabel("D (m)")
  _set_equal_3d_aspect(ax, [ref, pos, obs])
  ax.set_title(f"3D tracking — {trajectory} ({_format_best_episode_caption(ep)})")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "01_trajectory_3d.png", dpi=150)
  plt.close(fig)


def plot_tracking_errors(ep: dict, out_dir: Path, trajectory: TrajectoryKind):
  errors = np.asarray(ep["errors_xyz"])
  t = np.asarray(ep["times"])
  labels = ["x", "y", "z"]
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  for i, ax in enumerate(axes):
    ax.plot(t, errors[:, i], linewidth=1.0)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_ylabel(f"e_{labels[i]} (m)")
    ax.grid(True, alpha=0.3)
  axes[-1].set_xlabel("time (s)")
  fig.suptitle(f"Tracking errors — {trajectory}")
  fig.tight_layout()
  fig.savefig(out_dir / "02_tracking_errors_xyz.png", dpi=150)
  plt.close(fig)


def plot_rmse_summary(all_summaries: dict, out_dir: Path):
  names = list(all_summaries.keys())
  rmse_mean = [all_summaries[n]["rmse_3d_mean"] for n in names]
  rmse_std = [all_summaries[n]["rmse_3d_std"] for n in names]

  fig, ax = plt.subplots(figsize=(8, 5))
  x = np.arange(len(names))
  ax.bar(x, rmse_mean, yerr=rmse_std, capsize=4, color=["#4C72B0", "#55A868", "#C44E52"][: len(names)])
  ax.set_xticks(x)
  ax.set_xticklabels(names)
  ax.set_ylabel("RMSE (m)")
  ax.set_title("Overall 3D tracking RMSE (eval batch)")
  ax.grid(True, axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / "03_rmse_comparison.png", dpi=150)
  plt.close(fig)


def plot_actions(ep: dict, out_dir: Path, trajectory: TrajectoryKind):
  acts = np.asarray(ep["actions_phys"])
  t = np.asarray(ep["times"])
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  axes[0].plot(t, acts[:, 0], color="#1f77b4")
  axes[0].set_ylabel("Thrust (N)")
  axes[1].plot(t, np.rad2deg(acts[:, 1]), color="#ff7f0e")
  axes[1].set_ylabel("Vertical rudder (deg)")
  axes[2].plot(t, np.rad2deg(acts[:, 2]), color="#2ca02c")
  axes[2].set_ylabel("Horizontal rudder (deg)")
  axes[2].set_xlabel("time (s)")
  for ax in axes:
    ax.grid(True, alpha=0.3)
  fig.suptitle(f"Actuator outputs — {trajectory}")
  fig.tight_layout()
  fig.savefig(out_dir / "04_actuators.png", dpi=150)
  plt.close(fig)


def plot_energy_comparison(all_summaries: dict, rollouts_by_traj: dict, out_dir: Path):
  names = list(all_summaries.keys())
  energy_mean = [all_summaries[n]["energy_mean"] for n in names]
  energy_std = [all_summaries[n]["energy_std"] for n in names]

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  x = np.arange(len(names))
  axes[0].bar(x, energy_mean, yerr=energy_std, capsize=4, color=["#8172B2", "#CCB974", "#64B5CD"][: len(names)])
  axes[0].set_xticks(x)
  axes[0].set_xticklabels(names)
  axes[0].set_ylabel("Episode energy (undiscounted Σc)")
  axes[0].set_title("Mean energy consumption")
  axes[0].grid(True, axis="y", alpha=0.3)

  for i, name in enumerate(names):
    rollouts = rollouts_by_traj[name]
    best = _pick_best_episode(rollouts)
    costs = np.cumsum(best["step_costs"])
    axes[1].plot(best["times"], costs, label=name)
  axes[1].set_xlabel("time (s)")
  axes[1].set_ylabel("Cumulative energy")
  axes[1].set_title("Best-episode cumulative energy")
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig(out_dir / "05_energy_comparison.png", dpi=150)
  plt.close(fig)


def plot_success_collision_rates(all_summaries: dict, out_dir: Path):
  names = list(all_summaries.keys())
  success = [all_summaries[n]["success_rate"] * 100 for n in names]
  collision = [all_summaries[n]["collision_rate"] * 100 for n in names]

  fig, ax = plt.subplots(figsize=(9, 5))
  x = np.arange(len(names))
  w = 0.35
  ax.bar(x - w / 2, success, width=w, label="Success rate (%)", color="#2ca02c")
  ax.bar(x + w / 2, collision, width=w, label="Collision rate (%)", color="#d62728")
  ax.set_xticks(x)
  ax.set_xticklabels(names)
  ax.set_ylim(0, 105)
  ax.set_title("Task success vs obstacle collision")
  ax.legend()
  ax.grid(True, axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / "06_success_collision_rates.png", dpi=150)
  plt.close(fig)


def plot_training_scores(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  out_dir: Path,
  train_tag: str = TRAIN_TAG_DEFAULT,
  current_label: str | None = None,
  curriculum: bool = False,
  algorithm: AlgorithmKind = "csac",
):
  import pickle

  ckpt_dir, _ = _train_dirs(
    paths,
    trajectory,
    train_tag=train_tag,
    current_label=current_label,
    curriculum=curriculum,
  )
  scores_tags = {"sac": "SAC", "sacr": "SACr", "csac": "CSAC"}
  scores_tag = scores_tags.get(algorithm, "CSAC")
  p = ckpt_dir / f"scores_{scores_tag}_{seed}.pkl"
  if not p.exists():
    return
  scores = pickle.load(open(p, "rb"))
  if len(scores) < 10:
    return
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(scores, alpha=0.4, label="episode return")
  window = min(50, len(scores) // 5 + 1)
  if window > 1:
    kernel = np.ones(window) / window
    smooth = np.convolve(scores, kernel, mode="valid")
    ax.plot(np.arange(window - 1, len(scores)), smooth, linewidth=2, label=f"moving avg ({window})")
  ax.set_xlabel("episode")
  ax.set_ylabel("return")
  ax.set_title(f"Training returns — {trajectory} seed {seed} ({algorithm.upper()})")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / "07_training_returns.png", dpi=150)
  plt.close(fig)


def plot_training_cmdp_metrics(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  out_dir: Path,
  train_tag: str = TRAIN_TAG_DEFAULT,
  current_label: str | None = None,
  curriculum: bool = False,
  algorithm: AlgorithmKind = "csac",
):
  _, train_log_dir = _train_dirs(
    paths,
    trajectory,
    train_tag=train_tag,
    current_label=current_label,
    curriculum=curriculum,
  )
  p = train_log_dir / f"episode_costs_seed{seed}.jsonl"
  if not p.exists():
    return

  records: list[dict] = []
  with open(p, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      records.append(json.loads(line))
  if len(records) < 2:
    return

  episodes = np.array([r.get("episode", i + 1) for i, r in enumerate(records)], dtype=np.int32)
  collisions = np.array([1 if r.get("collision", False) else 0 for r in records], dtype=np.int32)
  warnings = np.array([r.get("collision_warning_count", 0) for r in records], dtype=np.float64)
  tracking_error = np.array([r.get("tracking_error_sum", np.nan) for r in records], dtype=np.float64)
  jc = np.array([r.get("discounted_Jc", np.nan) for r in records], dtype=np.float64)
  lam = np.array([r.get("lambda", np.nan) for r in records], dtype=np.float64)
  alpha = np.array([r.get("alpha", np.nan) for r in records], dtype=np.float64)
  theta = np.array([r.get("theta", np.nan) for r in records], dtype=np.float64)

  if "collision_count_last_100" in records[0]:
    coll_100 = np.array([r.get("collision_count_last_100", 0) for r in records], dtype=np.int32)
  else:
    # 兼容旧日志：由 collision 序列在线重建“近100回合碰撞次数”。
    coll_100 = np.zeros_like(collisions)
    run = 0
    for i in range(len(collisions)):
      run += int(collisions[i])
      if i >= 100:
        run -= int(collisions[i - 100])
      coll_100[i] = run

  fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
  ax0, ax1, ax2, ax3, ax4, ax5 = axes.ravel()

  ax0.plot(episodes, coll_100, color="#d62728", linewidth=1.5)
  ax0.set_ylabel("count / 100 episodes")
  ax0.set_title("Collisions in recent 100 episodes")
  ax0.grid(True, alpha=0.3)

  ax1.plot(episodes, tracking_error, color="#1f77b4", alpha=0.4, linewidth=1.0, label="per-episode")
  w = min(50, max(5, len(tracking_error) // 20))
  if w > 1 and len(tracking_error) >= w:
    k = np.ones(w, dtype=np.float64) / w
    tr_smooth = np.convolve(tracking_error, k, mode="valid")
    ax1.plot(episodes[w - 1 : w - 1 + len(tr_smooth)], tr_smooth, color="#1f77b4", linewidth=2.0, label=f"moving avg ({w})")
  ax1.set_ylabel("sum d_path")
  ax1.set_title("Per-episode cumulative tracking error")
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  ax2.plot(episodes, jc, color="#9467bd", linewidth=1.3, label="discounted Jc")
  if np.any(np.isfinite(theta)):
    ax2.plot(episodes, theta, "r--", linewidth=1.2, label="theta")
  ax2.set_xlabel("episode")
  ax2.set_ylabel("value")
  ax2.set_title("Jc evolution")
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  if algorithm in ("sac", "sacr"):
    ax3.plot(episodes, alpha, color="#2ca02c", linewidth=1.3)
    ax3.set_xlabel("episode")
    ax3.set_ylabel("alpha")
    ax3.set_title("Entropy coefficient (alpha)")
  else:
    ax3.plot(episodes, lam, color="#2ca02c", linewidth=1.3)
    ax3.set_xlabel("episode")
    ax3.set_ylabel("lambda")
    ax3.set_title("Lagrange multiplier evolution")
  ax3.grid(True, alpha=0.3)

  ax4.plot(episodes, warnings, color="#ff7f0e", alpha=0.5, linewidth=1.0, label="per-episode")
  w2 = min(50, max(5, len(warnings) // 20))
  if w2 > 1 and len(warnings) >= w2:
    k2 = np.ones(w2, dtype=np.float64) / w2
    w_smooth = np.convolve(warnings, k2, mode="valid")
    ax4.plot(episodes[w2 - 1 : w2 - 1 + len(w_smooth)], w_smooth, color="#ff7f0e", linewidth=2.0, label=f"moving avg ({w2})")
  ax4.set_xlabel("episode")
  ax4.set_ylabel("warning count")
  ax4.set_title("Per-episode warning count")
  ax4.legend()
  ax4.grid(True, alpha=0.3)

  clean = np.array([1.0 if r.get("clean_full_episode", False) else 0.0 for r in records], dtype=np.float64)
  clean_rolling = np.zeros_like(clean)
  run = 0.0
  for i in range(clean.size):
    run += clean[i]
    if i >= 100:
      run -= clean[i - 100]
    clean_rolling[i] = run
  ax5.plot(episodes, clean_rolling, color="#17becf", linewidth=1.5)
  ax5.set_xlabel("episode")
  ax5.set_ylabel("count / last 100")
  ax5.set_title("Clean full episodes in recent 100")
  ax5.grid(True, alpha=0.3)

  fig.suptitle(f"Training metrics — {trajectory} seed {seed} ({algorithm.upper()})")
  fig.tight_layout()
  fig.savefig(out_dir / "08_training_cmdp_metrics.png", dpi=150)
  plt.close(fig)


def plot_training_eval_rewards(
  paths: PathConfig,
  seed: int,
  out_dir: Path,
  train_tag: str = TRAIN_TAG_DEFAULT,
  current_label: str | None = None,
  curriculum: bool = False,
  anchor_trajectory: TrajectoryKind = "circle",
  algorithm: AlgorithmKind = "csac",
):
  _, train_log_dir = _train_dirs(
    paths,
    anchor_trajectory,
    train_tag=train_tag,
    current_label=current_label,
    curriculum=curriculum,
  )
  p = train_log_dir / f"eval_during_train_seed{seed}.jsonl"
  if not p.exists():
    return

  records: list[dict] = []
  with open(p, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      records.append(json.loads(line))
  if len(records) < 1:
    return

  episodes = np.array([r["episode"] for r in records], dtype=np.int32)
  overall_reward = np.array([r["overall"]["reward_mean"] for r in records], dtype=np.float64)
  traj_names = list(records[0].get("by_trajectory", {}).keys())

  fig, ax = plt.subplots(figsize=(10, 4.6))
  ax.plot(episodes, overall_reward, marker="o", linewidth=1.8, label="overall eval reward")
  for traj in traj_names:
    vals = np.array([r["by_trajectory"][traj]["reward_mean"] for r in records], dtype=np.float64)
    ax.plot(episodes, vals, marker=".", linewidth=1.2, alpha=0.85, label=f"{traj} eval reward")

  best_json = train_log_dir / f"best_eval_model_seed{seed}.json"
  if best_json.exists():
    best = json.loads(best_json.read_text(encoding="utf-8"))
    best_ep = int(best["episode"])
    best_reward = float(best["overall"]["reward_mean"])
    ax.scatter([best_ep], [best_reward], color="red", zorder=5, label=f"best checkpoint (ep {best_ep})")
    ax.axvline(best_ep, color="red", linestyle="--", linewidth=1.0, alpha=0.6)

  ax.set_xlabel("training episode")
  ax.set_ylabel("eval reward (mean)")
  ax.set_title(f"Evaluation reward during training — seed {seed} ({algorithm.upper()})")
  ax.grid(True, alpha=0.3)
  ax.legend(ncol=2, fontsize=9)
  fig.tight_layout()
  fig.savefig(out_dir / "09_eval_reward_curve.png", dpi=150)
  plt.close(fig)


def plot_reference_path(trajectory: TrajectoryKind, out_dir: Path):
  traj = make_trajectory(trajectory)
  t = np.linspace(0, 100, 500)
  pts = np.stack([traj.position(float(ti)) for ti in t], axis=0)
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection="3d")
  ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "r-", linewidth=1.5)
  ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="#2ca02c", s=36, marker="o", label="start")
  ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="#d62728", s=36, marker="o", label="end")
  ax.set_title(f"Reference path — {trajectory}")
  ax.set_xlabel("N"); ax.set_ylabel("E"); ax.set_zlabel("D")
  _set_equal_3d_aspect(ax, [pts])
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "00_reference_path.png", dpi=150)
  plt.close(fig)


def _plot_cross_eval(
  paths: PathConfig,
  seed: int,
  train_trajectory: TrajectoryKind = "circle",
  *,
  train_tag: str = TRAIN_TAG_DEFAULT,
  current_label: str | None = None,
  curriculum: bool = False,
):
  test_trajectories: list[TrajectoryKind] = ["circle", "sinusoid", "figure8"]
  all_summaries = {}
  rollouts_by_traj = {}
  out_root = paths.cross_eval_figure_dir(train_trajectory, seed)

  for traj in test_trajectories:
    rollouts = _load_cross_eval_rollouts(paths, traj, train_trajectory, seed)
    rollouts_by_traj[traj] = rollouts
    summary_path = paths.cross_eval_log_dir(train_trajectory, seed) / traj / "eval_summary.json"
    if summary_path.exists():
      all_summaries[traj] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
      successes = [r["success"] for r in rollouts]
      collisions = [r["collision"] for r in rollouts]
      warning_counts = [r.get("collision_warning_count", 0) for r in rollouts]
      tracking_errors = [r.get("cumulative_error", np.nan) for r in rollouts]
      rmses = [r["rmse_3d"] for r in rollouts]
      energies = [r["undiscounted_energy"] for r in rollouts]
      all_summaries[traj] = {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "warning_count_mean": float(np.mean(warning_counts)),
        "tracking_error_mean": float(np.nanmean(tracking_errors)),
        "rmse_3d_mean": float(np.mean(rmses)),
        "rmse_3d_std": float(np.std(rmses)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
      }

    out_dir = out_root / traj
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_reference_path(traj, out_dir)
    best = _log_best_episode_selection(traj, rollouts)
    plot_3d_trajectory(best, out_dir, traj)
    plot_tracking_errors(best, out_dir, traj)
    plot_actions(best, out_dir, traj)

  global_dir = out_root / "summary"
  global_dir.mkdir(parents=True, exist_ok=True)
  plot_rmse_summary(all_summaries, global_dir)
  plot_energy_comparison(all_summaries, rollouts_by_traj, global_dir)
  plot_success_collision_rates(all_summaries, global_dir)
  plot_training_eval_rewards(
    paths,
    seed,
    global_dir,
    train_tag=train_tag,
    current_label=current_label,
    curriculum=curriculum,
  )

  table_path = global_dir / "metrics_table.txt"
  lines = [
    f"Cross-eval (train={train_trajectory}, seed={seed})",
    "TestTrajectory\tSuccess%\tCollision%\tWarnMean\tTrackErrMean\tRMSE_mean\tEnergy_mean\tTheta",
  ]
  theta = CSACConfig().energy_budget_theta
  for name, s in all_summaries.items():
    lines.append(
      f"{name}\t{s['success_rate']*100:.1f}\t{s['collision_rate']*100:.1f}\t"
      f"{s.get('warning_count_mean', 0.0):.2f}\t{s.get('tracking_error_mean', 0.0):.2f}\t"
      f"{s['rmse_3d_mean']:.4f}\t{s['energy_mean']:.2f}\t{theta}"
    )
  table_path.write_text("\n".join(lines), encoding="utf-8")
  print(f"Cross-eval figures saved under {out_root}")
  print(table_path.read_text(encoding="utf-8"))


def build_arg_parser(
  *,
  default_algorithm: AlgorithmKind = "csac",
  default_train_tag: str | None = None,
) -> argparse.ArgumentParser:
  if default_train_tag is None:
    default_train_tag = TRAIN_TAG_DEFAULTS[default_algorithm]
  parser = argparse.ArgumentParser()
  parser.add_argument("--trajectory", type=str, default="all", choices=["circle", "sinusoid", "figure8", "all"])
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument(
    "--algorithm",
    type=str,
    default=default_algorithm,
    choices=["csac", "sac", "sacr"],
    help="算法类型：决定读取 logs/<algo>/ 与输出 figures/<algo>/",
  )
  parser.add_argument("--train-tag", type=str, default=default_train_tag, help="训练实验目录名（为空则按旧目录）")
  parser.add_argument("--current-label", type=str, default="", help="训练子目录标签（如 mild/moderate/strong）")
  parser.add_argument("--curriculum", action="store_true", help="读取 <train-tag>/curriculum 下训练日志")
  parser.add_argument(
    "--cross",
    action="store_true",
    help="绘制跨轨迹测试结果（需先运行 test_cross_trajectory.py）",
  )
  parser.add_argument("--train-trajectory", type=str, default="circle", choices=["circle"])
  return parser


def run_plot(args: argparse.Namespace) -> None:
  algorithm: AlgorithmKind = args.algorithm  # type: ignore[assignment]
  paths = PathConfig()
  paths.ensure()

  if args.cross:
    _plot_cross_eval(
      paths,
      args.seed,
      args.train_trajectory,  # type: ignore[arg-type]
      train_tag=args.train_tag,
      current_label=args.current_label or None,
      curriculum=args.curriculum,
    )
    return

  trajectories: list[TrajectoryKind] = (
    ["circle", "sinusoid", "figure8"] if args.trajectory == "all" else [args.trajectory]  # type: ignore
  )

  all_summaries = {}
  rollouts_by_traj = {}

  for traj in trajectories:
    rollouts = _load_eval_rollouts(paths, traj, args.seed, algorithm=algorithm)
    rollouts_by_traj[traj] = rollouts
    summary_path = paths.traj_eval_log_dir(traj, algorithm) / f"eval_summary_seed{args.seed}.json"
    if summary_path.exists():
      all_summaries[traj] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
      successes = [r["success"] for r in rollouts]
      collisions = [r["collision"] for r in rollouts]
      warning_counts = [r.get("collision_warning_count", 0) for r in rollouts]
      tracking_errors = [r.get("cumulative_error", np.nan) for r in rollouts]
      rmses = [r["rmse_3d"] for r in rollouts]
      energies = [r["undiscounted_energy"] for r in rollouts]
      all_summaries[traj] = {
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "warning_count_mean": float(np.mean(warning_counts)),
        "tracking_error_mean": float(np.nanmean(tracking_errors)),
        "rmse_3d_mean": float(np.mean(rmses)),
        "rmse_3d_std": float(np.std(rmses)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
      }

    out_dir = paths.traj_eval_figure_dir(traj, algorithm)
    plot_reference_path(traj, out_dir)
    best = _log_best_episode_selection(traj, rollouts)
    plot_3d_trajectory(best, out_dir, traj)
    plot_tracking_errors(best, out_dir, traj)
    plot_actions(best, out_dir, traj)
    plot_training_scores(
      paths,
      traj,
      args.seed,
      out_dir,
      train_tag=args.train_tag,
      current_label=args.current_label or None,
      curriculum=args.curriculum,
      algorithm=algorithm,
    )
    plot_training_cmdp_metrics(
      paths,
      traj,
      args.seed,
      out_dir,
      train_tag=args.train_tag,
      current_label=args.current_label or None,
      curriculum=args.curriculum,
      algorithm=algorithm,
    )

  global_dir = paths.summary_figure_dir(algorithm)
  plot_rmse_summary(all_summaries, global_dir)
  plot_energy_comparison(all_summaries, rollouts_by_traj, global_dir)
  plot_success_collision_rates(all_summaries, global_dir)
  plot_training_eval_rewards(
    paths,
    args.seed,
    global_dir,
    train_tag=args.train_tag,
    current_label=args.current_label or None,
    curriculum=args.curriculum,
    algorithm=algorithm,
  )

  table_path = global_dir / "metrics_table.txt"
  lines = [f"Algorithm\t{algorithm.upper()}", "Trajectory\tSuccess%\tCollision%\tWarnMean\tTrackErrMean\tRMSE_mean\tEnergy_mean\tTheta"]
  theta = CSACConfig().energy_budget_theta
  for name, s in all_summaries.items():
    lines.append(
      f"{name}\t{s['success_rate']*100:.1f}\t{s['collision_rate']*100:.1f}\t"
      f"{s.get('warning_count_mean', 0.0):.2f}\t{s.get('tracking_error_mean', 0.0):.2f}\t"
      f"{s['rmse_3d_mean']:.4f}\t{s['energy_mean']:.2f}\t{theta}"
    )
  table_path.write_text("\n".join(lines), encoding="utf-8")
  print(f"Figures saved under {paths.algorithm_figure_root(algorithm)}")
  print(table_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None):
  run_plot(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
  main()
