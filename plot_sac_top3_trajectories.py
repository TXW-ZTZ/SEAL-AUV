"""Evaluate SAC best checkpoints and plot the top-3 rollouts per trajectory.

Default checkpoint set:
  checkpoints/single_traj_curriculum_sac/{trajectory}_seed666/sac_actor_seed666_best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig, SimConfig, TrajectoryKind
from evaluate_sac import evaluate_episode


PALETTE = {
  "blue_main": "#0F4D92",
  "blue_secondary": "#3775BA",
  "green_3": "#8BCF8B",
  "red_strong": "#B64342",
  "neutral": "#CFCECE",
  "teal": "#42949E",
  "violet": "#9A4D8E",
}


TRAJECTORIES: tuple[TrajectoryKind, ...] = ("circle", "sinusoid", "figure8")


def apply_publication_style(font_size: int = 11) -> None:
  plt.rcParams.update(
    {
      "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
      "font.size": font_size,
      "axes.linewidth": 1.4,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "legend.frameon": False,
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
      "svg.fonttype": "none",
      "figure.dpi": 150,
      "savefig.dpi": 300,
    }
  )


def checkpoint_for(paths: PathConfig, train_tag: str, trajectory: TrajectoryKind, seed: int) -> Path:
  ckpt = paths.checkpoint_dir / train_tag / f"{trajectory}_seed{seed}" / f"sac_actor_seed{seed}_best.pt"
  if not ckpt.exists():
    raise FileNotFoundError(f"Missing SAC best checkpoint: {ckpt}")
  return ckpt


def avoidance_cost(min_surface: float) -> float:
  if not np.isfinite(min_surface):
    return 100.0
  if min_surface < 0.0:
    return 10.0 + abs(min_surface) * 5.0
  return 1.0 / (min_surface + 0.2)


def rank_key(result: dict) -> tuple:
  episode_steps = SimConfig().episode_steps
  success = bool(result.get("success", False))
  collision = bool(result.get("collision", False))
  reason = str(result.get("reason", ""))
  steps = int(result.get("steps", 0))
  clean_full = steps >= episode_steps and not collision and reason != "deviate"
  rmse = finite_float(result.get("rmse_3d"), np.inf)
  min_surface = finite_float(result.get("min_obstacle_surface"), -np.inf)
  warnings = int(result.get("collision_warning_count", 0))
  cumulative_error = finite_float(result.get("cumulative_error"), np.inf)
  endpoint_error = finite_float(result.get("d_endpoint"), np.inf)
  energy = finite_float(result.get("undiscounted_energy"), np.inf)
  combined = 0.5 * rmse + 0.5 * avoidance_cost(min_surface)
  return (
    0 if success else 1,
    1 if collision else 0,
    0 if clean_full else 1,
    combined,
    warnings,
    cumulative_error,
    endpoint_error,
    energy,
  )


def finite_float(value: object, default: float) -> float:
  try:
    out = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return out if np.isfinite(out) else default


def set_equal_3d_aspect(ax, arrays: list[np.ndarray]) -> None:
  valid = [np.asarray(a, dtype=np.float64) for a in arrays if np.asarray(a).size > 0]
  if not valid:
    return
  pts = np.vstack(valid)
  xyz_min = np.min(pts, axis=0)
  xyz_max = np.max(pts, axis=0)
  center = 0.5 * (xyz_min + xyz_max)
  half = 0.5 * float(np.max(np.maximum(xyz_max - xyz_min, 1e-6)))
  ax.set_xlim(center[0] - half, center[0] + half)
  ax.set_ylim(center[1] - half, center[1] + half)
  ax.set_zlim(center[2] - half, center[2] + half)
  if hasattr(ax, "set_box_aspect"):
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_obstacle_spheres(ax, centers: np.ndarray, radius: float) -> None:
  u = np.linspace(0, 2 * np.pi, 20)
  v = np.linspace(0, np.pi, 12)
  for c in centers:
    xs = c[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=PALETTE["green_3"], alpha=0.28, linewidth=0, shade=True)


def metric_label(ep: dict) -> str:
  return (
    f"RMSE {finite_float(ep.get('rmse_3d'), np.nan):.2f} m\n"
    f"clearance {finite_float(ep.get('min_obstacle_surface'), np.nan):.2f} m\n"
    f"energy {finite_float(ep.get('undiscounted_energy'), np.nan):.1f}"
  )


def plot_top3(trajectory: TrajectoryKind, top3: list[tuple[int, dict]], out_dir: Path) -> list[Path]:
  fig = plt.figure(figsize=(13.5, 4.5))
  handles = None
  for panel_idx, (episode_idx, ep) in enumerate(top3, start=1):
    ax = fig.add_subplot(1, 3, panel_idx, projection="3d")
    pos = np.asarray(ep["positions"], dtype=np.float64)
    ref = np.asarray(ep["ref_positions"], dtype=np.float64)
    obs = np.asarray(ep["obstacle_centers"], dtype=np.float64)
    radius = float(ep["obstacle_radius"])

    ref_line = ax.plot(
      ref[:, 0],
      ref[:, 1],
      ref[:, 2],
      linestyle="--",
      linewidth=1.6,
      color=PALETTE["red_strong"],
      label="Reference",
    )[0]
    auv_line = ax.plot(
      pos[:, 0],
      pos[:, 1],
      pos[:, 2],
      linewidth=1.8,
      color=PALETTE["blue_main"],
      label="SAC rollout",
    )[0]
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], s=28, color=PALETTE["teal"], marker="o", label="Start")
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], s=34, color=PALETTE["violet"], marker="^", label="End")
    draw_obstacle_spheres(ax, obs, radius)
    set_equal_3d_aspect(ax, [ref, pos, obs])
    ax.set_title(f"Rank {panel_idx} (test #{episode_idx + 1})", pad=8)
    ax.text2D(0.03, 0.88, metric_label(ep), transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("N (m)", labelpad=4)
    ax.set_ylabel("E (m)", labelpad=4)
    ax.set_zlabel("D (m)", labelpad=4)
    ax.view_init(elev=24, azim=-54)
    handles = [ref_line, auv_line]

  if handles:
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
  fig.suptitle(f"SAC top-3 test trajectories: {trajectory}", y=0.99, fontsize=14)
  fig.tight_layout(rect=(0, 0.04, 1, 0.95))
  out_dir.mkdir(parents=True, exist_ok=True)
  png = out_dir / f"sac_top3_{trajectory}_3d.png"
  pdf = out_dir / f"sac_top3_{trajectory}_3d.pdf"
  fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
  fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
  plt.close(fig)
  return [png, pdf]


def write_selection_table(trajectory: TrajectoryKind, top3: list[tuple[int, dict]], out_dir: Path) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"sac_top3_{trajectory}_selection.csv"
  fields = [
    "rank",
    "test_index",
    "eval_seed",
    "success",
    "collision",
    "reason",
    "steps",
    "rmse_3d",
    "min_obstacle_surface",
    "collision_warning_count",
    "cumulative_error",
    "d_endpoint",
    "undiscounted_energy",
    "total_reward",
  ]
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for rank, (episode_idx, ep) in enumerate(top3, start=1):
      writer.writerow(
        {
          "rank": rank,
          "test_index": episode_idx + 1,
          "eval_seed": ep.get("eval_seed", ""),
          "success": ep.get("success", ""),
          "collision": ep.get("collision", ""),
          "reason": ep.get("reason", ""),
          "steps": ep.get("steps", ""),
          "rmse_3d": ep.get("rmse_3d", ""),
          "min_obstacle_surface": ep.get("min_obstacle_surface", ""),
          "collision_warning_count": ep.get("collision_warning_count", ""),
          "cumulative_error": ep.get("cumulative_error", ""),
          "d_endpoint": ep.get("d_endpoint", ""),
          "undiscounted_energy": ep.get("undiscounted_energy", ""),
          "total_reward": ep.get("total_reward", ""),
        }
      )
  return path


def evaluate_trajectory(
  trajectory: TrajectoryKind,
  seed: int,
  episodes: int,
  train_tag: str,
  eval_seed_base: int,
  device: str | None,
  paths: PathConfig,
) -> list[dict]:
  ckpt = checkpoint_for(paths, train_tag, trajectory, seed)
  results = []
  for i in range(episodes):
    eval_seed = eval_seed_base + i
    result = evaluate_episode(
      trajectory=trajectory,
      seed=seed,
      checkpoint=ckpt,
      eval_seed=eval_seed,
      device=device,
      train_tag=train_tag,
    )
    result["eval_seed"] = eval_seed
    result["checkpoint"] = str(ckpt)
    results.append(result)
    print(
      f"{trajectory} test {i + 1:02d}/{episodes}: "
      f"success={result['success']} collision={result['collision']} "
      f"rmse={result['rmse_3d']:.3f} clearance={result['min_obstacle_surface']:.3f}"
    )
  return results


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=666)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--train-tag", type=str, default="single_traj_curriculum_sac")
  parser.add_argument("--eval-seed-base", type=int, default=1000)
  parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "cuda:0", "auto"])
  parser.add_argument("--reuse", action="store_true", help="Reuse saved rollouts when available.")
  args = parser.parse_args()

  apply_publication_style()
  paths = PathConfig()
  paths.ensure()
  out_root = paths.figure_dir / "sac_top3" / f"seed{args.seed}_episodes{args.episodes}"
  log_root = paths.log_dir / "sac_top3" / f"seed{args.seed}_episodes{args.episodes}"
  out_root.mkdir(parents=True, exist_ok=True)
  log_root.mkdir(parents=True, exist_ok=True)

  summary: dict[str, list[dict]] = {}
  generated: list[str] = []
  for trajectory in TRAJECTORIES:
    rollout_path = log_root / f"{trajectory}_rollouts.json"
    if args.reuse and rollout_path.exists():
      results = json.loads(rollout_path.read_text(encoding="utf-8"))
    else:
      results = evaluate_trajectory(
        trajectory=trajectory,
        seed=args.seed,
        episodes=args.episodes,
        train_tag=args.train_tag,
        eval_seed_base=args.eval_seed_base,
        device=args.device,
        paths=paths,
      )
      rollout_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ranked = sorted(enumerate(results), key=lambda item: rank_key(item[1]))
    top3 = ranked[:3]
    traj_out = out_root / trajectory
    generated.extend(str(p) for p in plot_top3(trajectory, top3, traj_out))
    table = write_selection_table(trajectory, top3, traj_out)
    generated.append(str(table))
    summary[trajectory] = [
      {
        "rank": rank,
        "test_index": idx + 1,
        "eval_seed": ep.get("eval_seed"),
        "success": ep.get("success"),
        "collision": ep.get("collision"),
        "reason": ep.get("reason"),
        "rmse_3d": ep.get("rmse_3d"),
        "min_obstacle_surface": ep.get("min_obstacle_surface"),
        "undiscounted_energy": ep.get("undiscounted_energy"),
        "total_reward": ep.get("total_reward"),
      }
      for rank, (idx, ep) in enumerate(top3, start=1)
    ]

  summary_path = out_root / "sac_top3_selection_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  generated.append(str(summary_path))
  print("Generated files:")
  for item in generated:
    print(item)


if __name__ == "__main__":
  main()
