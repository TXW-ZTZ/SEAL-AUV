"""Evaluate CSAC best checkpoints and overlay the top-3 rollouts per scenario."""

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
from evaluate import evaluate_episode


PALETTE = {
  "blue_main": "#0F4D92",
  "red_strong": "#B64342",
  "green_3": "#8BCF8B",
  "teal": "#42949E",
  "violet": "#9A4D8E",
}
ROLLOUT_COLORS = (PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"])
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
  checkpoint = paths.checkpoint_dir / train_tag / f"{trajectory}_seed{seed}" / f"csac_actor_seed{seed}_best.pt"
  if not checkpoint.exists():
    raise FileNotFoundError(f"Missing CSAC best checkpoint: {checkpoint}")
  return checkpoint


def finite_float(value: object, default: float) -> float:
  try:
    number = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return number if np.isfinite(number) else default


def avoidance_cost(min_surface: float) -> float:
  if not np.isfinite(min_surface):
    return 100.0
  if min_surface < 0.0:
    return 10.0 + abs(min_surface) * 5.0
  return 1.0 / (min_surface + 0.2)


def rank_key(result: dict) -> tuple:
  """Prefer successful, collision-free, full-length rollouts, then accuracy and clearance."""
  success = bool(result.get("success", False))
  collision = bool(result.get("collision", False))
  full_clean = (
    int(result.get("steps", 0)) >= SimConfig().episode_steps
    and not collision
    and str(result.get("reason", "")) != "deviate"
  )
  rmse = finite_float(result.get("rmse_3d"), np.inf)
  min_surface = finite_float(result.get("min_obstacle_surface"), -np.inf)
  return (
    0 if success else 1,
    1 if collision else 0,
    0 if full_clean else 1,
    0.5 * rmse + 0.5 * avoidance_cost(min_surface),
    int(result.get("collision_warning_count", 0)),
    finite_float(result.get("cumulative_error"), np.inf),
    finite_float(result.get("d_endpoint"), np.inf),
    finite_float(result.get("undiscounted_energy"), np.inf),
  )


def set_equal_3d_aspect(ax, arrays: list[np.ndarray]) -> None:
  valid = [np.asarray(array, dtype=np.float64) for array in arrays if np.asarray(array).size]
  if not valid:
    return
  points = np.vstack(valid)
  low, high = np.min(points, axis=0), np.max(points, axis=0)
  center = (low + high) / 2
  half_range = 0.5 * float(np.max(np.maximum(high - low, 1e-6)))
  ax.set_xlim(center[0] - half_range, center[0] + half_range)
  ax.set_ylim(center[1] - half_range, center[1] + half_range)
  ax.set_zlim(center[2] - half_range, center[2] + half_range)
  if hasattr(ax, "set_box_aspect"):
    ax.set_box_aspect((1, 1, 1))


def draw_obstacle_spheres(ax, centers: np.ndarray, radius: float) -> None:
  u, v = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 12)
  for center in centers:
    ax.plot_surface(
      center[0] + radius * np.outer(np.cos(u), np.sin(v)),
      center[1] + radius * np.outer(np.sin(u), np.sin(v)),
      center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
      color=PALETTE["green_3"], alpha=0.28, linewidth=0, shade=True,
    )


def plot_top3(trajectory: TrajectoryKind, top3: list[tuple[int, dict]], out_dir: Path) -> list[Path]:
  fig = plt.figure(figsize=(6.7, 5.8))
  ax = fig.add_subplot(111, projection="3d")
  first = top3[0][1]
  reference = np.asarray(first["ref_positions"], dtype=np.float64)
  obstacles = np.asarray(first["obstacle_centers"], dtype=np.float64)
  positions = [np.asarray(episode["positions"], dtype=np.float64) for _, episode in top3]
  all_points = [reference, obstacles, *positions]

  ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], "--", color=PALETTE["red_strong"], linewidth=1.2, label="Reference")
  for rank, ((test_index, episode), position, color) in enumerate(zip(top3, positions, ROLLOUT_COLORS), start=1):
    ax.plot(
      position[:, 0], position[:, 1], position[:, 2], color=color, linewidth=1.15,
      label=(f"CSAC rank {rank} (test #{test_index + 1}, RMSE {finite_float(episode.get('rmse_3d'), np.nan):.2f} m)"),
    )
    ax.scatter(position[0, 0], position[0, 1], position[0, 2], s=18, color=color, marker="o")
    ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2], s=24, color=color, marker="^")
  draw_obstacle_spheres(ax, obstacles, float(first["obstacle_radius"]))
  set_equal_3d_aspect(ax, all_points)
  ax.set_xlabel("N (m)", labelpad=4)
  ax.set_ylabel("E (m)", labelpad=4)
  ax.set_zlabel("D (m)", labelpad=4)
  ax.set_title(f"CSAC top-3 test trajectories: {trajectory}", pad=10, fontsize=13)
  ax.view_init(elev=24, azim=-54)
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=9, ncol=1)
  fig.tight_layout(rect=(0, 0.09, 1, 1))
  out_dir.mkdir(parents=True, exist_ok=True)
  png, pdf = out_dir / f"csac_top3_{trajectory}_3d.png", out_dir / f"csac_top3_{trajectory}_3d.pdf"
  fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
  fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
  plt.close(fig)
  return [png, pdf]


def evaluate_trajectory(trajectory: TrajectoryKind, seed: int, episodes: int, train_tag: str, eval_seed_base: int, device: str | None, paths: PathConfig) -> list[dict]:
  checkpoint = checkpoint_for(paths, train_tag, trajectory, seed)
  results: list[dict] = []
  for index in range(episodes):
    eval_seed = eval_seed_base + index
    result = evaluate_episode(trajectory=trajectory, seed=seed, checkpoint=checkpoint, eval_seed=eval_seed, device=device, train_tag=train_tag)
    result.update(eval_seed=eval_seed, checkpoint=str(checkpoint))
    results.append(result)
    print(f"{trajectory} test {index + 1:02d}/{episodes}: success={result['success']} collision={result['collision']} rmse={result['rmse_3d']:.3f} clearance={result['min_obstacle_surface']:.3f}")
  return results


def write_selection_table(trajectory: TrajectoryKind, top3: list[tuple[int, dict]], out_dir: Path) -> Path:
  path = out_dir / f"csac_top3_{trajectory}_selection.csv"
  fields = ["rank", "test_index", "eval_seed", "success", "collision", "reason", "steps", "rmse_3d", "min_obstacle_surface", "collision_warning_count", "cumulative_error", "d_endpoint", "undiscounted_energy", "total_reward"]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for rank, (index, episode) in enumerate(top3, start=1):
      writer.writerow({field: episode.get(field, "") for field in fields} | {"rank": rank, "test_index": index + 1})
  return path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=666)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--train-tag", type=str, default="single_traj_curriculum")
  parser.add_argument("--eval-seed-base", type=int, default=1000)
  parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0", "auto"], default="cpu")
  parser.add_argument("--reuse", action="store_true")
  args = parser.parse_args()

  apply_publication_style()
  paths = PathConfig()
  paths.ensure()
  out_root = paths.figure_dir / "csac_top3" / f"seed{args.seed}_episodes{args.episodes}"
  log_root = paths.log_dir / "csac_top3" / f"seed{args.seed}_episodes{args.episodes}"
  out_root.mkdir(parents=True, exist_ok=True)
  log_root.mkdir(parents=True, exist_ok=True)
  summary: dict[str, list[dict]] = {}

  for trajectory in TRAJECTORIES:
    rollout_path = log_root / f"{trajectory}_rollouts.json"
    if args.reuse and rollout_path.exists():
      results = json.loads(rollout_path.read_text(encoding="utf-8"))
    else:
      results = evaluate_trajectory(trajectory, args.seed, args.episodes, args.train_tag, args.eval_seed_base, args.device, paths)
      rollout_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    top3 = sorted(enumerate(results), key=lambda item: rank_key(item[1]))[:3]
    scene_dir = out_root / trajectory
    generated = plot_top3(trajectory, top3, scene_dir)
    generated.append(write_selection_table(trajectory, top3, scene_dir))
    summary[trajectory] = [
      {"rank": rank, "test_index": index + 1, "eval_seed": episode.get("eval_seed"), "success": episode.get("success"), "collision": episode.get("collision"), "reason": episode.get("reason"), "rmse_3d": episode.get("rmse_3d"), "min_obstacle_surface": episode.get("min_obstacle_surface"), "undiscounted_energy": episode.get("undiscounted_energy"), "total_reward": episode.get("total_reward")}
      for rank, (index, episode) in enumerate(top3, start=1)
    ]
    print("Generated:", *map(str, generated), sep="\n")

  summary_path = out_root / "csac_top3_selection_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(f"Generated: {summary_path}")


if __name__ == "__main__":
  main()
