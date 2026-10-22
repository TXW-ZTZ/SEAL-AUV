"""Compare best SEAL, SAC, and SAC-E rollouts across three tracking scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PathConfig, SimConfig, TrajectoryKind
from env import AUVTrackingCMDPEnv
from evaluate import _build_eval_cfg_from_checkpoint
from rl_networks import GaussianActor
from utils import discounted_sum, distance_before_collision_m, episode_distance_m, episode_energy_fields


TRAJECTORIES: tuple[TrajectoryKind, ...] = ("circle", "sinusoid", "figure8")
METHODS = {
  "seal": {
    "label": "SEAL",
    "color": "#0F4D92",
    "train_tag": "single_traj_curriculum",
    "actor": "csac_actor",
  },
  "sac": {
    "label": "SAC",
    "color": "#B64342",
    "train_tag": "single_traj_curriculum_sac",
    "actor": "sac_actor",
  },
  "sace": {
    "label": "SAC-E",
    "color": "#42949E",
    "train_tag": "single_traj_curriculum_sacr",
    "actor": "sacr_actor",
  },
}
RMSE_DISPLAY = (
  ("SAC", "#B64342"),
  ("SAC-E", "#42949E"),
  ("SEAL", "#0F4D92"),
)
REFERENCE_COLOR = "#6E6E6E"
OBSTACLE_COLOR = "#8BCF8B"


def apply_publication_style() -> None:
  plt.rcParams.update(
    {
      "font.family": "Times New Roman",
      "font.serif": ["Times New Roman"],
      "font.size": 11,
      "axes.linewidth": 1.2,
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


def finite_float(value: object, default: float) -> float:
  try:
    out = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return out if np.isfinite(out) else default


def avoidance_cost(clearance: float) -> float:
  if not np.isfinite(clearance):
    return 100.0
  if clearance < 0.0:
    return 10.0 + abs(clearance) * 5.0
  return 1.0 / (clearance + 0.2)


def rank_key(result: dict) -> tuple:
  """Prioritize successful, safe full rollouts before tracking accuracy and clearance."""
  collision = bool(result.get("collision", False))
  clean_full = (
    int(result.get("steps", 0)) >= SimConfig().episode_steps
    and not collision
    and str(result.get("reason", "")) != "deviate"
  )
  rmse = finite_float(result.get("rmse_3d"), np.inf)
  clearance = finite_float(result.get("min_obstacle_surface"), -np.inf)
  return (
    0 if bool(result.get("success", False)) else 1,
    1 if collision else 0,
    0 if clean_full else 1,
    0.5 * rmse + 0.5 * avoidance_cost(clearance),
    int(result.get("collision_warning_count", 0)),
    finite_float(result.get("cumulative_error"), np.inf),
    finite_float(result.get("d_endpoint"), np.inf),
    finite_float(result.get("undiscounted_energy"), np.inf),
  )


def checkpoint_for(paths: PathConfig, method: str, trajectory: TrajectoryKind, seed: int) -> Path:
  spec = METHODS[method]
  path = paths.checkpoint_dir / spec["train_tag"] / f"{trajectory}_seed{seed}" / f"{spec['actor']}_seed{seed}_best.pt"
  if not path.exists():
    raise FileNotFoundError(f"Missing {spec['label']} best checkpoint: {path}")
  return path


class InferenceActor:
  """Actor-only deterministic policy used to avoid loading unused critics during evaluation."""

  def __init__(self, checkpoint: Path, device: str):
    self.device = torch.device(device)
    state = torch.load(checkpoint, map_location=self.device, weights_only=False)
    cfg = _build_eval_cfg_from_checkpoint(checkpoint, device=device)
    self.action_dim = cfg.action_dim
    self.actor = GaussianActor(cfg.obs_dim, cfg.action_dim, hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers).to(self.device)
    self.actor.load_state_dict(state["actor"])
    self.actor.eval()

  def select_action(self, observation: np.ndarray, random_explore: bool = False, deterministic: bool = True) -> np.ndarray:
    obs = torch.from_numpy(np.ascontiguousarray(observation)).to(self.device, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
      action, _ = self.actor(obs, deterministic=True)
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def evaluate_method(
  method: str,
  trajectory: TrajectoryKind,
  seed: int,
  episodes: int,
  eval_seed_base: int,
  device: str,
  paths: PathConfig,
) -> list[dict]:
  spec = METHODS[method]
  checkpoint = checkpoint_for(paths, method, trajectory, seed)
  cfg = _build_eval_cfg_from_checkpoint(checkpoint, device=device)
  agent = InferenceActor(checkpoint, device)
  results: list[dict] = []
  for index in range(episodes):
    eval_seed = eval_seed_base + index
    env = AUVTrackingCMDPEnv(sim_cfg=SimConfig(seed=seed, trajectory=trajectory), randomize_start=True)
    obs, total_reward, done, info = env.reset(seed=eval_seed), 0.0, False, None
    while not done:
      action = agent.select_action(obs, random_explore=False, deterministic=True)
      obs, reward, _, done, info = env.step(action)
      total_reward += reward
    errors = np.asarray(env.episode_log.errors_xyz)
    rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))) if len(errors) else float("nan")
    rmse_xyz = np.sqrt(np.mean(errors ** 2, axis=0)) if len(errors) else np.zeros(3)
    collision = bool(info.collision)
    result = {
      "trajectory": trajectory,
      "algorithm": method,
      "success": bool(info.success),
      "collision": collision,
      "collision_state": str(info.collision_state),
      "collision_warning": bool(info.collision_warning),
      "collision_warning_count": int(info.collision_warning_count),
      "reason": info.reason,
      "d_endpoint": float(info.d_endpoint),
      "min_obstacle_surface": float(info.min_obstacle_surface),
      "total_reward": float(total_reward),
      "undiscounted_energy": float(env.episode_cost),
      **episode_energy_fields(env),
      "discounted_Jc": discounted_sum(env.episode_log.step_costs, cfg.gamma),
      "rmse_3d": rmse,
      "rmse_x": float(rmse_xyz[0]),
      "rmse_y": float(rmse_xyz[1]),
      "rmse_z": float(rmse_xyz[2]),
      "steps": env.step_counter,
      "cumulative_error": float(env.cumulative_error),
      "episode_distance_m": float(episode_distance_m(env.start, env.episode_log.positions)),
      "distance_before_collision_m": distance_before_collision_m(env.start, env.episode_log.positions, collision),
      "positions": [point.tolist() for point in env.episode_log.positions],
      "ref_positions": [point.tolist() for point in env.episode_log.ref_positions],
      "errors_xyz": [error.tolist() for error in env.episode_log.errors_xyz],
      "times": env.episode_log.times,
      "obstacle_centers": env.obstacles.centers.tolist(),
      "obstacle_radius": env.obstacles.radius,
    }
    result.update(method=method, method_label=spec["label"], eval_seed=eval_seed, checkpoint=str(checkpoint))
    results.append(result)
    print(
      f"{trajectory} {spec['label']} {index + 1:03d}/{episodes}: "
      f"success={result['success']} collision={result['collision']} rmse={result['rmse_3d']:.3f}",
      flush=True,
    )
  del agent
  if device.startswith("cuda"):
    torch.cuda.empty_cache()
  return results


def set_equal_3d_aspect(ax, arrays: list[np.ndarray]) -> None:
  points = np.vstack([np.asarray(array, dtype=np.float64) for array in arrays if np.asarray(array).size])
  lower, upper = np.min(points, axis=0), np.max(points, axis=0)
  center = (lower + upper) / 2.0
  half = 0.5 * float(np.max(np.maximum(upper - lower, 1e-6)))
  ax.set_xlim(center[0] - half, center[0] + half)
  ax.set_ylim(center[1] - half, center[1] + half)
  ax.set_zlim(center[2] - half, center[2] + half)
  if hasattr(ax, "set_box_aspect"):
    ax.set_box_aspect((1, 1, 1))


def draw_obstacles(ax, centers: np.ndarray, radius: float) -> None:
  u, v = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 12)
  for center in centers:
    ax.plot_surface(
      center[0] + radius * np.outer(np.cos(u), np.sin(v)),
      center[1] + radius * np.outer(np.sin(u), np.sin(v)),
      center[2] + radius * np.outer(np.ones_like(u), np.cos(v)),
      color=OBSTACLE_COLOR, alpha=0.48, linewidth=0.25, edgecolor="#4B8D4B", shade=True,
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
  stem.parent.mkdir(parents=True, exist_ok=True)
  saved = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
  for path in saved:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
  plt.close(fig)
  return saved


def plot_trajectory_comparison(trajectory: TrajectoryKind, selected: dict[str, dict], out_dir: Path) -> list[Path]:
  fig = plt.figure(figsize=(6.45, 5.55))
  ax = fig.add_subplot(111, projection="3d")
  seal = selected["seal"]
  reference = np.asarray(seal["ref_positions"], dtype=np.float64)
  obstacles = np.asarray(seal["obstacle_centers"], dtype=np.float64)
  display = rmse_display_order(selected)
  positions = [np.asarray(selected[name]["positions"], dtype=np.float64) for name, _, _ in display]
  ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], "--", color=REFERENCE_COLOR, linewidth=0.70, label="Reference")
  for (name, label, color), position in zip(display, positions):
    episode = selected[name]
    ax.plot(
      position[:, 0], position[:, 1], position[:, 2], color=color, linewidth=0.72,
      label=f"{label} (RMSE {episode['rmse_3d']:.2f} m)",
    )
    ax.scatter(position[0, 0], position[0, 1], position[0, 2], s=12, color=color, marker="o")
    ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2], s=18, color=color, marker="^")
  draw_obstacles(ax, obstacles, float(seal["obstacle_radius"]))
  set_equal_3d_aspect(ax, [reference, obstacles, *positions])
  ax.set_xlabel("N (m)", labelpad=4)
  ax.set_ylabel("E (m)", labelpad=4)
  ax.set_zlabel("D (m)", labelpad=4)
  ax.set_title(f"Best test trajectories: {trajectory}", pad=8, fontsize=13)
  ax.view_init(elev=24, azim=-54)
  handles, labels = ax.get_legend_handles_labels()
  method_handles = [(handle, label) for handle, label in zip(handles, labels) if label != "Reference"]
  reference_handles = [(handle, label) for handle, label in zip(handles, labels) if label == "Reference"]
  ordered_handles, ordered_labels = zip(*(method_handles + reference_handles))
  ax.legend(ordered_handles, ordered_labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, ncol=1)
  fig.tight_layout(rect=(0, 0.09, 1, 1))
  return save_figure(fig, out_dir / f"seal_sac_sace_{trajectory}_trajectory_comparison")


def plot_ned_errors(trajectory: TrajectoryKind, selected: dict[str, dict], out_dir: Path) -> list[Path]:
  fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), sharex=False)
  component_names = ("N", "E", "D")
  display = rmse_display_order(selected)
  for component, ax in enumerate(axes):
    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.65, label="Reference (zero error)")
    for name, label, color in display:
      episode = selected[name]
      errors = np.asarray(episode["errors_xyz"], dtype=np.float64)
      times = np.asarray(episode["times"], dtype=np.float64)
      count = min(len(times), len(errors))
      ax.plot(times[:count], errors[:count, component], color=color, linewidth=0.72, label=label)
    ax.set_title(f"{component_names[component]}-direction error", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)" if component == 0 else "")
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.5, alpha=0.8)
  handles, labels = axes[0].get_legend_handles_labels()
  method_handles = [(handle, label) for handle, label in zip(handles, labels) if label != "Reference (zero error)"]
  reference_handles = [(handle, label) for handle, label in zip(handles, labels) if label == "Reference (zero error)"]
  ordered_handles, ordered_labels = zip(*(method_handles + reference_handles))
  fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)
  fig.suptitle(f"NED tracking errors of best test trajectories: {trajectory}", y=0.99, fontsize=13)
  fig.tight_layout(rect=(0, 0.10, 1, 0.92))
  return save_figure(fig, out_dir / f"seal_sac_sace_{trajectory}_ned_error_comparison")


def selection_record(index: int, episode: dict) -> dict:
  return {
    "test_index": index + 1,
    "eval_seed": episode.get("eval_seed"),
    "success": episode.get("success"),
    "collision": episode.get("collision"),
    "reason": episode.get("reason"),
    "rmse_3d": finite_float(episode.get("rmse_3d"), np.nan),
    "min_obstacle_surface": finite_float(episode.get("min_obstacle_surface"), np.nan),
    "undiscounted_energy": finite_float(episode.get("undiscounted_energy"), np.nan),
    "total_reward": finite_float(episode.get("total_reward"), np.nan),
  }


def rmse_display_order(selected: dict[str, dict]) -> list[tuple[str, str, str]]:
  """Return median, largest, then smallest RMSE for the requested legend order."""
  ranked = sorted(selected, key=lambda name: (finite_float(selected[name].get("rmse_3d"), np.inf), name))
  best, middle, worst = ranked
  return [
    (middle, *RMSE_DISPLAY[0]),
    (worst, *RMSE_DISPLAY[1]),
    (best, *RMSE_DISPLAY[2]),
  ]


def write_selection_csv(trajectory: TrajectoryKind, selected: dict[str, dict], out_dir: Path) -> Path:
  path = out_dir / f"seal_sac_sace_{trajectory}_best_selection.csv"
  fields = ["method", "label", "test_index", "eval_seed", "success", "collision", "reason", "rmse_3d", "min_obstacle_surface", "undiscounted_energy", "total_reward"]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for name, episode in selected.items():
      writer.writerow({field: episode.get(field, "") for field in fields} | {"method": name, "label": METHODS[name]["label"]})
  return path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=666)
  parser.add_argument("--episodes", type=int, default=500)
  parser.add_argument("--trajectory", choices=["all", "circle", "sinusoid", "figure8"], default="all")
  parser.add_argument("--eval-seed-base", type=int, default=1000)
  parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0", "auto"], default="cuda")
  parser.add_argument("--reuse", action="store_true", help="Reuse saved rollout data when available.")
  args = parser.parse_args()

  apply_publication_style()
  paths = PathConfig()
  paths.ensure()
  tag = f"seed{args.seed}_episodes{args.episodes}"
  out_root = paths.figure_dir / "seal_sac_sace_best" / tag
  log_root = paths.log_dir / "seal_sac_sace_best" / tag
  out_root.mkdir(parents=True, exist_ok=True)
  log_root.mkdir(parents=True, exist_ok=True)
  all_summary: dict[str, dict[str, dict]] = {}

  trajectories = TRAJECTORIES if args.trajectory == "all" else (args.trajectory,)
  for trajectory in trajectories:
    selected: dict[str, dict] = {}
    for method in METHODS:
      rollouts_path = log_root / trajectory / f"{method}_rollouts.json"
      if args.reuse and rollouts_path.exists():
        results = json.loads(rollouts_path.read_text(encoding="utf-8"))
      else:
        results = evaluate_method(method, trajectory, args.seed, args.episodes, args.eval_seed_base, args.device, paths)
        rollouts_path.parent.mkdir(parents=True, exist_ok=True)
        rollouts_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
      best_index, best = min(enumerate(results), key=lambda item: rank_key(item[1]))
      selected[method] = best | selection_record(best_index, best)
    scene_dir = out_root / trajectory
    plot_trajectory_comparison(trajectory, selected, scene_dir)
    plot_ned_errors(trajectory, selected, scene_dir)
    write_selection_csv(trajectory, selected, scene_dir)
    all_summary[trajectory] = {name: selection_record(episode["test_index"] - 1, episode) for name, episode in selected.items()}

  summary_path = out_root / "seal_sac_sace_best_selection_summary.json"
  summary_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
  print(f"Generated: {summary_path}")


if __name__ == "__main__":
  main()
