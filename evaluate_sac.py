"""加载 SAC 策略，评估并保存 rollout 数据供绘图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import (
  SimConfig,
  CSACConfig,
  PathConfig,
  TrajectoryKind,
  CurrentIntensity,
  apply_fast_training_profile,
)
from env import AUVTrackingCMDPEnv
from sac_agent import SACAgent
from ocean_current import CURRENT_LABELS, parse_current_intensity
from utils import discounted_sum, episode_distance_m, distance_before_collision_m, episode_energy_fields


TRAIN_TAG_DEFAULT = "single_traj_curriculum_sac"


def _cfg_from_checkpoint_meta(state: dict) -> CSACConfig | None:
  raw = state.get("cfg")
  if raw is None:
    return None
  if isinstance(raw, CSACConfig):
    return raw
  if isinstance(raw, dict):
    cfg = CSACConfig()
    for key in cfg.__dataclass_fields__.keys():
      if key in raw:
        setattr(cfg, key, raw[key])
    return cfg
  return None


def _infer_arch_from_actor_state(state: dict) -> tuple[int, int] | None:
  actor = state.get("actor")
  if not isinstance(actor, dict):
    return None
  linear_weights = []
  for key, value in actor.items():
    if key.startswith("backbone.net.") and key.endswith(".weight"):
      if hasattr(value, "ndim") and int(value.ndim) == 2:
        linear_weights.append(value)
  if not linear_weights:
    return None
  hidden_dim = int(linear_weights[0].shape[0])
  num_hidden_layers = len(linear_weights)
  return hidden_dim, num_hidden_layers


def _build_eval_cfg_from_checkpoint(
  ckpt_path: Path,
  *,
  device: str | None = None,
  fast: bool = False,
) -> CSACConfig:
  cfg = CSACConfig()
  if fast:
    apply_fast_training_profile(cfg)
  state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  ckpt_cfg = _cfg_from_checkpoint_meta(state)
  if ckpt_cfg is not None:
    cfg = ckpt_cfg
    if fast:
      print("Info: checkpoint 内含 cfg，忽略 --fast，按 checkpoint 结构评估。")
  else:
    inferred = _infer_arch_from_actor_state(state)
    if inferred is not None:
      cfg.hidden_dim, cfg.num_hidden_layers = inferred
      print(
        f"Info: checkpoint 无 cfg，已从 actor 权重推断结构 "
        f"(hidden_dim={cfg.hidden_dim}, num_hidden_layers={cfg.num_hidden_layers})。"
      )
  if device:
    cfg.device = device
  return cfg


def evaluate_episode(
  trajectory: TrajectoryKind,
  seed: int,
  checkpoint: Path | str | None = None,
  eval_seed: int | None = None,
  device: str | None = None,
  current_intensity: CurrentIntensity = 0,
  train_tag: str = TRAIN_TAG_DEFAULT,
  curriculum: bool = False,
  fast: bool = False,
) -> dict:
  if checkpoint is not None:
    checkpoint = Path(checkpoint)
  sim_cfg = SimConfig(seed=seed, trajectory=trajectory, current_intensity=current_intensity)
  paths = PathConfig()
  current_label = CURRENT_LABELS[current_intensity]
  if train_tag:
    ckpt_root = paths.checkpoint_dir / train_tag
  else:
    ckpt_root = paths.traj_checkpoint_dir(trajectory)
  if curriculum:
    ckpt_root = ckpt_root / "curriculum"
  elif current_intensity > 0:
    ckpt_root = ckpt_root / current_label
  ckpt_path = checkpoint or ckpt_root / f"sac_actor_seed{seed}.pt"
  if not ckpt_path.exists():
    best_ckpt = ckpt_root / f"sac_actor_seed{seed}_best.pt"
    if best_ckpt.exists():
      ckpt_path = best_ckpt
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
  csac_cfg = _build_eval_cfg_from_checkpoint(ckpt_path, device=device, fast=fast)

  env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
  agent = SACAgent(csac_cfg, device=device)
  agent.load(ckpt_path)

  obs = env.reset(seed=eval_seed if eval_seed is not None else seed)
  total_reward = 0.0
  done = False
  info = None
  while not done:
    action = agent.select_action(obs, random_explore=False, deterministic=True)
    obs, reward, cost, done, info = env.step(action)
    total_reward += reward

  errors = np.asarray(env.episode_log.errors_xyz)
  rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))) if len(errors) else float("nan")
  rmse_xyz = np.sqrt(np.mean(errors ** 2, axis=0)) if len(errors) else np.zeros(3)
  collision = bool(info.collision)
  ep_distance_m = episode_distance_m(env.start, env.episode_log.positions)
  dist_before_collision = distance_before_collision_m(env.start, env.episode_log.positions, collision)

  result = {
    "trajectory": trajectory,
    "current_intensity": current_intensity,
    "current_label": current_label,
    "algorithm": "sac",
    "seed": seed,
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
    "discounted_Jc": discounted_sum(env.episode_log.step_costs, csac_cfg.gamma),
    "rmse_3d": rmse,
    "rmse_x": float(rmse_xyz[0]),
    "rmse_y": float(rmse_xyz[1]),
    "rmse_z": float(rmse_xyz[2]),
    "steps": env.step_counter,
    "cumulative_error": float(env.cumulative_error),
    "episode_distance_m": float(ep_distance_m),
    "distance_before_collision_m": dist_before_collision,
    "positions": [p.tolist() for p in env.episode_log.positions],
    "ref_positions": [p.tolist() for p in env.episode_log.ref_positions],
    "errors_xyz": [e.tolist() for e in env.episode_log.errors_xyz],
    "actions_phys": [a.tolist() for a in env.episode_log.actions_phys],
    "step_costs": env.episode_log.step_costs,
    "step_energy_components": env.episode_log.step_energy_components,
    "times": env.episode_log.times,
    "obstacle_centers": env.obstacles.centers.tolist(),
    "obstacle_radius": env.obstacles.radius,
  }
  return result


def run_batch_eval(
  trajectory: TrajectoryKind,
  seed: int,
  num_episodes: int = 20,
  checkpoint: Path | str | None = None,
  device: str | None = None,
  current_intensity: CurrentIntensity = 0,
  train_tag: str = TRAIN_TAG_DEFAULT,
  curriculum: bool = False,
  fast: bool = False,
) -> list[dict]:
  results = []
  for i in range(num_episodes):
    results.append(
      evaluate_episode(
        trajectory,
        seed,
        checkpoint,
        eval_seed=1000 + i,
        device=device,
        current_intensity=current_intensity,
        train_tag=train_tag,
        curriculum=curriculum,
        fast=fast,
      )
    )
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--trajectory", type=str, default="all", choices=["circle", "sinusoid", "figure8", "all"])
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--episodes", type=int, default=20)
  parser.add_argument("--checkpoint", type=str, default="")
  parser.add_argument(
    "--device",
    type=str,
    default="",
    choices=["", "cuda", "cpu", "auto", "cuda:0"],
    help="推理设备，默认 cuda",
  )
  parser.add_argument(
    "--current",
    type=str,
    default="0",
    help="洋流强度：0–3 或 none/mild/moderate/strong（须与训练时一致）",
  )
  parser.add_argument(
    "--fast",
    action="store_true",
    help="兼容旧 fast checkpoint（优先级低于 checkpoint 内置 cfg/结构推断）",
  )
  parser.add_argument("--train-tag", type=str, default=TRAIN_TAG_DEFAULT, help="训练实验目录名（为空则按旧目录）")
  parser.add_argument("--curriculum", action="store_true", help="从 checkpoints/<train-tag>/curriculum 加载")
  args = parser.parse_args()

  current_intensity = parse_current_intensity(args.current)
  current_label = CURRENT_LABELS[current_intensity]

  paths = PathConfig()
  paths.ensure()
  trajectories: list[TrajectoryKind] = (
    ["circle", "sinusoid", "figure8"] if args.trajectory == "all" else [args.trajectory]  # type: ignore
  )
  ckpt = Path(args.checkpoint) if args.checkpoint else None

  all_summary = {}
  for traj in trajectories:
    device = args.device or None
    results = run_batch_eval(
      traj,
      args.seed,
      args.episodes,
      ckpt,
      device=device,
      current_intensity=current_intensity,
      train_tag=args.train_tag,
      curriculum=args.curriculum,
      fast=args.fast,
    )
    out_dir = paths.traj_eval_log_dir(traj, "sac", current_label if current_intensity > 0 else None)
    out_path = out_dir / f"eval_rollouts_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    successes = [r["success"] for r in results]
    collisions = [r["collision"] for r in results]
    warning_counts = [r.get("collision_warning_count", 0) for r in results]
    rmses = [r["rmse_3d"] for r in results]
    energies = [r["undiscounted_energy"] for r in results]
    rewards = [r["total_reward"] for r in results]
    tracking_errors = [r["cumulative_error"] for r in results]
    coll_dist = [
      float(r["distance_before_collision_m"])
      for r in results
      if r.get("distance_before_collision_m") is not None
    ]
    clean_full_no_risk = [
      1.0
      if (r["steps"] >= SimConfig().episode_steps and not r["collision"] and r["reason"] != "deviate")
      else 0.0
      for r in results
    ]
    summary = {
      "algorithm": "sac",
      "trajectory": traj,
      "seed": args.seed,
      "num_episodes": args.episodes,
      "success_rate": float(np.mean(successes)),
      "collision_rate": float(np.mean(collisions)),
      "warning_count_mean": float(np.mean(warning_counts)),
      "warning_count_std": float(np.std(warning_counts)),
      "rmse_3d_mean": float(np.mean(rmses)),
      "rmse_3d_std": float(np.std(rmses)),
      "tracking_error_mean": float(np.mean(tracking_errors)),
      "tracking_error_std": float(np.std(tracking_errors)),
      "energy_mean": float(np.mean(energies)),
      "energy_std": float(np.std(energies)),
      "reward_mean": float(np.mean(rewards)),
      "reward_std": float(np.std(rewards)),
      "avg_episode_reward": float(np.mean(rewards)),
      "distance_before_collision_mean": float(np.mean(coll_dist)) if coll_dist else None,
      "clean_full_no_risk_rate": float(np.mean(clean_full_no_risk)),
    }
    all_summary[traj] = summary
    summary_path = out_dir / f"eval_summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

  combined = paths.eval_summary_all_path(args.seed, "sac")
  combined.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
