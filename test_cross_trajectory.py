"""
跨轨迹泛化测试：加载在螺旋线 (circle) 上训练的 CSAC 策略，
在 circle / sinusoid / figure8 三种参考轨迹上分别评估并记录结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import SimConfig, PathConfig, TrajectoryKind, CurrentIntensity
from env import AUVTrackingCMDPEnv
from csac_agent import CSACAgent
from evaluate import _build_eval_cfg_from_checkpoint
from ocean_current import CURRENT_LABELS, parse_current_intensity
from utils import discounted_sum, episode_energy_fields

TRAIN_TRAJECTORY: TrajectoryKind = "circle"
TEST_TRAJECTORIES: list[TrajectoryKind] = ["circle", "sinusoid", "figure8"]
TRAIN_TAG_DEFAULT = "single_traj_curriculum"


def _checkpoint_dir(
  paths: PathConfig,
  train_trajectory: TrajectoryKind,
  seed: int,
  current_intensity: CurrentIntensity,
  curriculum: bool,
  train_tag: str = TRAIN_TAG_DEFAULT,
) -> Path:
  d = (paths.checkpoint_dir / train_tag) if train_tag else paths.traj_checkpoint_dir(train_trajectory)
  if curriculum:
    return d / "curriculum"
  if current_intensity > 0:
    return d / CURRENT_LABELS[current_intensity]
  return d


def evaluate_one_episode(
  agent: CSACAgent,
  test_trajectory: TrajectoryKind,
  train_seed: int,
  eval_seed: int,
  current_intensity: CurrentIntensity,
) -> dict:
  sim_cfg = SimConfig(
    seed=train_seed,
    trajectory=test_trajectory,
    current_intensity=current_intensity,
  )
  env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
  obs = env.reset(seed=eval_seed)
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

  return {
    "train_trajectory": TRAIN_TRAJECTORY,
    "test_trajectory": test_trajectory,
    "trajectory": test_trajectory,
    "current_intensity": current_intensity,
    "current_label": CURRENT_LABELS[current_intensity],
    "train_seed": train_seed,
    "eval_seed": eval_seed,
    "success": bool(info.success),
    "collision": bool(info.collision),
    "reason": info.reason,
    "d_endpoint": float(info.d_endpoint),
    "min_obstacle_surface": float(info.min_obstacle_surface),
    "total_reward": float(total_reward),
    "undiscounted_energy": float(env.episode_cost),
    **episode_energy_fields(env),
    "discounted_Jc": discounted_sum(env.episode_log.step_costs, agent.cfg.gamma),
    "rmse_3d": rmse,
    "rmse_x": float(rmse_xyz[0]),
    "rmse_y": float(rmse_xyz[1]),
    "rmse_z": float(rmse_xyz[2]),
    "steps": env.step_counter,
    "cumulative_error": float(env.cumulative_error),
    "positions": [p.tolist() for p in env.episode_log.positions],
    "ref_positions": [p.tolist() for p in env.episode_log.ref_positions],
    "errors_xyz": [e.tolist() for e in env.episode_log.errors_xyz],
    "actions_phys": [a.tolist() for a in env.episode_log.actions_phys],
    "step_costs": env.episode_log.step_costs,
    "step_energy_components": env.episode_log.step_energy_components,
    "times": env.episode_log.times,
    "obstacle_centers": env.obstacles.centers.tolist(),
    "obstacle_radius": env.obstacles.radius,
    "initial_position": env.start.tolist(),
    "initial_yaw_deg": float(np.rad2deg(env.x[5, 0])),
    "initial_pitch_deg": float(np.rad2deg(env.x[4, 0])),
  }


def run_cross_eval(
  train_seed: int,
  num_episodes: int = 20,
  checkpoint: Path | None = None,
  device: str | None = None,
  current_intensity: CurrentIntensity = 0,
  curriculum: bool = False,
  train_tag: str = TRAIN_TAG_DEFAULT,
  fast: bool = False,
) -> dict[str, list[dict]]:
  paths = PathConfig()
  paths.ensure()

  ckpt_root = _checkpoint_dir(
    paths, TRAIN_TRAJECTORY, train_seed, current_intensity, curriculum, train_tag
  )
  ckpt_path = checkpoint or ckpt_root / f"csac_actor_seed{train_seed}.pt"
  if not ckpt_path.exists() and checkpoint is None:
    stage_root = ckpt_root / f"{TRAIN_TRAJECTORY}_seed{train_seed}"
    for candidate in (
      stage_root / f"csac_actor_seed{train_seed}_best.pt",
      stage_root / f"csac_actor_seed{train_seed}.pt",
      ckpt_root / f"csac_actor_seed{train_seed}_best.pt",
    ):
      if candidate.exists():
        ckpt_path = candidate
        break
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
  csac_cfg = _build_eval_cfg_from_checkpoint(ckpt_path, device=device, fast=fast)

  agent = CSACAgent(csac_cfg, device=device)
  agent.load(ckpt_path)

  all_results: dict[str, list[dict]] = {}
  for test_traj in TEST_TRAJECTORIES:
    results = []
    for i in range(num_episodes):
      results.append(
        evaluate_one_episode(
          agent,
          test_traj,
          train_seed,
          eval_seed=10_000 + train_seed * 100 + i,
          current_intensity=current_intensity,
        )
      )
    all_results[test_traj] = results
  return all_results


def _summarize(results: list[dict]) -> dict:
  successes = [r["success"] for r in results]
  collisions = [r["collision"] for r in results]
  warning_counts = [r.get("collision_warning_count", 0) for r in results]
  tracking_errors = [r.get("cumulative_error", np.nan) for r in results]
  rmses = [r["rmse_3d"] for r in results]
  energies = [r["undiscounted_energy"] for r in results]
  return {
    "num_episodes": len(results),
    "success_rate": float(np.mean(successes)),
    "collision_rate": float(np.mean(collisions)),
    "warning_count_mean": float(np.mean(warning_counts)),
    "warning_count_std": float(np.std(warning_counts)),
    "tracking_error_mean": float(np.nanmean(tracking_errors)),
    "tracking_error_std": float(np.nanstd(tracking_errors)),
    "rmse_3d_mean": float(np.mean(rmses)),
    "rmse_3d_std": float(np.std(rmses)),
    "energy_mean": float(np.mean(energies)),
    "energy_std": float(np.std(energies)),
  }


def main():
  parser = argparse.ArgumentParser(description="跨轨迹泛化测试（训练=circle，测试=三种轨迹）")
  parser.add_argument("--seed", type=int, default=1, help="训练时使用的 checkpoint seed")
  parser.add_argument("--episodes", type=int, default=20)
  parser.add_argument("--checkpoint", type=str, default="")
  parser.add_argument("--device", type=str, default="", choices=["", "cuda", "cpu", "auto", "cuda:0"])
  parser.add_argument("--current", type=str, default="0", help="测试时洋流强度（建议与训练一致）")
  parser.add_argument("--curriculum", action="store_true", help="从 checkpoints/circle/curriculum/ 加载")
  parser.add_argument("--train-tag", type=str, default=TRAIN_TAG_DEFAULT, help="训练实验目录名（为空则按旧目录）")
  parser.add_argument(
    "--fast",
    action="store_true",
    help="兼容旧 fast checkpoint（优先级低于 checkpoint 内置 cfg/结构推断）",
  )
  args = parser.parse_args()

  current_intensity = parse_current_intensity(args.current)
  ckpt = Path(args.checkpoint) if args.checkpoint else None

  all_results = run_cross_eval(
    train_seed=args.seed,
    num_episodes=args.episodes,
    checkpoint=ckpt,
    device=args.device or None,
    current_intensity=current_intensity,
    curriculum=args.curriculum,
    train_tag=args.train_tag,
    fast=args.fast,
  )

  paths = PathConfig()
  out_root = paths.cross_eval_log_dir(TRAIN_TRAJECTORY, args.seed)
  all_summary = {}

  for test_traj, results in all_results.items():
    out_dir = out_root / test_traj
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "eval_rollouts.json"
    rollouts_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = _summarize(results)
    summary.update({
      "train_trajectory": TRAIN_TRAJECTORY,
      "test_trajectory": test_traj,
      "train_seed": args.seed,
      "current_intensity": current_intensity,
      "current_label": CURRENT_LABELS[current_intensity],
    })
    all_summary[test_traj] = summary
    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

  combined_path = out_root / "eval_summary_all.json"
  combined_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
  print(f"\nSaved cross-eval logs to {out_root}")


if __name__ == "__main__":
  main()
