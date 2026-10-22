"""Evaluate a supplied CSAC circle checkpoint and plot its RMSE-ranked top three rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import plot_csac_circle_top3_rmse as figure
from config import SimConfig
from env import AUVTrackingCMDPEnv
from evaluate import _build_eval_cfg_from_checkpoint
from plot_seal_sac_sace_best_trajectories import InferenceActor
from utils import distance_before_collision_m, episode_distance_m


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "csac_sinusoid_no_current_ep2200_eval100_seed666" / "circle_seed666" / "csac_actor_seed666_best.pt"
OUT = ROOT / "figures" / "csac_circle_custom_top3" / "episodes1000"
LOG = ROOT / "logs" / "csac_circle_custom_top3" / "episodes1000" / "rollouts.json"
DEVICE = "cpu"


def run() -> list[dict]:
  cfg = _build_eval_cfg_from_checkpoint(CHECKPOINT, device=DEVICE)
  policy = InferenceActor(CHECKPOINT, DEVICE)
  results: list[dict] = []
  for index in range(1000):
    env = AUVTrackingCMDPEnv(sim_cfg=SimConfig(seed=666, trajectory="circle"), randomize_start=True)
    obs, done, info, reward_sum = env.reset(seed=1000 + index), False, None, 0.0
    while not done:
      obs, reward, _, done, info = env.step(policy.select_action(obs))
      reward_sum += reward
    errors = np.asarray(env.episode_log.errors_xyz)
    rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))) if len(errors) else float("nan")
    results.append({
      "eval_seed": 1000 + index, "success": bool(info.success), "collision": bool(info.collision), "reason": info.reason,
      "collision_warning_count": int(info.collision_warning_count), "d_endpoint": float(info.d_endpoint), "min_obstacle_surface": float(info.min_obstacle_surface),
      "rmse_3d": rmse, "cumulative_error": float(env.cumulative_error), "undiscounted_energy": float(env.episode_cost), "total_reward": float(reward_sum),
      "steps": env.step_counter, "episode_distance_m": float(episode_distance_m(env.start, env.episode_log.positions)),
      "distance_before_collision_m": distance_before_collision_m(env.start, env.episode_log.positions, bool(info.collision)),
      "positions": [point.tolist() for point in env.episode_log.positions], "ref_positions": [point.tolist() for point in env.episode_log.ref_positions],
      "errors_xyz": [error.tolist() for error in env.episode_log.errors_xyz], "times": env.episode_log.times,
      "obstacle_centers": env.obstacles.centers.tolist(), "obstacle_radius": env.obstacles.radius,
    })
    if (index + 1) % 25 == 0: print(f"circle {index + 1}/1000", flush=True)
  return results


def main() -> None:
  if LOG.exists(): results = json.loads(LOG.read_text(encoding="utf-8"))
  else:
    results = run(); LOG.parent.mkdir(parents=True, exist_ok=True); LOG.write_text(json.dumps(results), encoding="utf-8")
  top3 = sorted(enumerate(results), key=lambda item: figure.quality_key(item[1]))[:3]
  figure.OUT = OUT; figure.style(); items = figure.display_order(top3); figure.trajectory_plot(items); figure.error_plot(items)
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "selection.json").write_text(json.dumps([{ "label": label, "test_index": index + 1, "eval_seed": item["eval_seed"], "rmse_3d": item["rmse_3d"], "success": item["success"], "collision": item["collision"] } for index, item, label, _ in items], indent=2), encoding="utf-8")


if __name__ == "__main__": main()
