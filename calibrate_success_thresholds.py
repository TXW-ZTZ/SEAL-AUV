"""标定任务成功阈值：RMSE、累计误差、逐步 d_path 分位数。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from config import PathConfig, SimConfig, TrajectoryKind
from env import AUVTrackingCMDPEnv


def rollout_random(env: AUVTrackingCMDPEnv, episodes: int, seed: int) -> dict:
  rng = np.random.default_rng(seed)
  rmses = []
  cum_errors = []
  d_paths_all = []
  min_surfaces = []
  d_endpoints = []
  for _ in range(episodes):
    env.reset(seed=int(rng.integers(0, 1_000_000)))
    done = False
    while not done:
      action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
      _, _, _, done, _ = env.step(action)
    rmses.append(env._episode_rmse())
    cum_errors.append(env.cumulative_error)
    d_paths_all.append(
      np.linalg.norm(np.asarray(env.episode_log.errors_xyz, dtype=np.float64), axis=1)
    )
    min_surfaces.append(env.min_obstacle_surface)
    d_endpoints.append(env.d_endpoint)

  rmses_a = np.asarray(rmses)
  cum_a = np.asarray(cum_errors)
  d_flat = np.concatenate(d_paths_all) if d_paths_all else np.array([0.0])
  return {
    "label": "random_policy",
    "episodes": episodes,
    "rmse_3d": {
      "mean": float(rmses_a.mean()),
      "p50": float(np.percentile(rmses_a, 50)),
      "p75": float(np.percentile(rmses_a, 75)),
      "p90": float(np.percentile(rmses_a, 90)),
    },
    "cumulative_error": {
      "mean": float(cum_a.mean()),
      "p50": float(np.percentile(cum_a, 50)),
      "p75": float(np.percentile(cum_a, 75)),
      "p90": float(np.percentile(cum_a, 90)),
    },
    "d_path_per_step": {
      "p75": float(np.percentile(d_flat, 75)),
      "p90": float(np.percentile(d_flat, 90)),
      "p95": float(np.percentile(d_flat, 95)),
    },
    "min_obstacle_surface": {
      "min": float(np.min(min_surfaces)),
      "p10": float(np.percentile(min_surfaces, 10)),
    },
    "d_endpoint": {
      "mean": float(np.mean(d_endpoints)),
      "p50": float(np.percentile(d_endpoints, 50)),
    },
  }


def recommend_thresholds(
  stats: dict,
  *,
  cumulative_floor: float = 1200.0,
  rmse_cap: float = 3.5,
  rmse_floor: float = 2.0,
) -> dict:
  """从随机策略统计外推“可学习”阈值：比随机松、比理想跟踪严。"""
  rmse = stats["rmse_3d"]
  dps = stats["d_path_per_step"]
  rmse_rec = float(np.clip(rmse["p50"] * 0.10, rmse_floor, rmse_cap))
  step_lim = float(np.clip(dps["p75"] * 0.12, 3.5, 5.0))
  return {
    "rmse_success_threshold": round(rmse_rec, 3),
    "cumulative_error_threshold": cumulative_floor,
    "path_error_per_step_limit": round(step_lim, 2),
    "path_error_step_fraction": 0.90,
    "terminal_success_dist": 5.0,
    "min_obstacle_surface_clearance": 1.5,
    "source": (
      f"RMSE=clip(p50*0.10,{rmse_floor},{rmse_cap}); "
      f"cumulative=floor {cumulative_floor}; "
      f"per-step=clip(p75*0.12,3.5,5.0)"
    ),
  }


def update_config_thresholds(rec: dict, config_path: Path) -> bool:
  text = config_path.read_text(encoding="utf-8")
  replacements = [
    (r"(rmse_success_threshold:\s*float\s*=\s*)[0-9.]+", rec["rmse_success_threshold"]),
    (r"(cumulative_error_threshold:\s*float\s*=\s*)[0-9.]+", rec["cumulative_error_threshold"]),
    (r"(path_error_per_step_limit:\s*float\s*=\s*)[0-9.]+", rec["path_error_per_step_limit"]),
  ]
  for pattern, value in replacements:
    text, n = re.subn(pattern, rf"\g<1>{value}", text, count=1)
    if n == 0:
      return False
  config_path.write_text(text, encoding="utf-8")
  return True


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--trajectory", type=str, default="circle", choices=["circle", "sinusoid", "figure8"])
  parser.add_argument("--episodes", type=int, default=40)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--write-config", action="store_true")
  parser.add_argument("--cumulative-floor", type=float, default=1200.0)
  args = parser.parse_args()

  sim = SimConfig(trajectory=args.trajectory)  # type: ignore[arg-type]
  env = AUVTrackingCMDPEnv(sim_cfg=sim)
  stat = rollout_random(env, args.episodes, args.seed)
  rec = recommend_thresholds(stat, cumulative_floor=args.cumulative_floor)

  paths = PathConfig()
  paths.ensure()
  report = {"trajectory": args.trajectory, "stats": stat, "recommendation": rec}
  out = paths.log_dir / f"success_calibration_{args.trajectory}.json"
  out.write_text(json.dumps(report, indent=2), encoding="utf-8")
  print(json.dumps(report, indent=2, ensure_ascii=False))

  if args.write_config:
    ok = update_config_thresholds(rec, paths.root / "config.py")
    print(f"config.py updated: {ok}")


if __name__ == "__main__":
  main()
