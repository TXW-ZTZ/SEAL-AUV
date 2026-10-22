"""标定 CSAC 能耗预算 theta（折扣累计 Jc）。

两种标定来源：
1. SAC 训练日志（推荐）：从 episode_costs_seed*.jsonl 统计满程 episode 的 Jc
2. 随机策略 rollout（兜底）：短 episode，尺度与满程训练不一致，仅作参考
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from config import CSACConfig, PathConfig, SimConfig, TrajectoryKind
from env import AUVTrackingCMDPEnv
from utils import discounted_sum

SAC_TRAIN_TAG_DEFAULT = "single_traj_curriculum_sac"


def load_episode_costs(path: Path) -> list[dict]:
  rows: list[dict] = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  if not rows:
    raise ValueError(f"No episode records in {path}")
  return rows


def resolve_sac_log_path(
  paths: PathConfig,
  *,
  trajectory: TrajectoryKind,
  seed: int,
  train_tag: str,
) -> Path:
  log_path = (
    paths.log_dir / train_tag / f"{trajectory}_seed{seed}" / f"episode_costs_seed{seed}.jsonl"
  )
  if not log_path.exists():
    raise FileNotFoundError(
      f"SAC log not found: {log_path}\n"
      f"Train SAC first, e.g.:\n"
      f"  python train_sac.py --seed {seed} --stages <stage_id> --train-tag {train_tag}\n"
      f"  (circle=1, figure8=2, sinusoid=3)"
    )
  return log_path


def filter_sac_episodes(
  records: list[dict],
  *,
  full_horizon_only: bool,
  tail_frac: float,
  min_episodes: int,
) -> tuple[list[dict], dict]:
  """筛选用于标定的 SAC episode（默认：满程 + 训练末段）。"""
  meta: dict = {"input_episodes": len(records)}
  pool = records
  if full_horizon_only:
    pool = [r for r in pool if r.get("clean_full_episode")]
    meta["full_horizon_episodes"] = len(pool)
  if not pool:
    raise ValueError("No episodes left after filtering (try --no-full-horizon-only).")

  tail_frac = float(np.clip(tail_frac, 0.05, 1.0))
  tail_n = max(min_episodes, int(len(pool) * tail_frac))
  tail_n = min(tail_n, len(pool))
  subset = pool[-tail_n:]
  meta.update(
    {
      "used_episodes": len(subset),
      "tail_frac": tail_frac,
      "episode_range": [int(subset[0]["episode"]), int(subset[-1]["episode"])],
    }
  )
  return subset, meta


def summarize_jc(records: list[dict]) -> dict:
  jc = np.array([float(r["discounted_Jc"]) for r in records], dtype=np.float64)
  returns = np.array([float(r.get("episode_return", np.nan)) for r in records], dtype=np.float64)
  return {
    "discounted_Jc": {
      "mean": float(jc.mean()),
      "std": float(jc.std()),
      "p25": float(np.percentile(jc, 25)),
      "p50": float(np.percentile(jc, 50)),
      "p75": float(np.percentile(jc, 75)),
      "p90": float(np.percentile(jc, 90)),
      "min": float(jc.min()),
      "max": float(jc.max()),
    },
    "episode_return": {
      "mean": float(np.nanmean(returns)),
      "p50": float(np.nanpercentile(returns, 50)),
    },
  }


def _round_theta(value: float) -> float:
  return round(float(value), 1)


def recommend_theta_from_sac(stats: dict, *, curriculum_levels: int) -> dict:
  """
  由 SAC 满程 Jc 分位数构造 θ 与课程档位（单调递减）。

  最终 θ = p50（略严于 SAC 典型能耗，鼓励 CSAC 省能）
  课程首档 = p90 * 1.05（留余量，便于前期学跟踪）
  中间档在 [p75, p90] 与 [p50, p75] 间线性插值
  """
  jc = stats["discounted_Jc"]
  p50 = jc["p50"]
  p75 = jc["p75"]
  p90 = jc["p90"]

  final_theta = _round_theta(p50)
  loose = _round_theta(p90 * 1.05)

  n = max(2, int(curriculum_levels))
  if n == 1:
    levels = (final_theta,)
  elif n == 2:
    levels = (_round_theta(p75), final_theta)
  else:
    # 首档宽松，末档 p50，中间均匀插值
    anchors = np.linspace(loose, final_theta, n, dtype=np.float64)
    # 保证严格递减
    for i in range(1, len(anchors)):
      if anchors[i] >= anchors[i - 1]:
        anchors[i] = anchors[i - 1] - 0.5
    levels = tuple(_round_theta(v) for v in anchors)

  # 末档对齐 final_theta
  levels = tuple(list(levels[:-1]) + [final_theta])

  return {
    "recommended_theta": final_theta,
    "theta_curriculum_levels": levels,
    "source": "SAC full-horizon Jc percentiles (final=p50, loose=1.05*p90)",
    "reference_jc": {"p50": p50, "p75": p75, "p90": p90},
  }


def rollout_random(env: AUVTrackingCMDPEnv, episodes: int, seed: int, gamma: float) -> dict:
  rng = np.random.default_rng(seed)
  disc = []
  for _ in range(episodes):
    env.reset(seed=int(rng.integers(0, 1_000_000)))
    costs = []
    done = False
    while not done:
      action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
      _, _, cost, done, _ = env.step(action)
      costs.append(cost)
    disc.append(discounted_sum(costs, gamma))
  d = np.asarray(disc)
  return {
    "label": "random_policy",
    "episodes": len(d),
    "discounted_Jc": {
      "mean": float(d.mean()),
      "p50": float(np.percentile(d, 50)),
      "p75": float(np.percentile(d, 75)),
      "p90": float(np.percentile(d, 90)),
    },
  }


def recommend_theta_random(stats: list[dict], gamma: float) -> dict:
  random_jc = next(s["discounted_Jc"] for s in stats if s["label"] == "random_policy")
  theta = random_jc["p75"] * 0.85
  return {
    "recommended_theta": round(float(theta), 4),
    "theta_curriculum_levels": (round(float(theta * 1.2), 4), round(float(theta), 4)),
    "source": "0.85 * random_policy p75 (short episodes — not recommended for full-horizon training)",
    "gamma": gamma,
  }


def update_config_theta(theta: float, config_path: Path) -> bool:
  text = config_path.read_text(encoding="utf-8")
  new_text, n = re.subn(
    r"(energy_budget_theta:\s*float\s*=\s*)[0-9.]+",
    rf"\g<1>{theta}",
    text,
    count=1,
  )
  if n == 0:
    return False
  config_path.write_text(new_text, encoding="utf-8")
  return True


def update_config_theta_curriculum(levels: tuple[float, ...], config_path: Path) -> bool:
  text = config_path.read_text(encoding="utf-8")
  levels_str = ", ".join(f"{v:.1f}" if abs(v - round(v)) < 1e-6 else f"{v:g}" for v in levels)
  new_text, n = re.subn(
    r"(theta_curriculum_levels:\s*tuple\[float,\s*\.\.\.\]\s*=\s*)\([^)]*\)",
    rf"\g<1>({levels_str})",
    text,
    count=1,
  )
  if n == 0:
    return False
  config_path.write_text(new_text, encoding="utf-8")
  return True


def calibrate_from_sac_log(
  *,
  trajectory: TrajectoryKind,
  seed: int,
  train_tag: str,
  full_horizon_only: bool,
  tail_frac: float,
  min_episodes: int,
  curriculum_levels: int,
) -> dict:
  paths = PathConfig()
  log_path = resolve_sac_log_path(paths, trajectory=trajectory, seed=seed, train_tag=train_tag)
  records = load_episode_costs(log_path)
  subset, filter_meta = filter_sac_episodes(
    records,
    full_horizon_only=full_horizon_only,
    tail_frac=tail_frac,
    min_episodes=min_episodes,
  )
  stats = summarize_jc(subset)
  recommendation = recommend_theta_from_sac(stats, curriculum_levels=curriculum_levels)
  return {
    "method": "sac_training_log",
    "trajectory": trajectory,
    "seed": seed,
    "train_tag": train_tag,
    "log_path": str(log_path),
    "filter": filter_meta,
    "stats": stats,
    "recommendation": recommendation,
  }


def main():
  parser = argparse.ArgumentParser(description="标定 CSAC 能耗预算 θ")
  parser.add_argument(
    "--trajectory",
    type=str,
    default="sinusoid",
    choices=["circle", "sinusoid", "figure8"],
  )
  parser.add_argument("--seed", type=int, default=666)
  parser.add_argument(
    "--train-tag",
    type=str,
    default=SAC_TRAIN_TAG_DEFAULT,
    help="SAC 训练日志目录名（logs/<train-tag>/...）",
  )
  parser.add_argument(
    "--method",
    type=str,
    default="sac",
    choices=["sac", "random"],
    help="标定方法：sac=读 SAC 训练日志（推荐）；random=随机策略 rollout",
  )
  parser.add_argument(
    "--tail-frac",
    type=float,
    default=0.2,
    help="使用训练末段 episode 比例（SAC 方法，默认 0.2）",
  )
  parser.add_argument(
    "--min-episodes",
    type=int,
    default=50,
    help="SAC 统计最少 episode 数",
  )
  parser.add_argument(
    "--curriculum-levels",
    type=int,
    default=4,
    help="θ 课程档位数（SAC 方法，默认 4）",
  )
  parser.add_argument(
    "--no-full-horizon-only",
    action="store_true",
    help="SAC 方法：不过滤满程 episode（不推荐）",
  )
  parser.add_argument("--episodes", type=int, default=30, help="random 方法 rollout 回合数")
  parser.add_argument(
    "--write-config",
    action="store_true",
    help="写回 config.py 的 energy_budget_theta 与 theta_curriculum_levels",
  )
  args = parser.parse_args()

  paths = PathConfig()
  paths.ensure()
  cfg = CSACConfig()

  if args.method == "sac":
    report = calibrate_from_sac_log(
      trajectory=args.trajectory,  # type: ignore[arg-type]
      seed=args.seed,
      train_tag=args.train_tag,
      full_horizon_only=not args.no_full_horizon_only,
      tail_frac=args.tail_frac,
      min_episodes=args.min_episodes,
      curriculum_levels=args.curriculum_levels,
    )
  else:
    sim = SimConfig(trajectory=args.trajectory)  # type: ignore[arg-type]
    env = AUVTrackingCMDPEnv(sim_cfg=sim)
    stat = rollout_random(env, args.episodes, args.seed, cfg.gamma)
    rec = recommend_theta_random([stat], cfg.gamma)
    report = {
      "method": "random_policy",
      "trajectory": args.trajectory,
      "stats": [stat],
      "recommendation": rec,
    }

  out = paths.log_dir / f"theta_calibration_{args.trajectory}_seed{args.seed}.json"
  out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
  print(json.dumps(report, indent=2, ensure_ascii=False))
  print(f"\nSaved: {out}")

  rec = report["recommendation"]
  if args.write_config:
    config_path = paths.root / "config.py"
    ok_theta = update_config_theta(rec["recommended_theta"], config_path)
    ok_levels = update_config_theta_curriculum(
      tuple(rec["theta_curriculum_levels"]), config_path
    )
    print(
      f"config.py updated: energy_budget_theta={ok_theta}, "
      f"theta_curriculum_levels={ok_levels}"
    )
    print(f"  energy_budget_theta -> {rec['recommended_theta']}")
    print(f"  theta_curriculum_levels -> {rec['theta_curriculum_levels']}")
    stage_id = {"circle": 1, "figure8": 2, "sinusoid": 3}[args.trajectory]
    levels = ",".join(str(v) for v in rec["theta_curriculum_levels"])
    print("\nTrain CSAC with calibrated θ:")
    print(
      f"  python train_csac.py --seed {args.seed} --stages {stage_id} "
      f"--theta-levels {levels}"
    )


if __name__ == "__main__":
  main()
