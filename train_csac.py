"""CSAC training entry for staged single-trajectory curriculum."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Callable, TypeVar

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
from csac_agent import CSACAgent
from ocean_current import CURRENT_LABELS, parse_current_intensity
from trajectory import make_trajectory
from utils import discounted_sum, episode_distance_m, distance_before_collision_m


TRAIN_TAG_DEFAULT = "single_traj_curriculum"
T = TypeVar("T")


@dataclass(frozen=True)
class CurriculumScenario:
  trajectory: TrajectoryKind
  obstacle_num: int
  weight: float


@dataclass(frozen=True)
class StageSpec:
  stage_id: int
  name: str
  scenarios: tuple[CurriculumScenario, ...]


STAGE_SPECS: dict[int, StageSpec] = {
  1: StageSpec(1, "circle", (CurriculumScenario("circle", 3, 1.0),)),
  2: StageSpec(2, "figure8", (CurriculumScenario("figure8", 3, 1.0),)),
  3: StageSpec(3, "sinusoid", (CurriculumScenario("sinusoid", 3, 1.0),)),
  4: StageSpec(
    4,
    "all",
    (
      CurriculumScenario("circle", 3, 1.0 / 3.0),
      CurriculumScenario("figure8", 3, 1.0 / 3.0),
      CurriculumScenario("sinusoid", 3, 1.0 / 3.0),
    ),
  ),
}


def _parse_stage_ids(raw: str) -> list[int]:
  ids: list[int] = []
  for token in raw.split(","):
    token = token.strip()
    if not token:
      continue
    sid = int(token)
    if sid not in STAGE_SPECS:
      raise ValueError(f"Invalid stage id {sid}. Allowed: 1,2,3,4")
    ids.append(sid)
  if not ids:
    raise ValueError("At least one stage id is required.")

  out: list[int] = []
  seen: set[int] = set()
  for sid in ids:
    if sid not in seen:
      out.append(sid)
      seen.add(sid)
  return out


def _sample_training_scenario(stage: StageSpec, rng: random.Random) -> CurriculumScenario:
  return rng.choices(stage.scenarios, weights=[s.weight for s in stage.scenarios], k=1)[0]


def _set_env_scenario(env: AUVTrackingCMDPEnv, scenario: CurriculumScenario):
  if env.sim.trajectory != scenario.trajectory:
    env.sim.trajectory = scenario.trajectory
    env.traj = make_trajectory(scenario.trajectory, env.sim.los_lookahead)
  env.obs_cfg.obstacle_num = scenario.obstacle_num


def _eval_trajectories_for_stage(stage: StageSpec) -> tuple[TrajectoryKind, ...]:
  """训练中评估：单轨迹 stage 只评对应轨迹，stage4（混合）评全部。"""
  if len(stage.scenarios) == 1:
    return (stage.scenarios[0].trajectory,)
  return tuple(dict.fromkeys(s.trajectory for s in stage.scenarios))


def _is_cuda_oom(exc: Exception) -> bool:
  if not isinstance(exc, RuntimeError):
    return False
  msg = str(exc).lower()
  return "cuda out of memory" in msg or "cuda error: out of memory" in msg


def _run_with_cuda_oom_retry(fn: Callable[[], T], *, context: str, retries: int = 2) -> T:
  """Retry short CUDA ops after empty_cache for transient OOM spikes."""
  last_exc: RuntimeError | None = None
  for attempt in range(retries + 1):
    try:
      return fn()
    except RuntimeError as exc:
      if not _is_cuda_oom(exc) or not torch.cuda.is_available():
        raise
      last_exc = exc
      torch.cuda.empty_cache()
      time.sleep(0.15 * (attempt + 1))
  raise RuntimeError(
    f"{context} failed due to persistent CUDA OOM. "
    "Try --device cpu, reduce batch size, or close GPU-heavy apps."
  ) from last_exc


def _scheduled_lambda_actor_scale(cfg: CSACConfig, done_episodes: int) -> float:
  """λ 预热：前期 actor 不受能耗约束，后期线性放开。"""
  total_episodes = max(cfg.train_episodes, 1)
  frac = done_episodes / total_episodes
  start = float(np.clip(cfg.lambda_warmup_start_frac, 0.0, 1.0))
  end = float(np.clip(cfg.lambda_warmup_end_frac, 0.0, 1.0))
  end = max(end, start + 1e-6)
  if frac <= start:
    return 0.0
  if frac >= end:
    return 1.0
  ratio = (frac - start) / max(end - start, 1e-8)
  return float(np.clip(ratio, 0.0, 1.0))


def _scheduled_target_entropy_scale(cfg: CSACConfig, done_episodes: int) -> float:
  total_episodes = max(cfg.train_episodes, 1)
  frac = done_episodes / total_episodes
  start = float(np.clip(cfg.target_entropy_decay_start_frac, 0.0, 1.0))
  end = float(np.clip(cfg.target_entropy_decay_end_frac, 0.0, 1.0))
  end = max(end, start + 1e-6)
  if frac <= start:
    return float(cfg.target_entropy_scale)
  if frac >= end:
    return float(cfg.target_entropy_final_scale)
  ratio = (frac - start) / max(end - start, 1e-8)
  ratio = float(np.clip(ratio, 0.0, 1.0))
  return float(cfg.target_entropy_scale + ratio * (cfg.target_entropy_final_scale - cfg.target_entropy_scale))


def _update_exploration_schedule(agent: CSACAgent, cfg: CSACConfig):
  scale = _scheduled_target_entropy_scale(cfg, agent.episodes)
  agent.set_target_entropy_scale(scale)
  freeze_frac = float(cfg.alpha_freeze_start_frac)
  freeze = (
    freeze_frac < 1.0
    and (agent.episodes / max(cfg.train_episodes, 1)) >= freeze_frac
  )
  agent.set_alpha_frozen(freeze)


def _apply_current_level(env: AUVTrackingCMDPEnv, intensity: CurrentIntensity) -> str:
  label = CURRENT_LABELS[intensity]
  if env.ocean_current.intensity != intensity:
    env.ocean_current.set_intensity(intensity)
    env.ocean_current.reset(env.rng)
  return label


def _episode_distance_m(env: AUVTrackingCMDPEnv) -> float:
  if env.record_episode_log:
    return episode_distance_m(env.start, env.episode_log.positions)
  return float(env.episode_distance_m)


def _distance_before_collision_from_env(env: AUVTrackingCMDPEnv, collision: bool) -> float | None:
  if not collision:
    return None
  if env.record_episode_log:
    return distance_before_collision_m(env.start, env.episode_log.positions, collision)
  return env.distance_before_collision_m


def _flush_cost_jsonl(path: Path, records: list[dict]) -> None:
  if not records:
    path.write_text("", encoding="utf-8")
    return
  path.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
    encoding="utf-8",
  )


def _theta_quarter_boundary(cfg: CSACConfig, theta_idx: int) -> int:
  """θ 从 theta_idx 晋级到下一档所需的最小 episode 数（每 1/4 总 episode 允许降一档）。"""
  levels = cfg.theta_curriculum_levels
  if not cfg.theta_curriculum_enabled or len(levels) <= 1:
    return cfg.train_episodes + 1
  quarter = max(1, cfg.train_episodes // 4)
  return (theta_idx + 1) * quarter


def _init_theta_curriculum(cfg: CSACConfig) -> tuple[int, float, tuple[float, ...], float | None]:
  levels = tuple(float(t) for t in cfg.theta_curriculum_levels)
  if cfg.theta_curriculum_enabled and len(levels) >= 1:
    theta_idx = 0
    theta_ref, active_theta = _resolve_active_theta((), theta_idx, levels, cfg)
    cfg.energy_budget_theta = active_theta
    return theta_idx, active_theta, levels, theta_ref
  active_theta = float(cfg.energy_budget_theta)
  cfg.energy_budget_theta = active_theta
  return 0, active_theta, (active_theta,), None


def _resolve_active_theta(
  jc_window: tuple[float, ...] | deque[float],
  theta_idx: int,
  levels: tuple[float, ...],
  cfg: CSACConfig,
) -> tuple[float | None, float]:
  if not cfg.theta_curriculum_enabled:
    theta = float(cfg.energy_budget_theta)
    return None, theta

  idx = int(np.clip(theta_idx, 0, len(levels) - 1))
  if cfg.theta_curriculum_mode == "relative":
    min_samples = max(1, int(cfg.theta_ref_min_samples))
    if len(jc_window) < min_samples:
      return None, float("inf")
    theta_ref = float(np.median(np.asarray(jc_window, dtype=np.float64)))
    active_theta = float(theta_ref * levels[idx])
    return theta_ref, active_theta

  active_theta = float(levels[idx])
  return None, active_theta


def _jc_satisfaction_rate(jc_window: deque[float], active_theta: float) -> float:
  """满程 episode 窗口内 Jc <= active_theta 的比例。"""
  if not jc_window:
    return 0.0
  return float(np.mean([1.0 if j <= active_theta else 0.0 for j in jc_window]))


def _append_full_horizon_jc(jc_window: deque[float], ep_jc: float, *, clean_full_episode: bool) -> None:
  """θ 课程晋级仅统计满程 episode 的 Jc，排除 deviate/碰撞早停。"""
  if clean_full_episode:
    jc_window.append(ep_jc)


def _maybe_advance_theta_curriculum(
  *,
  agent_episodes: int,
  theta_idx: int,
  active_theta: float,
  jc_window: deque[float],
  clean_full_rate: float,
  success_rate: float,
  cfg: CSACConfig,
  levels: tuple[float, ...],
) -> tuple[int, float, bool]:
  if not cfg.theta_curriculum_enabled or theta_idx >= len(levels) - 1:
    return theta_idx, active_theta, False
  if not math.isfinite(active_theta):
    return theta_idx, active_theta, False
  wlen = max(1, int(cfg.curriculum_window_episodes))
  if agent_episodes < _theta_quarter_boundary(cfg, theta_idx):
    return theta_idx, active_theta, False
  if len(jc_window) < wlen:
    return theta_idx, active_theta, False
  if cfg.theta_curriculum_mode == "relative" and len(jc_window) < max(1, int(cfg.theta_ref_min_samples)):
    return theta_idx, active_theta, False

  clean_threshold = float(np.clip(cfg.theta_curriculum_clean_full_rate_threshold, 0.0, 1.0))
  success_threshold = float(np.clip(cfg.theta_curriculum_success_rate_threshold, 0.0, 1.0))
  if clean_full_rate < clean_threshold or success_rate < success_threshold:
    return theta_idx, active_theta, False

  satisfaction = _jc_satisfaction_rate(jc_window, active_theta)
  threshold = float(np.clip(cfg.theta_satisfaction_rate_threshold, 0.0, 1.0))
  if satisfaction < threshold:
    return theta_idx, active_theta, False
  new_idx = theta_idx + 1
  _, new_theta = _resolve_active_theta(jc_window, new_idx, levels, cfg)
  cfg.energy_budget_theta = new_theta
  return new_idx, new_theta, True


def _mean_collision_distance(records: list[dict], window: int | None = None) -> float | None:
  subset = records[-window:] if window is not None and window > 0 else records
  vals = [
    float(r["distance_before_collision_m"])
    for r in subset
    if r.get("collision") and r.get("distance_before_collision_m") is not None
  ]
  if not vals:
    return None
  return float(np.mean(vals))


def _write_cost_summary(log_dir: Path, seed: int, stage: StageSpec, records: list[dict], cfg: CSACConfig):
  if not records:
    return
  jc = np.array([r["discounted_Jc"] for r in records], dtype=np.float64)
  collisions = np.array([1.0 if r["collision"] else 0.0 for r in records], dtype=np.float64)
  returns = np.array([r["episode_return"] for r in records], dtype=np.float64)
  warning_counts = np.array([r.get("collision_warning_count", 0.0) for r in records], dtype=np.float64)
  warning_per_meter = np.array([r.get("warning_per_meter", 0.0) for r in records], dtype=np.float64)
  rmse = np.array([r.get("rmse_3d", np.nan) for r in records], dtype=np.float64)
  tracking_err = np.array([r.get("tracking_error_sum", np.nan) for r in records], dtype=np.float64)
  clean_full = np.array([1.0 if r.get("clean_full_episode", False) else 0.0 for r in records], dtype=np.float64)
  successes = np.array([1.0 if r.get("success", False) else 0.0 for r in records], dtype=np.float64)
  w = max(1, int(cfg.curriculum_window_episodes))
  tail_k = max(1, int(len(records) * 0.2))
  coll_dist_mean = _mean_collision_distance(records)
  coll_dist_last_w = _mean_collision_distance(records, w)
  summary = {
    "seed": seed,
    "stage_id": stage.stage_id,
    "stage_name": stage.name,
    "episodes": len(records),
    "theta": cfg.energy_budget_theta,
    "curriculum_thresholds": {
      "window_episodes": w,
      "clean_full_rate_threshold": cfg.curriculum_clean_full_rate_threshold,
      "current_levels": list(cfg.curriculum_current_levels),
    },
    "collision_rate": float(collisions.mean()),
    "avg_episode_reward": float(returns.mean()),
    "episode_return": {
      "mean": float(returns.mean()),
      "last_window_mean": float(returns[-w:].mean()),
      "last_stage_mean": float(returns[-tail_k:].mean()),
    },
    "distance_before_collision_m": {
      "mean_collision_episodes": coll_dist_mean,
      "last_window_mean": coll_dist_last_w,
    },
    "collision_warning_count": {
      "mean": float(warning_counts.mean()),
      "last_window_mean": float(warning_counts[-w:].mean()),
    },
    "warning_per_meter": {
      "mean": float(warning_per_meter.mean()),
      "last_window_mean": float(warning_per_meter[-w:].mean()),
    },
    "rmse_3d": {
      "mean": float(np.nanmean(rmse)),
      "last_window_mean": float(np.nanmean(rmse[-w:])),
    },
    "tracking_error_sum": {
      "mean": float(np.nanmean(tracking_err)),
      "last_window_mean": float(np.nanmean(tracking_err[-w:])),
    },
    "clean_full_episode_rate": {
      "mean": float(clean_full.mean()),
      "last_window_mean": float(clean_full[-w:].mean()),
    },
    "success_rate": {
      "mean": float(successes.mean()),
      "last_window_mean": float(successes[-w:].mean()),
    },
    "discounted_Jc": {
      "mean": float(jc.mean()),
      "p50": float(np.percentile(jc, 50)),
      "p75": float(np.percentile(jc, 75)),
      "max": float(jc.max()),
    },
    "violation_rate": float(
      np.mean([
        r.get("theta") is not None and r["discounted_Jc"] > r["theta"]
        for r in records
      ])
    ),
    "theta_curriculum": {
      "enabled": cfg.theta_curriculum_enabled,
      "mode": cfg.theta_curriculum_mode,
      "levels": list(cfg.theta_curriculum_levels),
      "satisfaction_rate_threshold": cfg.theta_satisfaction_rate_threshold,
      "clean_full_rate_threshold": cfg.theta_curriculum_clean_full_rate_threshold,
      "success_rate_threshold": cfg.theta_curriculum_success_rate_threshold,
      "full_horizon_only": True,
      "final_theta": float(cfg.energy_budget_theta),
    },
    "lambda_warmup": {
      "start_frac": cfg.lambda_warmup_start_frac,
      "end_frac": cfg.lambda_warmup_end_frac,
    },
  }
  out = log_dir / f"episode_cost_summary_seed{seed}.json"
  out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_eval_snapshot(agent: CSACAgent, ckpt_dir: Path, seed: int, episode: int) -> Path:
  eval_ckpt_dir = ckpt_dir / "eval_snapshots"
  eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
  ckpt_path = eval_ckpt_dir / f"csac_actor_seed{seed}_ep{episode:04d}.pt"
  torch.save(
    {
      "actor": agent.actor.state_dict(),
      "qr1": agent.qr1.state_dict(),
      "qc1": agent.qc1.state_dict(),
      "log_lambda": agent.log_lambda.detach().cpu(),
      "cfg": agent.cfg,
      "episode": int(episode),
      "total_step": int(agent.total_step),
    },
    ckpt_path,
  )
  return ckpt_path


def _evaluate_policy_once(
  agent: CSACAgent,
  trajectory: TrajectoryKind,
  eval_seed: int,
  current_intensity: CurrentIntensity,
) -> dict:
  sim_cfg = SimConfig(seed=eval_seed, trajectory=trajectory, current_intensity=current_intensity)
  env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)

  obs = env.reset(seed=eval_seed)
  total_reward = 0.0
  total_cost = 0.0
  done = False
  info = None
  with torch.inference_mode():
    while not done:
      action = _run_with_cuda_oom_retry(
        lambda: agent.select_action(obs, random_explore=False, deterministic=True),
        context="Evaluation select_action",
      )
      obs, reward, cost, done, info = env.step(action)
      total_reward += float(reward)
      total_cost += float(cost)

  if info is None:
    raise RuntimeError("Evaluation failed: env info is missing at episode end.")
  collision = bool(info.collision)
  ep_distance_m = episode_distance_m(env.start, env.episode_log.positions)
  dist_before_collision = distance_before_collision_m(env.start, env.episode_log.positions, collision)
  return {
    "reward": float(total_reward),
    "success": bool(info.success),
    "collision": collision,
    "rmse_3d": float(info.rmse_3d),
    "tracking_error_sum": float(env.cumulative_error),
    "undiscounted_energy": float(env.episode_cost if env.episode_cost > 0.0 else total_cost),
    "discounted_Jc": discounted_sum(env.episode_log.step_costs, agent.cfg.gamma),
    "min_obstacle_surface": float(info.min_obstacle_surface),
    "episode_distance_m": float(ep_distance_m),
    "distance_before_collision_m": dist_before_collision,
  }


def _evaluate_agent_during_train(
  agent: CSACAgent,
  *,
  seed: int,
  episode: int,
  total_step: int,
  cfg: CSACConfig,
  current_intensity: CurrentIntensity,
  checkpoint: str,
  eval_trajectories: tuple[TrajectoryKind, ...],
) -> dict:
  """使用当前内存中的 agent 做阶段内轨迹评估（不重新实例化/加载 checkpoint）。"""
  actor_was_training = agent.actor.training
  agent.actor.eval()
  try:
    eval_seeds = tuple(int(s) for s in cfg.eval_seeds)
    traj_stats: dict[str, dict] = {}
    reward_pool: list[float] = []
    success_pool: list[float] = []
    collision_pool: list[float] = []
    rmse_pool: list[float] = []
    tracking_pool: list[float] = []
    energy_pool: list[float] = []
    jc_pool: list[float] = []
    coll_dist_pool: list[float] = []
    worst_traj_reward = math.inf
    worst_traj_success = math.inf
    worst_traj_collision = 0.0
    worst_traj_rmse = 0.0
    worst_traj_tracking = 0.0
    worst_traj_energy = 0.0
    worst_traj_jc = 0.0
    for traj in eval_trajectories:
      traj_rewards: list[float] = []
      traj_success: list[float] = []
      traj_collision: list[float] = []
      traj_rmse: list[float] = []
      traj_tracking: list[float] = []
      traj_energy: list[float] = []
      traj_jc: list[float] = []
      traj_coll_dist: list[float] = []
      for eval_seed in eval_seeds:
        out = _evaluate_policy_once(agent, traj, eval_seed, current_intensity)
        traj_rewards.append(out["reward"])
        traj_success.append(1.0 if out["success"] else 0.0)
        traj_collision.append(1.0 if out["collision"] else 0.0)
        traj_rmse.append(out["rmse_3d"])
        traj_tracking.append(out["tracking_error_sum"])
        traj_energy.append(out["undiscounted_energy"])
        traj_jc.append(out["discounted_Jc"])
        if out["distance_before_collision_m"] is not None:
          traj_coll_dist.append(float(out["distance_before_collision_m"]))
          coll_dist_pool.append(float(out["distance_before_collision_m"]))
      reward_mean = float(np.mean(traj_rewards))
      success_rate = float(np.mean(traj_success))
      collision_rate = float(np.mean(traj_collision))
      rmse_mean = float(np.mean(traj_rmse))
      tracking_mean = float(np.mean(traj_tracking))
      energy_mean = float(np.mean(traj_energy))
      jc_mean = float(np.mean(traj_jc))
      worst_traj_reward = min(worst_traj_reward, reward_mean)
      worst_traj_success = min(worst_traj_success, success_rate)
      worst_traj_collision = max(worst_traj_collision, collision_rate)
      worst_traj_rmse = max(worst_traj_rmse, rmse_mean)
      worst_traj_tracking = max(worst_traj_tracking, tracking_mean)
      worst_traj_energy = max(worst_traj_energy, energy_mean)
      worst_traj_jc = max(worst_traj_jc, jc_mean)
      traj_stats[traj] = {
        "reward_mean": reward_mean,
        "reward_std": float(np.std(traj_rewards)),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "rmse_3d_mean": rmse_mean,
        "tracking_error_mean": tracking_mean,
        "energy_mean": energy_mean,
        "energy_std": float(np.std(traj_energy)),
        "discounted_Jc_mean": jc_mean,
        "discounted_Jc_std": float(np.std(traj_jc)),
        "distance_before_collision_mean": float(np.mean(traj_coll_dist)) if traj_coll_dist else None,
        "num_eval_episodes": len(eval_seeds),
      }
      reward_pool.extend(traj_rewards)
      success_pool.extend(traj_success)
      collision_pool.extend(traj_collision)
      rmse_pool.extend(traj_rmse)
      tracking_pool.extend(traj_tracking)
      energy_pool.extend(traj_energy)
      jc_pool.extend(traj_jc)

    return {
      "seed": seed,
      "episode": int(episode),
      "total_step": int(total_step),
      "checkpoint": checkpoint,
      "eval_trajectories": list(eval_trajectories),
      "eval_seeds": list(eval_seeds),
      "by_trajectory": traj_stats,
      "overall": {
        "reward_mean": float(np.mean(reward_pool)),
        "reward_std": float(np.std(reward_pool)),
        "avg_episode_reward": float(np.mean(reward_pool)),
        "success_rate": float(np.mean(success_pool)),
        "collision_rate": float(np.mean(collision_pool)),
        "rmse_3d_mean": float(np.mean(rmse_pool)),
        "tracking_error_mean": float(np.mean(tracking_pool)),
        "energy_mean": float(np.mean(energy_pool)),
        "discounted_Jc_mean": float(np.mean(jc_pool)),
        "distance_before_collision_mean": float(np.mean(coll_dist_pool)) if coll_dist_pool else None,
        "num_eval_episodes": len(reward_pool),
      },
      "selection_metric": {
        "worst_traj_reward_mean": float(worst_traj_reward),
        "worst_traj_success_rate": float(worst_traj_success),
        "worst_traj_collision_rate": float(worst_traj_collision),
        "worst_traj_rmse_3d_mean": float(worst_traj_rmse),
        "worst_traj_tracking_error_mean": float(worst_traj_tracking),
        "worst_traj_energy_mean": float(worst_traj_energy),
        "worst_traj_discounted_Jc_mean": float(worst_traj_jc),
        "overall_reward_mean": float(np.mean(reward_pool)),
        "overall_success_rate": float(np.mean(success_pool)),
        "overall_collision_rate": float(np.mean(collision_pool)),
        "overall_rmse_3d_mean": float(np.mean(rmse_pool)),
        "overall_tracking_error_mean": float(np.mean(tracking_pool)),
        "overall_energy_mean": float(np.mean(energy_pool)),
        "overall_discounted_Jc_mean": float(np.mean(jc_pool)),
        "policy_selection": "safety_tracking_energy_v2",
      },
    }
  finally:
    if actor_was_training:
      agent.actor.train()


def _is_better_eval_record(current: dict, best: dict | None) -> bool:
  if best is None:
    return True
  cur = current["selection_metric"]
  old = best["selection_metric"]
  eps = 1e-9

  def val(src: dict, key: str, default: float) -> float:
    raw = src.get(key, default)
    try:
      out = float(raw)
    except (TypeError, ValueError):
      return default
    return out if math.isfinite(out) else default

  for key in ("worst_traj_collision_rate", "overall_collision_rate"):
    cur_v = val(cur, key, cur.get("overall_collision_rate", math.inf))
    old_v = val(old, key, old.get("overall_collision_rate", math.inf))
    if cur_v < old_v - eps:
      return True
    if old_v < cur_v - eps:
      return False

  for key in ("worst_traj_success_rate", "overall_success_rate"):
    cur_v = val(cur, key, cur.get("overall_success_rate", -math.inf))
    old_v = val(old, key, old.get("overall_success_rate", -math.inf))
    if cur_v > old_v + eps:
      return True
    if old_v > cur_v + eps:
      return False

  lower_is_better = (
    "worst_traj_rmse_3d_mean",
    "overall_rmse_3d_mean",
    "worst_traj_tracking_error_mean",
    "overall_tracking_error_mean",
    "worst_traj_energy_mean",
    "overall_energy_mean",
    "worst_traj_discounted_Jc_mean",
    "overall_discounted_Jc_mean",
  )
  for key in lower_is_better:
    cur_v = val(cur, key, math.inf)
    old_v = val(old, key, math.inf)
    if cur_v < old_v - eps:
      return True
    if old_v < cur_v - eps:
      return False

  for key in ("worst_traj_reward_mean", "overall_reward_mean"):
    cur_v = val(cur, key, -math.inf)
    old_v = val(old, key, -math.inf)
    if cur_v > old_v + eps:
      return True
    if old_v > cur_v + eps:
      return False
  return False


def _stage_output_dirs(paths: PathConfig, train_tag: str, stage: StageSpec, seed: int) -> tuple[Path, Path]:
  run_dir = f"{stage.name}_seed{seed}"
  ckpt_dir = paths.checkpoint_dir / train_tag / run_dir
  log_dir = paths.log_dir / train_tag / run_dir
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)
  return ckpt_dir, log_dir


def train_one_stage(
  seed: int,
  stage: StageSpec,
  csac_cfg: CSACConfig | None = None,
  train_tag: str = TRAIN_TAG_DEFAULT,
):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)

  csac_cfg = csac_cfg or CSACConfig()
  paths = PathConfig()
  paths.ensure()
  ckpt_dir, log_dir = _stage_output_dirs(paths, train_tag, stage, seed)

  stage_rng = random.Random(seed + 2026 + stage.stage_id * 1000)
  scenario = _sample_training_scenario(stage, stage_rng)
  current_levels = tuple(int(v) for v in csac_cfg.curriculum_current_levels)
  if len(current_levels) < 1:
    raise ValueError("curriculum_current_levels must contain at least one level.")
  current_levels = tuple(int(np.clip(v, 0, 3)) for v in current_levels)

  sim_cfg = SimConfig(seed=seed, trajectory=scenario.trajectory, current_intensity=current_levels[0])  # type: ignore[arg-type]
  env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True, record_episode_log=False)
  _set_env_scenario(env, scenario)

  theta_idx, active_theta, theta_levels, theta_ref = _init_theta_curriculum(csac_cfg)
  agent = CSACAgent(csac_cfg)
  train_start = time.perf_counter()
  last_report_step = 0

  max_train_steps = max(csac_cfg.train_frames, csac_cfg.train_episodes * sim_cfg.episode_steps)
  phase_idx = 0
  active_current = int(current_levels[phase_idx])  # type: ignore[assignment]
  active_label = _apply_current_level(env, active_current)  # type: ignore[arg-type]
  obs = env.reset(seed=seed)

  total_steps = 0
  ep_reward = 0.0
  ep_cost = 0.0
  ep_step_costs: list[float] = []
  traj_cache: list[np.ndarray] = []
  cost_records: list[dict] = []
  wlen = max(1, int(csac_cfg.curriculum_window_episodes))
  clean_rate_threshold = float(np.clip(csac_cfg.curriculum_clean_full_rate_threshold, 0.0, 1.0))
  collision_window: deque[int] = deque(maxlen=wlen)
  warning_count_window: deque[int] = deque(maxlen=wlen)
  tracking_error_window: deque[float] = deque(maxlen=wlen)
  clean_full_episode_window: deque[int] = deque(maxlen=wlen)
  success_window: deque[int] = deque(maxlen=wlen)
  reward_window: deque[float] = deque(maxlen=wlen)
  rmse_window: deque[float] = deque(maxlen=wlen)
  jc_window: deque[float] = deque(maxlen=wlen)
  cost_log_path = log_dir / f"episode_costs_seed{seed}.jsonl"
  eval_log_path = log_dir / f"eval_during_train_seed{seed}.jsonl"
  best_eval_path = log_dir / f"best_eval_model_seed{seed}.json"
  stage_meta_path = log_dir / "stage_meta.json"
  stage_eval_trajectories = _eval_trajectories_for_stage(stage)
  stage_meta_path.write_text(
    json.dumps(
      {
        "stage_id": stage.stage_id,
        "stage_name": stage.name,
        "seed": seed,
        "train_tag": train_tag,
        "scenarios": [
          {
            "trajectory": s.trajectory,
            "obstacle_num": s.obstacle_num,
            "weight": s.weight,
          }
          for s in stage.scenarios
        ],
        "curriculum_current_levels": list(current_levels),
        "clean_full_rate_threshold": clean_rate_threshold,
        "window_episodes": wlen,
        "eval_trajectories": list(stage_eval_trajectories),
        "theta_curriculum_enabled": csac_cfg.theta_curriculum_enabled,
        "theta_curriculum_mode": csac_cfg.theta_curriculum_mode,
        "theta_curriculum_levels": list(theta_levels),
        "theta_ref_min_samples": csac_cfg.theta_ref_min_samples,
        "theta_satisfaction_rate_threshold": csac_cfg.theta_satisfaction_rate_threshold,
        "theta_curriculum_clean_full_rate_threshold": csac_cfg.theta_curriculum_clean_full_rate_threshold,
        "theta_curriculum_success_rate_threshold": csac_cfg.theta_curriculum_success_rate_threshold,
        "theta_curriculum_replay_trim_frac": csac_cfg.theta_curriculum_replay_trim_frac,
        "theta_curriculum_full_horizon_only": True,
        "initial_theta": active_theta,
        "lambda_warmup_start_frac": csac_cfg.lambda_warmup_start_frac,
        "lambda_warmup_end_frac": csac_cfg.lambda_warmup_end_frac,
      },
      indent=2,
      ensure_ascii=False,
    ),
    encoding="utf-8",
  )
  if eval_log_path.exists():
    eval_log_path.unlink()
  if cost_log_path.exists():
    cost_log_path.unlink()
  cost_log_flush = max(1, int(csac_cfg.cost_log_flush_episodes))
  train_print_every = max(0, int(csac_cfg.train_log_print_episodes))
  episodes_at_current_level = 0
  best_eval_record: dict | None = None

  while agent.episodes < csac_cfg.train_episodes and total_steps < max_train_steps:
    total_steps += 1
    agent.total_step = total_steps
    _update_exploration_schedule(agent, csac_cfg)
    lambda_actor_scale = _scheduled_lambda_actor_scale(csac_cfg, agent.episodes)
    agent.set_lambda_actor_scale(lambda_actor_scale)
    explore = total_steps < csac_cfg.initial_random_steps
    action = _run_with_cuda_oom_retry(
      lambda: agent.select_action(obs, random_explore=explore),
      context="Training select_action",
    )
    next_obs, reward, cost, done, info = env.step(action)
    agent.replay.store(obs, info.action_exec_norm, reward, cost, next_obs, done)
    ep_reward += reward
    ep_cost += cost
    ep_step_costs.append(cost)
    if agent.episodes >= 499:
      traj_cache.append(env.x[:3, 0].copy())
    obs = next_obs

    if (
      len(agent.replay) >= csac_cfg.batch_size
      and total_steps > csac_cfg.update_after
      and total_steps % csac_cfg.update_every == 0
    ):
      for _ in range(csac_cfg.utd_ratio):
        _run_with_cuda_oom_retry(
          lambda: agent.update(),
          context="Training update",
        )

    if total_steps - last_report_step >= 20_000:
      elapsed = time.perf_counter() - train_start
      sps = total_steps / max(elapsed, 1e-9)
      eta_h = (max_train_steps - total_steps) / max(sps, 1e-9) / 3600.0
      print(
        f"[speed/{stage.name}] step={total_steps}/{max_train_steps} "
        f"{sps:.1f} steps/s | ETA ~{eta_h:.1f} h | "
        f"update_every={csac_cfg.update_every} utd={csac_cfg.utd_ratio} amp={csac_cfg.use_amp}"
      )
      last_report_step = total_steps

    if not done:
      continue

    agent.episodes += 1
    collision = bool(info.collision)
    ep_success = bool(info.success)
    ep_tracking_error = float(env.cumulative_error)
    ep_warning_count = int(info.collision_warning_count)
    ep_rmse = float(info.rmse_3d)
    ep_reward_per_step = ep_reward / max(env.step_counter, 1)
    ep_distance_m = _episode_distance_m(env)
    ep_dist_before_collision = _distance_before_collision_from_env(env, collision)
    ep_warning_per_meter = ep_warning_count / max(ep_distance_m, 1e-6)
    clean_full_episode = bool(env.step_counter >= env.sim.episode_steps and not collision and info.reason != "deviate")
    ep_jc = discounted_sum(ep_step_costs, csac_cfg.gamma)

    collision_window.append(1 if collision else 0)
    warning_count_window.append(ep_warning_count)
    tracking_error_window.append(ep_tracking_error)
    clean_full_episode_window.append(1 if clean_full_episode else 0)
    success_window.append(1 if ep_success else 0)
    reward_window.append(ep_reward)
    rmse_window.append(ep_rmse)
    _append_full_horizon_jc(jc_window, ep_jc, clean_full_episode=clean_full_episode)

    clean_full_no_risk_rate_last_w = float(np.mean(clean_full_episode_window))
    success_rate_last_w = float(np.mean(success_window))
    avg_warning_count_last_w = float(np.mean(warning_count_window))
    avg_tracking_error_last_w = float(np.mean(tracking_error_window))
    avg_reward_last_w = float(np.mean(reward_window))
    avg_rmse_last_w = float(np.mean(rmse_window))
    collision_count_last_w = int(sum(collision_window))

    theta_ref, active_theta = _resolve_active_theta(jc_window, theta_idx, theta_levels, csac_cfg)
    csac_cfg.energy_budget_theta = active_theta
    jc_satisfaction_rate_last_w = _jc_satisfaction_rate(jc_window, active_theta)
    jc_full_horizon_window_len = len(jc_window)

    if (
      total_steps > csac_cfg.initial_random_steps
      and lambda_actor_scale > 0.0
      and math.isfinite(active_theta)
    ):
      agent.update_lambda_from_episode(ep_jc)

    new_theta_idx, new_theta, theta_advanced = _maybe_advance_theta_curriculum(
      agent_episodes=agent.episodes,
      theta_idx=theta_idx,
      active_theta=active_theta,
      jc_window=jc_window,
      clean_full_rate=clean_full_no_risk_rate_last_w,
      success_rate=success_rate_last_w,
      cfg=csac_cfg,
      levels=theta_levels,
    )
    if theta_advanced:
      old_theta = active_theta
      theta_idx = new_theta_idx
      active_theta = new_theta
      theta_ref, active_theta = _resolve_active_theta(jc_window, theta_idx, theta_levels, csac_cfg)
      csac_cfg.energy_budget_theta = active_theta
      jc_window.clear()
      jc_satisfaction_rate_last_w = 0.0
      jc_full_horizon_window_len = 0
      trim_frac = float(np.clip(csac_cfg.theta_curriculum_replay_trim_frac, 0.0, 0.95))
      trimmed = agent.replay.trim_oldest(trim_frac) if trim_frac > 0.0 else 0
      print(
        f"[theta-curriculum/{stage.name}] θ {old_theta:.3f} -> {active_theta:.3f} "
        f"(ep={agent.episodes}, ref={theta_ref}, replay_trim={trimmed})"
      )

    episodes_at_current_level += 1
    if (
      phase_idx < len(current_levels) - 1
      and episodes_at_current_level >= wlen
      and len(clean_full_episode_window) >= wlen
      and clean_full_no_risk_rate_last_w >= clean_rate_threshold
    ):
      old_label = active_label
      phase_idx += 1
      active_current = int(current_levels[phase_idx])  # type: ignore[assignment]
      active_label = _apply_current_level(env, active_current)  # type: ignore[arg-type]
      clean_full_episode_window.clear()
      clean_full_no_risk_rate_last_w = 0.0
      episodes_at_current_level = 0
      print(
        f"[current-curriculum/{stage.name}] {old_label} -> {active_label} "
        f"(clean_full_no_risk_rate reset to 0.0, window cleared)"
      )

    rec = {
      "episode": agent.episodes,
      "train_tag": train_tag,
      "stage_id": stage.stage_id,
      "stage_name": stage.name,
      "trajectory": scenario.trajectory,
      "obstacle_num": scenario.obstacle_num,
      "curriculum_phase": phase_idx + 1,
      "current_intensity": active_current,
      "current_label": active_label,
      "total_step": total_steps,
      "steps": env.step_counter,
      "undiscounted_cost": round(ep_cost, 6),
      "discounted_Jc": round(ep_jc, 6),
      "theta": active_theta if math.isfinite(active_theta) else None,
      "theta_ref": None if theta_ref is None else round(theta_ref, 6),
      "theta_curriculum_phase": theta_idx + 1,
      "jc_satisfied": bool(math.isfinite(active_theta) and ep_jc <= active_theta),
      "jc_satisfaction_rate_last_window": round(jc_satisfaction_rate_last_w, 6),
      "jc_full_horizon_window_len": jc_full_horizon_window_len,
      "jc_full_horizon_window_required": wlen,
      "theta_quarter_boundary_ep": _theta_quarter_boundary(csac_cfg, theta_idx),
      "lambda": float(agent.lam.detach().cpu()),
      "lambda_actor_scale": round(lambda_actor_scale, 6),
      "alpha": float(agent.alpha.detach().cpu()),
      "target_entropy_scale": float(agent.target_entropy_scale),
      "alpha_frozen": bool(agent.alpha_frozen),
      "episode_return": round(ep_reward, 6),
      "reward_per_step": round(ep_reward_per_step, 6),
      "collision": collision,
      "collision_state": info.collision_state,
      "collision_warning": bool(info.collision_warning),
      "collision_warning_count": ep_warning_count,
      "warning_per_meter": round(ep_warning_per_meter, 6),
      "episode_distance_m": round(ep_distance_m, 6),
      "distance_before_collision_m": (
        round(ep_dist_before_collision, 6) if ep_dist_before_collision is not None else None
      ),
      "rmse_3d": round(ep_rmse, 6),
      "collision_count_last_window": collision_count_last_w,
      "clean_full_episode": clean_full_episode,
      "success": ep_success,
      "clean_full_no_risk_rate_last_window": round(clean_full_no_risk_rate_last_w, 6),
      "success_rate_last_window": round(success_rate_last_w, 6),
      "reason": info.reason,
      "tracking_error_sum": round(ep_tracking_error, 6),
    }
    cost_records.append(rec)
    if agent.episodes % cost_log_flush == 0:
      _flush_cost_jsonl(cost_log_path, cost_records)

    if train_print_every > 0 and agent.episodes % train_print_every == 0:
      print(
        f"[train/{stage.name}/{active_label} seed {seed}] "
        f"ep={agent.episodes}/{csac_cfg.train_episodes} steps={env.step_counter} "
        f"total_steps={total_steps} reward={ep_reward:.2f} "
        f"avg_reward_lastW={avg_reward_last_w:.2f} rmse={ep_rmse:.3f} "
        f"success_lastW={success_rate_last_w*100:.1f}% "
        f"clean_full_lastW={clean_full_no_risk_rate_last_w*100:.1f}% "
        f"coll_lastW={collision_count_last_w} Jc={ep_jc:.3f} "
        f"theta={active_theta:.3f} ref={theta_ref} jc_ok_lastW={jc_satisfaction_rate_last_w*100:.1f}% "
        f"jc_fullW={jc_full_horizon_window_len}/{wlen} "
        f"lam={float(agent.lam):.3f} lam_scale={lambda_actor_scale:.2f} reason={info.reason}"
      )

    agent.scores.append(ep_reward)
    if agent.episodes > 499:
      agent.trajectory_demo.append(traj_cache)

    traj_cache = []
    ep_reward = 0.0
    ep_cost = 0.0
    ep_step_costs = []

    if total_steps % csac_cfg.log_interval == 0:
      agent.save(ckpt_dir, seed, agent.scores, agent.trajectory_demo)
      _write_cost_summary(log_dir, seed, stage, cost_records, csac_cfg)

    do_eval = (
      csac_cfg.eval_interval_episodes > 0
      and agent.episodes % csac_cfg.eval_interval_episodes == 0
    )
    if do_eval or agent.episodes >= csac_cfg.train_episodes:
      snapshot_ckpt = _save_eval_snapshot(agent, ckpt_dir, seed, agent.episodes)
      eval_rec = _evaluate_agent_during_train(
        agent,
        seed=seed,
        episode=agent.episodes,
        total_step=total_steps,
        cfg=csac_cfg,
        current_intensity=active_current,  # type: ignore[arg-type]
        checkpoint=str(snapshot_ckpt),
        eval_trajectories=stage_eval_trajectories,
      )
      with open(eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_rec, ensure_ascii=False) + "\n")
      if _is_better_eval_record(eval_rec, best_eval_record):
        best_eval_record = eval_rec
        shutil.copy2(snapshot_ckpt, ckpt_dir / f"csac_actor_seed{seed}_best.pt")
        best_eval_path.write_text(json.dumps(best_eval_record, indent=2, ensure_ascii=False), encoding="utf-8")
      sel = eval_rec["selection_metric"]
      overall = eval_rec["overall"]
      coll_dist = overall.get("distance_before_collision_mean")
      coll_dist_str = f"{coll_dist:.2f}m" if coll_dist is not None else "n/a"
      print(
        f"[eval/{stage.name} seed {seed}] ep={agent.episodes} "
        f"avg_reward={overall['avg_episode_reward']:.2f} "
        f"worst_traj_reward={sel['worst_traj_reward_mean']:.2f} "
        f"success={overall['success_rate']*100:.1f}% "
        f"collision={overall['collision_rate']*100:.1f}% "
        f"rmse={overall['rmse_3d_mean']:.2f} "
        f"energy={overall['energy_mean']:.2f} "
        f"Jc={overall['discounted_Jc_mean']:.2f} "
        f"dist_before_collision_mean={coll_dist_str} "
        f"ckpt={snapshot_ckpt.name}"
      )

    if agent.episodes >= csac_cfg.train_episodes:
      break
    scenario = _sample_training_scenario(stage, stage_rng)
    _set_env_scenario(env, scenario)
    obs = env.reset()

  _flush_cost_jsonl(cost_log_path, cost_records)
  agent.save(ckpt_dir, seed, agent.scores, agent.trajectory_demo)
  _write_cost_summary(log_dir, seed, stage, cost_records, csac_cfg)
  if best_eval_record is not None and not best_eval_path.exists():
    best_eval_path.write_text(json.dumps(best_eval_record, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(description="CSAC 分阶段训练（单轨迹课程 + 独立评估）")
  parser.add_argument("--seed", type=int, default=0, help="0=全部 num_seeds；否则只跑该 seed")
  parser.add_argument("--episodes", type=int, default=0, help="每个阶段训练总 episode（0=用 config 默认）")
  parser.add_argument("--frames", type=int, default=0, help="每个阶段训练总 steps 上限（0=episodes*episode_steps）")
  parser.add_argument(
    "--theta",
    type=float,
    default=0.0,
    help="覆盖能耗约束：启用 θ 课程时作为最终档（默认 100）；关闭课程时为固定 θ",
  )
  parser.add_argument(
    "--no-theta-curriculum",
    action="store_true",
    help="关闭 θ 约束课程，全程使用 --theta 或 config 中的 energy_budget_theta",
  )
  parser.add_argument(
    "--theta-levels",
    type=str,
    default="",
    help="θ 课程各档：relative 模式为倍数如 1.2,1.1,1.05,1.0；absolute 为能耗阈值",
  )
  parser.add_argument(
    "--theta-curriculum-mode",
    type=str,
    default="",
    choices=["", "absolute", "relative"],
    help="θ 课程模式：relative=按满程 Jc 中位数倍数（默认）；absolute=固定阈值",
  )
  parser.add_argument(
    "--theta-satisfaction-threshold",
    type=float,
    default=-1.0,
    help="Jc 稳定满足当前 θ 的窗口满足率阈值（0~1，默认 0.80）",
  )
  parser.add_argument("--device", type=str, default="", choices=["", "cuda", "cpu", "auto", "cuda:0"])
  parser.add_argument("--current", type=str, default="0", help="课程起始洋流强度：0–3 或 none/mild/moderate/strong")
  parser.add_argument(
    "--no-current-curriculum",
    action="store_true",
    help="固定全程无洋流，关闭洋流课程晋级（等价于 curriculum_levels=[0]）",
  )
  parser.add_argument("--fast", action="store_true")
  parser.add_argument("--update-every", type=int, default=0)
  parser.add_argument("--window", type=int, default=0, help="课程窗口长度")
  parser.add_argument("--clean-rate-threshold", type=float, default=-1.0, help="晋级 clean_full_no_risk_rate 阈值（0~1）")
  parser.add_argument("--stages", type=str, default="1,2,3,4", help="要运行的阶段，如 1,2,3,4")
  parser.add_argument("--train-tag", type=str, default=TRAIN_TAG_DEFAULT, help="训练实验目录名")
  args = parser.parse_args()

  if args.no_current_curriculum:
    start_intensity = int(parse_current_intensity(args.current))
    if start_intensity != 0:
      print(
        f"[warn] --no-current-curriculum 已启用，忽略 --current {args.current!r}，全程固定无洋流。"
      )
    levels = (0,)
  else:
    start_intensity = int(parse_current_intensity(args.current))
    levels = tuple(int(np.clip(start_intensity + i, 0, 3)) for i in range(3))
  stage_ids = _parse_stage_ids(args.stages)

  cfg = CSACConfig()
  if args.fast:
    apply_fast_training_profile(cfg)
  if args.device:
    cfg.device = args.device
  if args.episodes > 0:
    cfg.train_episodes = args.episodes
  if args.frames > 0:
    cfg.train_frames = args.frames
    cfg.log_interval = min(cfg.log_interval, args.frames)
  else:
    cfg.train_frames = cfg.train_episodes * SimConfig().episode_steps
  if args.no_theta_curriculum:
    cfg.theta_curriculum_enabled = False
  if args.theta_curriculum_mode:
    cfg.theta_curriculum_mode = args.theta_curriculum_mode  # type: ignore[assignment]
  if args.theta_levels:
    cfg.theta_curriculum_levels = tuple(float(x.strip()) for x in args.theta_levels.split(",") if x.strip())
  if args.theta_satisfaction_threshold >= 0.0:
    cfg.theta_satisfaction_rate_threshold = float(np.clip(args.theta_satisfaction_threshold, 0.0, 1.0))
  if args.theta > 0:
    if cfg.theta_curriculum_enabled:
      if cfg.theta_curriculum_mode == "relative":
        print("[warn] relative θ 课程已启用，忽略 --theta；请用 --theta-levels 调整末档倍数。")
      else:
        levels = list(cfg.theta_curriculum_levels)
        levels[-1] = float(args.theta)
        cfg.theta_curriculum_levels = tuple(levels)
    else:
      cfg.energy_budget_theta = args.theta
  if args.update_every > 0:
    cfg.update_every = args.update_every
  if args.window > 0:
    cfg.curriculum_window_episodes = args.window
  if args.clean_rate_threshold >= 0.0:
    cfg.curriculum_clean_full_rate_threshold = float(np.clip(args.clean_rate_threshold, 0.0, 1.0))
  cfg.curriculum_current_levels = levels

  seeds = [args.seed] if args.seed > 0 else list(range(1, cfg.num_seeds + 1))
  for seed in seeds:
    for sid in stage_ids:
      stage = STAGE_SPECS[sid]
      cur_label = (
        f"{list(cfg.curriculum_current_levels)} (fixed none)"
        if args.no_current_curriculum
        else str(list(cfg.curriculum_current_levels))
      )
      theta_label = (
        f"theta_curriculum={cfg.theta_curriculum_mode}:{list(cfg.theta_curriculum_levels)} "
        f"(jc>={cfg.theta_satisfaction_rate_threshold:.0%}, "
        f"clean>={cfg.theta_curriculum_clean_full_rate_threshold:.0%}, "
        f"success>={cfg.theta_curriculum_success_rate_threshold:.0%})"
        if cfg.theta_curriculum_enabled
        else f"theta={cfg.energy_budget_theta} (fixed)"
      )
      print(
        f"========== train={args.train_tag}/{stage.name}_seed{seed} "
        f"stage={sid} episodes={cfg.train_episodes} frames={cfg.train_frames} "
        f"{theta_label} current_levels={cur_label} =========="
      )
      train_one_stage(seed, stage, cfg, train_tag=args.train_tag)


if __name__ == "__main__":
  main()
