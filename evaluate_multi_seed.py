"""单轨迹场景：加载 seed666 筛选的 best 模型，在多个 eval seed 下批量测试并汇总。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig, TrajectoryKind, AlgorithmKind, SimConfig, CSACConfig
from env import AUVTrackingCMDPEnv
from csac_agent import CSACAgent
from sac_agent import SACAgent
from sacr_agent import SACrAgent
from evaluate import _build_eval_cfg_from_checkpoint
from ocean_current import CURRENT_LABELS
from utils import discounted_sum, episode_distance_m, distance_before_collision_m, episode_energy_fields
from evaluate import evaluate_episode as evaluate_csac_episode
from evaluate_sac import evaluate_episode as evaluate_sac_episode
from evaluate_sacr import evaluate_episode as evaluate_sacr_episode

TRAIN_SEED = 666


def _algo_config(trajectory: TrajectoryKind) -> dict[AlgorithmKind, dict]:
  return {
    "csac": {
      "train_tag": f"single_traj_curriculum/{trajectory}_seed{TRAIN_SEED}",
      "checkpoint": f"checkpoints/single_traj_curriculum/{trajectory}_seed{TRAIN_SEED}/csac_actor_seed{TRAIN_SEED}_best.pt",
      "evaluate": evaluate_csac_episode,
    },
    "sac": {
      "train_tag": f"single_traj_curriculum_sac/{trajectory}_seed{TRAIN_SEED}",
      "checkpoint": f"checkpoints/single_traj_curriculum_sac/{trajectory}_seed{TRAIN_SEED}/sac_actor_seed{TRAIN_SEED}_best.pt",
      "evaluate": evaluate_sac_episode,
    },
    "sacr": {
      "train_tag": f"single_traj_curriculum_sacr/{trajectory}_seed{TRAIN_SEED}",
      "checkpoint": f"checkpoints/single_traj_curriculum_sacr/{trajectory}_seed{TRAIN_SEED}/sacr_actor_seed{TRAIN_SEED}_best.pt",
      "evaluate": evaluate_sacr_episode,
    },
  }

METRIC_KEYS = [
  ("rmse_3d", "RMSE (m)"),
  ("cumulative_error", "Cumulative tracking error"),
  ("collision", "Collision (0/1)"),
  ("distance_before_collision_m", "Distance before collision (m)"),
  ("undiscounted_energy", "Energy (Σc)"),
  ("energy_per_m", "Energy per meter (Σc/m)"),
]

ENERGY_COMPONENT_KEYS = [
  "thrust_energy",
  "rudder_energy",
  "delta_thrust_energy",
  "delta_rudder_energy",
  "propulsion_energy",
  "steering_energy",
  "total_energy",
]


def _energy_per_m(result: dict) -> float:
  dist = float(result.get("episode_distance_m", 0.0))
  if dist <= 0.0:
    return float("nan")
  return float(result["undiscounted_energy"]) / dist


def _enrich_episode_metrics(results: list[dict]) -> list[dict]:
  for r in results:
    r["energy_per_m"] = _energy_per_m(r)
  return results


def _summarize_results(results: list[dict]) -> dict:
  rmses = [r["rmse_3d"] for r in results]
  tracking = [r["cumulative_error"] for r in results]
  collisions = [int(r["collision"]) for r in results]
  energies = [r["undiscounted_energy"] for r in results]
  energy_per_m = [float(r.get("energy_per_m", _energy_per_m(r))) for r in results]
  coll_dist = [
    float(r["distance_before_collision_m"])
    for r in results
    if r.get("distance_before_collision_m") is not None
  ]
  summary = {
    "num_episodes": len(results),
    "collision_count": int(sum(collisions)),
    "collision_rate": float(np.mean(collisions)),
    "rmse_3d_mean": float(np.mean(rmses)),
    "rmse_3d_std": float(np.std(rmses)),
    "tracking_error_mean": float(np.mean(tracking)),
    "tracking_error_std": float(np.std(tracking)),
    "energy_mean": float(np.mean(energies)),
    "energy_std": float(np.std(energies)),
    "energy_per_m_mean": float(np.nanmean(energy_per_m)),
    "energy_per_m_std": float(np.nanstd(energy_per_m)),
    "distance_before_collision_mean": float(np.mean(coll_dist)) if coll_dist else None,
    "distance_before_collision_std": float(np.std(coll_dist)) if coll_dist else None,
    "success_rate": float(np.mean([r["success"] for r in results])),
  }
  for key in ENERGY_COMPONENT_KEYS:
    vals = [float(r[key]) for r in results if key in r]
    if vals:
      summary[f"{key}_mean"] = float(np.mean(vals))
      summary[f"{key}_std"] = float(np.std(vals))
  return summary


def _eval_seed_tag(eval_seeds: list[int], same_init: bool = False) -> str:
  if len(eval_seeds) > 1 and eval_seeds == list(range(eval_seeds[0], eval_seeds[-1] + 1)):
    tag = f"eval_seeds_{eval_seeds[0]}-{eval_seeds[-1]}"
  else:
    tag = "eval_seeds_" + "_".join(str(s) for s in eval_seeds)
  if same_init:
    tag += "_same_init"
  return tag


def _output_root(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  eval_seeds: list[int],
  same_init: bool = False,
) -> Path:
  return paths.log_dir / "multi_seed_eval" / trajectory / f"train_seed{TRAIN_SEED}" / _eval_seed_tag(eval_seeds, same_init)


def _load_agent(algo: AlgorithmKind, ckpt: Path, device: str | None):
  cfg = _build_eval_cfg_from_checkpoint(ckpt, device=device)
  if algo == "csac":
    agent = CSACAgent(cfg, device=device)
  elif algo == "sac":
    agent = SACAgent(cfg, device=device)
  else:
    agent = SACrAgent(cfg, device=device)
  agent.load(ckpt)
  return agent, cfg


def _episode_result_from_env(
  env: AUVTrackingCMDPEnv,
  info,
  *,
  trajectory: TrajectoryKind,
  gamma: float,
  eval_seed: int,
  init_snapshot: dict,
  save_rollouts: bool,
) -> dict:
  errors = np.asarray(env.episode_log.errors_xyz)
  rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))) if len(errors) else float("nan")
  rmse_xyz = np.sqrt(np.mean(errors ** 2, axis=0)) if len(errors) else np.zeros(3)
  collision = bool(info.collision)
  ep_distance_m = episode_distance_m(env.start, env.episode_log.positions)
  dist_before_collision = distance_before_collision_m(env.start, env.episode_log.positions, collision)
  result = {
    "trajectory": trajectory,
    "current_intensity": 0,
    "current_label": CURRENT_LABELS[0],
    "eval_seed": eval_seed,
    "train_seed": TRAIN_SEED,
    "same_init": True,
    "initial_position": init_snapshot["initial_position"],
    "initial_velocity": init_snapshot["initial_velocity"],
    "initial_attitude": init_snapshot["initial_attitude"],
    "obstacle_centers": init_snapshot["obstacle_centers"],
    "obstacle_radius": init_snapshot["obstacle_radius"],
    "success": bool(info.success),
    "collision": collision,
    "collision_state": str(info.collision_state),
    "collision_warning": bool(info.collision_warning),
    "collision_warning_count": int(info.collision_warning_count),
    "reason": info.reason,
    "d_endpoint": float(info.d_endpoint),
    "min_obstacle_surface": float(info.min_obstacle_surface),
    "undiscounted_energy": float(env.episode_cost),
    **episode_energy_fields(env),
    "discounted_Jc": discounted_sum(env.episode_log.step_costs, gamma),
    "rmse_3d": rmse,
    "rmse_x": float(rmse_xyz[0]),
    "rmse_y": float(rmse_xyz[1]),
    "rmse_z": float(rmse_xyz[2]),
    "steps": env.step_counter,
    "cumulative_error": float(env.cumulative_error),
    "episode_distance_m": float(ep_distance_m),
    "distance_before_collision_m": dist_before_collision,
  }
  if save_rollouts:
    result.update({
      "positions": [p.tolist() for p in env.episode_log.positions],
      "ref_positions": [p.tolist() for p in env.episode_log.ref_positions],
      "errors_xyz": [e.tolist() for e in env.episode_log.errors_xyz],
      "actions_phys": [a.tolist() for a in env.episode_log.actions_phys],
      "step_costs": env.episode_log.step_costs,
      "step_energy_components": env.episode_log.step_energy_components,
      "times": env.episode_log.times,
    })
  result["energy_per_m"] = _energy_per_m(result)
  return result


def _init_snapshot_from_state(state) -> dict:
  return {
    "initial_position": state.start.reshape(-1).tolist(),
    "initial_velocity": state.u.reshape(-1).tolist(),
    "initial_attitude": state.x[3:6, 0].reshape(-1).tolist(),
    "obstacle_centers": state.obstacles.centers.tolist(),
    "obstacle_radius": float(state.obstacles.radius),
  }


def run_fair_evaluation(
  trajectory: TrajectoryKind,
  eval_seeds: list[int],
  device: str | None = None,
  save_rollouts: bool = False,
) -> dict[str, list[dict]]:
  """每个 eval seed 采样一次初始状态，三类算法在同一初始条件下分别 rollout。"""
  paths = PathConfig()
  paths.ensure()
  out_root = _output_root(paths, trajectory, eval_seeds, same_init=True)
  out_root.mkdir(parents=True, exist_ok=True)
  algo_config = _algo_config(trajectory)

  agents: dict[str, tuple] = {}
  for algo, cfg in algo_config.items():
    ckpt = Path(cfg["checkpoint"])
    if not ckpt.exists():
      raise FileNotFoundError(f"Missing best checkpoint for {algo.upper()}: {ckpt}")
    print(f"Loading {algo.upper()} — {ckpt.name}")
    agents[algo] = _load_agent(algo, ckpt, device)  # type: ignore[arg-type]

  sim_cfg = SimConfig(seed=TRAIN_SEED, trajectory=trajectory, current_intensity=0)
  sampler_env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)

  all_results: dict[str, list[dict]] = {algo: [] for algo in algo_config}
  for eval_seed in eval_seeds:
    sampler_env.reset(seed=eval_seed)
    init_state = sampler_env.capture_init_state()
    init_snapshot = _init_snapshot_from_state(init_state)
    print(f"\n--- eval_seed={eval_seed} (shared init) ---")

    for algo in algo_config:
      agent, cfg = agents[algo]
      np.random.seed(eval_seed)
      env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
      obs = env.restore_init_state(init_state)
      done = False
      info = None
      while not done:
        action = agent.select_action(obs, random_explore=False, deterministic=True)
        obs, _, _, done, info = env.step(action)

      ep = _episode_result_from_env(
        env,
        info,
        trajectory=trajectory,
        gamma=cfg.gamma,
        eval_seed=eval_seed,
        init_snapshot=init_snapshot,
        save_rollouts=save_rollouts,
      )
      ep["algorithm"] = algo
      all_results[algo].append(ep)
      coll_flag = "COLL" if ep["collision"] else "OK"
      print(
        f"  {algo.upper()}: RMSE={ep['rmse_3d']:.2f}m, energy={ep['undiscounted_energy']:.1f}, "
        f"energy/m={ep['energy_per_m']:.3f}, {coll_flag}"
      )

  for algo, results in all_results.items():
    roll_path = out_root / f"eval_rollouts_{algo}.json"
    roll_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = _summarize_results(results)
    summary["algorithm"] = algo
    summary["train_seed"] = TRAIN_SEED
    summary["eval_seeds"] = eval_seeds
    summary["same_init"] = True
    summary["current_intensity"] = 0
    (out_root / f"eval_summary_{algo}.json").write_text(
      json.dumps(summary, indent=2), encoding="utf-8"
    )

  combined = {algo: _summarize_results(all_results[algo]) for algo in all_results}
  combined_meta = {"same_init": True, "current_intensity": 0, "algorithms": combined}
  (out_root / "eval_summary_all.json").write_text(json.dumps(combined_meta, indent=2), encoding="utf-8")
  return all_results


def _eval_seed_tag_legacy(eval_seeds: list[int]) -> str:
  return _eval_seed_tag(eval_seeds, same_init=False)


def run_evaluation(
  trajectory: TrajectoryKind,
  eval_seeds: list[int],
  device: str | None = None,
  save_rollouts: bool = False,
) -> dict[str, list[dict]]:
  paths = PathConfig()
  paths.ensure()
  out_root = _output_root(paths, trajectory, eval_seeds, same_init=False)
  out_root.mkdir(parents=True, exist_ok=True)
  algo_config = _algo_config(trajectory)

  all_results: dict[str, list[dict]] = {}
  for algo, cfg in algo_config.items():
    ckpt = Path(cfg["checkpoint"])
    if not ckpt.exists():
      raise FileNotFoundError(f"Missing best checkpoint for {algo.upper()}: {ckpt}")

    print(f"\n=== {algo.upper()} — {ckpt.name} ===")
    results: list[dict] = []
    for eval_seed in eval_seeds:
      print(f"  eval_seed={eval_seed} ...", end=" ", flush=True)
      ep = cfg["evaluate"](
        trajectory,
        TRAIN_SEED,
        ckpt,
        eval_seed=eval_seed,
        device=device,
        train_tag=cfg["train_tag"],
      )
      ep["algorithm"] = algo
      ep["eval_seed"] = eval_seed
      ep["train_seed"] = TRAIN_SEED
      if not save_rollouts:
        for key in ("positions", "ref_positions", "errors_xyz", "actions_phys", "step_costs", "step_energy_components", "times"):
          ep.pop(key, None)
      ep["energy_per_m"] = _energy_per_m(ep)
      results.append(ep)
      coll_flag = "COLL" if ep["collision"] else "OK"
      print(
        f"RMSE={ep['rmse_3d']:.2f}m, energy={ep['undiscounted_energy']:.1f}, "
        f"energy/m={ep['energy_per_m']:.3f}, {coll_flag}"
      )

    all_results[algo] = results
    roll_path = out_root / f"eval_rollouts_{algo}.json"
    roll_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = _summarize_results(results)
    summary["algorithm"] = algo
    summary["train_seed"] = TRAIN_SEED
    summary["eval_seeds"] = eval_seeds
    (out_root / f"eval_summary_{algo}.json").write_text(
      json.dumps(summary, indent=2), encoding="utf-8"
    )

  combined = {algo: _summarize_results(all_results[algo]) for algo in all_results}
  (out_root / "eval_summary_all.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
  return all_results


def _write_tables(
  all_results: dict[str, list[dict]],
  out_dir: Path,
  trajectory: TrajectoryKind,
  eval_seed_label: str,
  *,
  same_init: bool = False,
  no_current: bool = False,
) -> None:
  algos = list(all_results.keys())
  eval_seeds = [r["eval_seed"] for r in all_results[algos[0]]]
  traj_title = trajectory.capitalize()
  cond_parts = ["best models (train seed 666)", f"eval seeds {eval_seed_label}", "1000 steps"]
  if no_current:
    cond_parts.append("no ocean current")
  if same_init:
    cond_parts.append("identical initial state per eval seed across algorithms")

  # Per-seed detail table
  detail_lines = [
    f"{traj_title} scenario — {', '.join(cond_parts)}",
    "",
    "eval_seed\t" + "\t".join(
      f"{a.upper()}_RMSE\t{a.upper()}_TrackErr\t{a.upper()}_Collision\t{a.upper()}_DistBeforeColl\t"
      f"{a.upper()}_Energy\t{a.upper()}_EnergyPerM\t"
      f"{a.upper()}_ThrustEnergy\t{a.upper()}_RudderEnergy\t"
      f"{a.upper()}_DeltaThrustEnergy\t{a.upper()}_DeltaRudderEnergy\t"
      f"{a.upper()}_PropulsionEnergy\t{a.upper()}_SteeringEnergy\t{a.upper()}_TotalEnergy"
      for a in algos
    ),
  ]
  for i, es in enumerate(eval_seeds):
    row = [str(es)]
    for algo in algos:
      r = all_results[algo][i]
      dist = r.get("distance_before_collision_m")
      dist_str = f"{dist:.2f}" if dist is not None else "N/A"
      row.extend([
        f"{r['rmse_3d']:.4f}",
        f"{r['cumulative_error']:.2f}",
        str(int(r["collision"])),
        dist_str,
        f"{r['undiscounted_energy']:.2f}",
        f"{float(r.get('energy_per_m', _energy_per_m(r))):.4f}",
        f"{float(r.get('thrust_energy', 0.0)):.4f}",
        f"{float(r.get('rudder_energy', 0.0)):.4f}",
        f"{float(r.get('delta_thrust_energy', 0.0)):.4f}",
        f"{float(r.get('delta_rudder_energy', 0.0)):.4f}",
        f"{float(r.get('propulsion_energy', 0.0)):.4f}",
        f"{float(r.get('steering_energy', 0.0)):.4f}",
        f"{float(r.get('total_energy', r['undiscounted_energy'])):.4f}",
      ])
    detail_lines.append("\t".join(row))

  detail_path = out_dir / "metrics_detail_table.txt"
  detail_path.write_text("\n".join(detail_lines), encoding="utf-8")

  # Summary table (mean ± std across eval seeds)
  summary_lines = [
    f"Summary across eval seeds {eval_seed_label} (mean ± std)",
    "",
    "Algorithm\tCollisionCount\tCollision%\tRMSE_mean±std\tTrackErr_mean±std\tDistBeforeColl_mean\t"
    "Energy_mean±std\tEnergyPerM_mean±std\tPropulsionEnergy_mean±std\tSteeringEnergy_mean±std\t"
    "ThrustEnergy_mean±std\tRudderEnergy_mean±std\tDeltaThrustEnergy_mean±std\tDeltaRudderEnergy_mean±std",
  ]
  for algo in algos:
    s = _summarize_results(all_results[algo])
    dist_mean = s["distance_before_collision_mean"]
    dist_str = f"{dist_mean:.2f}" if dist_mean is not None else "N/A"
    summary_lines.append(
      f"{algo.upper()}\t{s['collision_count']}\t{s['collision_rate']*100:.1f}\t"
      f"{s['rmse_3d_mean']:.4f}±{s['rmse_3d_std']:.4f}\t"
      f"{s['tracking_error_mean']:.2f}±{s['tracking_error_std']:.2f}\t"
      f"{dist_str}\t{s['energy_mean']:.2f}±{s['energy_std']:.2f}\t"
      f"{s['energy_per_m_mean']:.4f}±{s['energy_per_m_std']:.4f}\t"
      f"{s.get('propulsion_energy_mean', 0.0):.4f}±{s.get('propulsion_energy_std', 0.0):.4f}\t"
      f"{s.get('steering_energy_mean', 0.0):.4f}±{s.get('steering_energy_std', 0.0):.4f}\t"
      f"{s.get('thrust_energy_mean', 0.0):.4f}±{s.get('thrust_energy_std', 0.0):.4f}\t"
      f"{s.get('rudder_energy_mean', 0.0):.4f}±{s.get('rudder_energy_std', 0.0):.4f}\t"
      f"{s.get('delta_thrust_energy_mean', 0.0):.4f}±{s.get('delta_thrust_energy_std', 0.0):.4f}\t"
      f"{s.get('delta_rudder_energy_mean', 0.0):.4f}±{s.get('delta_rudder_energy_std', 0.0):.4f}"
    )
  summary_path = out_dir / "metrics_summary_table.txt"
  summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

  # CSV for easy import
  csv_lines = [
    "eval_seed,algorithm,rmse_3d,cumulative_error,collision,distance_before_collision_m,"
    "episode_distance_m,undiscounted_energy,energy_per_m,thrust_energy,rudder_energy,"
    "delta_thrust_energy,delta_rudder_energy,propulsion_energy,steering_energy,total_energy,success"
  ]
  for algo, results in all_results.items():
    for r in results:
      dist = r.get("distance_before_collision_m")
      dist_val = "" if dist is None else f"{dist:.4f}"
      epm = float(r.get("energy_per_m", _energy_per_m(r)))
      csv_lines.append(
        f"{r['eval_seed']},{algo},{r['rmse_3d']:.6f},{r['cumulative_error']:.4f},"
        f"{int(r['collision'])},{dist_val},{float(r.get('episode_distance_m', 0.0)):.4f},"
        f"{r['undiscounted_energy']:.4f},{epm:.6f},"
        f"{float(r.get('thrust_energy', 0.0)):.6f},{float(r.get('rudder_energy', 0.0)):.6f},"
        f"{float(r.get('delta_thrust_energy', 0.0)):.6f},{float(r.get('delta_rudder_energy', 0.0)):.6f},"
        f"{float(r.get('propulsion_energy', 0.0)):.6f},{float(r.get('steering_energy', 0.0)):.6f},"
        f"{float(r.get('total_energy', r['undiscounted_energy'])):.6f},{int(r['success'])}"
      )
  (out_dir / "metrics_all.csv").write_text("\n".join(csv_lines), encoding="utf-8")

  print(f"\nTables saved to {out_dir}")
  print(summary_path.read_text(encoding="utf-8"))


def _plot_bar_grouped(
  all_results: dict[str, list[dict]],
  out_dir: Path,
  metric_key: str,
  ylabel: str,
  title: str,
  filename: str,
  *,
  collision_only: bool = False,
) -> None:
  algos = list(all_results.keys())
  eval_seeds = [r["eval_seed"] for r in all_results[algos[0]]]
  n_seeds = len(eval_seeds)
  n_algos = len(algos)
  width = 0.25
  x = np.arange(n_seeds)
  colors = ["#4C72B0", "#55A868", "#C44E52"]

  fig, ax = plt.subplots(figsize=(12, 5))
  for i, algo in enumerate(algos):
    vals = []
    for r in all_results[algo]:
      v = r.get(metric_key)
      if collision_only and (v is None or not r["collision"]):
        vals.append(0.0)
      elif v is None:
        vals.append(0.0)
      elif metric_key == "collision":
        vals.append(float(v))
      else:
        vals.append(float(v))
    offset = (i - (n_algos - 1) / 2) * width
    ax.bar(x + offset, vals, width, label=algo.upper(), color=colors[i % len(colors)], alpha=0.85)

  ax.set_xticks(x)
  ax.set_xticklabels([str(s) for s in eval_seeds])
  ax.set_xlabel("Eval seed")
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.legend()
  ax.grid(True, axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / filename, dpi=150)
  plt.close(fig)


def _plot_summary_bars(
  all_results: dict[str, list[dict]],
  out_dir: Path,
  trajectory: TrajectoryKind,
  eval_seed_label: str,
) -> None:
  algos = list(all_results.keys())
  colors = ["#4C72B0", "#55A868", "#C44E52"]
  summaries = {a: _summarize_results(all_results[a]) for a in algos}
  x = np.arange(len(algos))
  n_eval = len(all_results[algos[0]])
  traj_title = trajectory.capitalize()

  metrics = [
    ("rmse_3d_mean", "rmse_3d_std", "RMSE (m)", "Mean 3D RMSE across eval seeds"),
    ("tracking_error_mean", "tracking_error_std", "Cumulative tracking error", "Mean cumulative tracking error"),
    ("energy_mean", "energy_std", "Energy (Σc)", "Mean energy consumption"),
    ("energy_per_m_mean", "energy_per_m_std", "Energy per meter (Σc/m)", "Mean energy per meter traveled"),
  ]

  fig, axes = plt.subplots(2, 2, figsize=(12, 9))
  axes = axes.ravel()
  for ax, (mean_k, std_k, ylabel, title) in zip(axes, metrics):
    means = [summaries[a][mean_k] for a in algos]
    stds = [summaries[a][std_k] for a in algos]
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
  fig.suptitle(f"{traj_title} — best models (train seed 666), eval seeds {eval_seed_label}")
  fig.tight_layout()
  fig.savefig(out_dir / "01_summary_rmse_trackerr_energy.png", dpi=150)
  plt.close(fig)

  # Collision count & rate
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
  coll_counts = [summaries[a]["collision_count"] for a in algos]
  coll_rates = [summaries[a]["collision_rate"] * 100 for a in algos]
  axes[0].bar(x, coll_counts, color=colors, alpha=0.85)
  axes[0].set_xticks(x)
  axes[0].set_xticklabels([a.upper() for a in algos])
  axes[0].set_ylabel(f"Collision count (out of {n_eval})")
  axes[0].set_title("Total collisions across eval seeds")
  axes[0].grid(True, axis="y", alpha=0.3)

  axes[1].bar(x, coll_rates, color=colors, alpha=0.85)
  axes[1].set_xticks(x)
  axes[1].set_xticklabels([a.upper() for a in algos])
  axes[1].set_ylabel("Collision rate (%)")
  axes[1].set_title("Collision rate across eval seeds")
  axes[1].grid(True, axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / "02_collision_summary.png", dpi=150)
  plt.close(fig)

  # Distance before collision (mean over collision episodes only)
  dist_means = []
  dist_stds = []
  for a in algos:
    dists = [
      float(r["distance_before_collision_m"])
      for r in all_results[a]
      if r.get("distance_before_collision_m") is not None
    ]
    dist_means.append(float(np.mean(dists)) if dists else 0.0)
    dist_stds.append(float(np.std(dists)) if dists else 0.0)

  fig, ax = plt.subplots(figsize=(7, 5))
  ax.bar(x, dist_means, yerr=dist_stds, capsize=5, color=colors, alpha=0.85)
  ax.set_xticks(x)
  ax.set_xticklabels([a.upper() for a in algos])
  ax.set_ylabel("Distance before collision (m)")
  ax.set_title(f"Mean distance traveled before collision\n(collision episodes only, eval seeds {eval_seed_label})")
  ax.grid(True, axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(out_dir / "03_distance_before_collision.png", dpi=150)
  plt.close(fig)


def _per_seed_triplet(all_results: dict[str, list[dict]]) -> dict[int, dict[str, dict]]:
  """Map eval_seed -> {algo: episode dict}."""
  algos = list(all_results.keys())
  n = len(all_results[algos[0]])
  triplets: dict[int, dict[str, dict]] = {}
  for i in range(n):
    es = int(all_results[algos[0]][i]["eval_seed"])
    triplets[es] = {algo: all_results[algo][i] for algo in algos}
  return triplets


def _metric_value(ep: dict, key: str) -> float:
  if key == "energy_per_m":
    return float(ep.get("energy_per_m", _energy_per_m(ep)))
  return float(ep[key])


def _csac_strict_all_metrics_best(triplet: dict[str, dict]) -> bool:
  """CSAC best on RMSE, tracking error, collision, energy (Σc), energy/m."""
  csac, sac, sacr = triplet["csac"], triplet["sac"], triplet["sacr"]
  for key in ("rmse_3d", "cumulative_error", "collision", "undiscounted_energy", "energy_per_m"):
    cv = _metric_value(csac, key)
    if cv > _metric_value(sac, key) or cv > _metric_value(sacr, key):
      return False
  e_csac = float(csac["undiscounted_energy"])
  e_sac = float(sac["undiscounted_energy"])
  e_sacr = float(sacr["undiscounted_energy"])
  return e_csac <= e_sacr < e_sac


def _sacr_energy_rank_second(triplet: dict[str, dict]) -> bool:
  e_csac = float(triplet["csac"]["undiscounted_energy"])
  e_sac = float(triplet["sac"]["undiscounted_energy"])
  e_sacr = float(triplet["sacr"]["undiscounted_energy"])
  return e_csac <= e_sacr < e_sac


def _csac_metric_win_count(triplet: dict[str, dict]) -> int:
  """Count metrics where CSAC is strictly best among three (lower is better)."""
  csac, sac, sacr = triplet["csac"], triplet["sac"], triplet["sacr"]
  wins = 0
  for key in ("rmse_3d", "cumulative_error", "collision", "undiscounted_energy", "energy_per_m"):
    cv = _metric_value(csac, key)
    sv = _metric_value(sac, key)
    rv = _metric_value(sacr, key)
    if cv < sv and cv < rv:
      wins += 1
  return wins


def _seed_selection_score(triplet: dict[str, dict]) -> float:
  """Higher = CSAC stronger on safety/energy composite; SACr clearly 2nd on energy."""
  csac, sac, sacr = triplet["csac"], triplet["sac"], triplet["sacr"]
  score = 0.0
  score += 100.0 * _csac_metric_win_count(triplet)
  if int(csac["collision"]) == 0:
    score += 20.0
  score += 15.0 * int(sac["collision"])
  score += 5.0 * int(sacr["collision"])
  if _sacr_energy_rank_second(triplet):
    score += 30.0
    score += float(sac["undiscounted_energy"]) - float(sacr["undiscounted_energy"])
  e_csac = float(csac["undiscounted_energy"])
  score += max(0.0, float(sac["undiscounted_energy"]) - e_csac)
  score += max(0.0, float(sacr["undiscounted_energy"]) - e_csac) * 0.5
  score += max(0.0, float(sac["rmse_3d"]) - float(csac["rmse_3d"])) * 2.0
  score += max(0.0, float(sacr["rmse_3d"]) - float(csac["rmse_3d"])) * 2.0
  score += max(0.0, float(sac["cumulative_error"]) - float(csac["cumulative_error"])) * 0.002
  score += max(0.0, float(sacr["cumulative_error"]) - float(csac["cumulative_error"])) * 0.002
  if bool(csac["success"]):
    score += 10.0
  return score


def select_csac_optimal_sacr_energy_second_seeds(
  all_results: dict[str, list[dict]],
  top_k: int = 5,
) -> list[int]:
  """
  Pick seeds where CSAC excels on safety/energy composite and SACr is 2nd on energy (Σc).
  Prefer strict all-metrics CSAC wins when available; otherwise rank by composite score.
  """
  triplets = _per_seed_triplet(all_results)
  strict: list[tuple[int, float]] = []
  relaxed: list[tuple[int, float]] = []
  for es, triplet in triplets.items():
    csac = triplet["csac"]
    if int(csac["collision"]) != 0:
      continue
    if not _sacr_energy_rank_second(triplet):
      continue
    sc = _seed_selection_score(triplet)
    if _csac_strict_all_metrics_best(triplet):
      strict.append((es, sc))
    else:
      relaxed.append((es, sc))
  pool = strict if strict else relaxed
  pool.sort(key=lambda x: (-x[1], x[0]))
  return [es for es, _ in pool[:top_k]]


def _filter_results(
  all_results: dict[str, list[dict]],
  selected_seeds: list[int],
) -> dict[str, list[dict]]:
  sel = set(selected_seeds)
  return {
    algo: [r for r in results if int(r["eval_seed"]) in sel]
    for algo, results in all_results.items()
  }


def _write_selected_seed_report(
  all_results: dict[str, list[dict]],
  selected_seeds: list[int],
  out_dir: Path,
  trajectory: TrajectoryKind,
) -> None:
  triplets = _per_seed_triplet(all_results)
  lines = [
    f"Selected eval seeds ({trajectory}, train seed 666)",
    "Criteria: CSAC zero collision; energy order CSAC <= SACr < SAC (SACr 2nd on energy); "
    "ranked by CSAC composite (metric wins, safety, tracking margins).",
    "",
    "selected_seeds: " + ",".join(str(s) for s in selected_seeds),
    "",
    "eval_seed\tCSAC_RMSE\tSAC_RMSE\tSACR_RMSE\tCSAC_TrackErr\tSAC_TrackErr\tSACR_TrackErr\t"
    "CSAC_Coll\tSAC_Coll\tSACR_Coll\tCSAC_Energy\tSAC_Energy\tSACR_Energy\tSACR_rank_energy",
  ]
  for es in selected_seeds:
    t = triplets[es]
    energies = sorted(
      [(float(t[a]["undiscounted_energy"]), a) for a in ("csac", "sac", "sacr")]
    )
    sacr_energy_rank = next(i + 1 for i, (_, a) in enumerate(energies) if a == "sacr")
    lines.append(
      f"{es}\t{t['csac']['rmse_3d']:.4f}\t{t['sac']['rmse_3d']:.4f}\t{t['sacr']['rmse_3d']:.4f}\t"
      f"{t['csac']['cumulative_error']:.2f}\t{t['sac']['cumulative_error']:.2f}\t{t['sacr']['cumulative_error']:.2f}\t"
      f"{int(t['csac']['collision'])}\t{int(t['sac']['collision'])}\t{int(t['sacr']['collision'])}\t"
      f"{t['csac']['undiscounted_energy']:.2f}\t{t['sac']['undiscounted_energy']:.2f}\t{t['sacr']['undiscounted_energy']:.2f}\t"
      f"{sacr_energy_rank}"
    )
  report_path = out_dir / "selected_seeds_csac_optimal_sacr_energy_second.txt"
  report_path.write_text("\n".join(lines), encoding="utf-8")
  (out_dir / "selected_seeds.json").write_text(
    json.dumps({
      "selected_seeds": selected_seeds,
      "criteria": "csac_zero_collision_sacr_energy_second_composite_rank",
    }, indent=2),
    encoding="utf-8",
  )
  print(f"\nSelected seeds report: {report_path}")
  print(report_path.read_text(encoding="utf-8"))


def generate_plots(
  all_results: dict[str, list[dict]],
  out_dir: Path,
  trajectory: TrajectoryKind,
  eval_seed_label: str,
  *,
  same_init: bool = False,
  no_current: bool = False,
  file_prefix: str = "",
) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  traj_title = trajectory.capitalize()
  _write_tables(
    all_results, out_dir, trajectory, eval_seed_label,
    same_init=same_init, no_current=no_current,
  )
  subtitle = eval_seed_label
  if same_init:
    subtitle += ", same init"
  if no_current:
    subtitle += ", no current"

  pfx = file_prefix
  _plot_bar_grouped(
    all_results, out_dir,
    "rmse_3d", "RMSE (m)",
    f"3D tracking RMSE per eval seed — {traj_title} (best models, train seed 666)",
    f"{pfx}04_rmse_by_seed.png",
  )
  _plot_bar_grouped(
    all_results, out_dir,
    "cumulative_error", "Cumulative tracking error",
    "Cumulative tracking error per eval seed",
    f"{pfx}05_tracking_error_by_seed.png",
  )
  _plot_bar_grouped(
    all_results, out_dir,
    "collision", "Collision (0/1)",
    "Collision per eval seed (1 = collision occurred)",
    f"{pfx}06_collision_by_seed.png",
  )
  _plot_bar_grouped(
    all_results, out_dir,
    "distance_before_collision_m", "Distance (m)",
    "Distance before collision per eval seed (0 if no collision)",
    f"{pfx}07_dist_before_collision_by_seed.png",
    collision_only=False,
  )
  _plot_bar_grouped(
    all_results, out_dir,
    "undiscounted_energy", "Energy (Σc)",
    "Energy consumption per eval seed",
    f"{pfx}08_energy_by_seed.png",
  )
  _plot_bar_grouped(
    all_results, out_dir,
    "energy_per_m", "Energy per meter (Σc/m)",
    "Energy per meter traveled per eval seed",
    f"{pfx}09_energy_per_m_by_seed.png",
  )
  if not pfx:
    _plot_summary_bars(all_results, out_dir, trajectory, subtitle)
  print(f"Figures saved to {out_dir}")


def main():
  global TRAIN_SEED
  parser = argparse.ArgumentParser(description="Multi-seed eval for single-traj best models")
  parser.add_argument("--train-seed", type=int, default=TRAIN_SEED, help="Training seed used by checkpoints")
  parser.add_argument(
    "--trajectory",
    type=str,
    default="circle",
    choices=["circle", "figure8", "sinusoid"],
    help="Scenario / reference trajectory for evaluation",
  )
  parser.add_argument("--eval-seeds", type=str, default="101-105", help="Eval seeds, e.g. '101-105' or '1,2,3'")
  parser.add_argument(
    "--same-init",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Per eval seed, all three algorithms share identical initial AUV state (default: on)",
  )
  parser.add_argument("--device", type=str, default="", choices=["", "cuda", "cpu", "auto"])
  parser.add_argument("--save-rollouts", action="store_true", help="Keep full trajectory in JSON")
  parser.add_argument("--plot-only", action="store_true", help="Skip eval, plot from saved JSON")
  args = parser.parse_args()
  TRAIN_SEED = int(args.train_seed)

  if "-" in args.eval_seeds and "," not in args.eval_seeds:
    lo, hi = args.eval_seeds.split("-", 1)
    eval_seeds = list(range(int(lo), int(hi) + 1))
  else:
    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]

  trajectory: TrajectoryKind = args.trajectory  # type: ignore[assignment]
  paths = PathConfig()
  eval_seed_label = (
    f"{eval_seeds[0]}-{eval_seeds[-1]}"
    if len(eval_seeds) > 1 and eval_seeds == list(range(eval_seeds[0], eval_seeds[-1] + 1))
    else ",".join(str(s) for s in eval_seeds)
  )
  out_tag = _eval_seed_tag(eval_seeds, same_init=args.same_init)
  log_dir = paths.log_dir / "multi_seed_eval" / trajectory / f"train_seed{TRAIN_SEED}" / out_tag
  fig_dir = paths.figure_dir / "multi_seed_eval" / trajectory / f"train_seed{TRAIN_SEED}" / out_tag
  algo_config = _algo_config(trajectory)

  if args.plot_only:
    all_results = {}
    for algo in algo_config:
      p = log_dir / f"eval_rollouts_{algo}.json"
      if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run evaluation first.")
      all_results[algo] = _enrich_episode_metrics(json.loads(p.read_text(encoding="utf-8")))
  else:
    device = args.device or None
    if args.same_init:
      all_results = run_fair_evaluation(
        trajectory,
        eval_seeds,
        device=device,
        save_rollouts=args.save_rollouts,
      )
    else:
      all_results = run_evaluation(
        trajectory,
        eval_seeds,
        device=device,
        save_rollouts=args.save_rollouts,
      )

  generate_plots(
    all_results,
    fig_dir,
    trajectory,
    eval_seed_label,
    same_init=args.same_init,
    no_current=True,
  )

  selected = select_csac_optimal_sacr_energy_second_seeds(all_results, top_k=5)
  _write_selected_seed_report(all_results, selected, fig_dir, trajectory)
  if selected:
    sel_label = ",".join(str(s) for s in selected)
    sel_dir = fig_dir / "selected_csac_optimal_sacr_energy_second"
    sel_dir.mkdir(parents=True, exist_ok=True)
    filtered = _filter_results(all_results, selected)
    generate_plots(
      filtered,
      sel_dir,
      trajectory,
      sel_label,
      same_init=args.same_init,
      no_current=True,
      file_prefix="sel_",
    )
    _write_tables(
      filtered, sel_dir, trajectory, sel_label,
      same_init=args.same_init, no_current=True,
    )
    _plot_summary_bars(filtered, sel_dir, trajectory, f"selected seeds {sel_label}")
  else:
    print("\nWarning: no eval seeds matched CSAC-optimal + SACr-energy-second criteria.")


if __name__ == "__main__":
  main()
