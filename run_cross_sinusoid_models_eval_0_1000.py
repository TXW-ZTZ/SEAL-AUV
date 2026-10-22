from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

import evaluate_multi_seed as em
from config import PathConfig, SimConfig, TrajectoryKind
from env import AUVTrackingCMDPEnv


TRAIN_SEED = 666
SACR_TRAIN_SEED = 1
START_SEED = 0
END_SEED = 1000
MODEL_SOURCE = "sinusoid"
ALGOS = ("csac", "sac", "sacr")
TRAJECTORIES: tuple[TrajectoryKind, ...] = ("sinusoid", "figure8", "circle")

WORKER_AGENTS = None
WORKER_DEVICE = None
WORKER_TRAJECTORY = None
WORKER_MODEL_SOURCE = None


def train_trajectory_for(test_trajectory: TrajectoryKind) -> TrajectoryKind:
    if MODEL_SOURCE == "matched":
        return test_trajectory
    return "sinusoid"


def run_suffix() -> str:
    if MODEL_SOURCE == "matched":
        return "matched_models_scene_best_sacr_seed1_20260620"
    return "sinusoid_models_latest_csac_sacr_seed1_20260620"


def train_dir_label(test_trajectory: TrajectoryKind) -> str:
    if MODEL_SOURCE == "matched":
        return f"train_{test_trajectory}_seed{TRAIN_SEED}_sacr_seed{SACR_TRAIN_SEED}"
    return f"train_sinusoid_seed{TRAIN_SEED}_sacr_seed{SACR_TRAIN_SEED}"


def model_config_for(test_trajectory: TrajectoryKind) -> dict:
    train_trajectory = train_trajectory_for(test_trajectory)
    return {
        "csac": {
            "train_seed": TRAIN_SEED,
            "train_trajectory": train_trajectory,
            "checkpoint": Path(
                "checkpoints/csac_sinusoid_no_current_ep2200_eval100_seed666/"
                f"{train_trajectory}_seed{TRAIN_SEED}/csac_actor_seed{TRAIN_SEED}_best.pt"
            ),
        },
        "sac": {
            "train_seed": TRAIN_SEED,
            "train_trajectory": train_trajectory,
            "checkpoint": Path(
                "checkpoints/single_traj_curriculum_sac/"
                f"{train_trajectory}_seed{TRAIN_SEED}/sac_actor_seed{TRAIN_SEED}_best.pt"
            ),
        },
        "sacr": {
            "train_seed": SACR_TRAIN_SEED,
            "train_trajectory": train_trajectory,
            "checkpoint": Path(
                "checkpoints/single_traj_curriculum_sacr_weak/"
                f"{train_trajectory}_seed{SACR_TRAIN_SEED}/sacr_actor_seed{SACR_TRAIN_SEED}_best.pt"
            ),
        },
    }


def eval_seeds() -> list[int]:
    return list(range(START_SEED, END_SEED + 1))


def out_tag() -> str:
    return f"eval_seeds_{START_SEED}-{END_SEED}_same_init_{run_suffix()}"


def output_dirs(test_trajectory: TrajectoryKind) -> tuple[Path, Path]:
    paths = PathConfig()
    log_dir = (
        paths.log_dir
        / "multi_seed_eval"
        / test_trajectory
        / train_dir_label(test_trajectory)
        / out_tag()
    )
    fig_dir = (
        paths.figure_dir
        / "multi_seed_eval"
        / test_trajectory
        / train_dir_label(test_trajectory)
        / out_tag()
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, fig_dir


def model_metadata(test_trajectory: TrajectoryKind) -> dict:
    train_trajectory = train_trajectory_for(test_trajectory)
    model_config = model_config_for(test_trajectory)
    return {
        "test_trajectory": test_trajectory,
        "train_trajectory": train_trajectory,
        "model_source": MODEL_SOURCE,
        "train_seed": TRAIN_SEED,
        "sacr_train_seed": SACR_TRAIN_SEED,
        "eval_seed_range": [START_SEED, END_SEED],
        "eval_seed_count": len(eval_seeds()),
        "same_init": True,
        "episode_steps": 1000,
        "current_intensity": 0,
        "models": {
            algo: {
                **{k: v for k, v in cfg.items() if k != "checkpoint"},
                "checkpoint": str(cfg["checkpoint"].resolve()),
            }
            for algo, cfg in model_config.items()
        },
        "selection_criteria": {
            "strict": (
                "CSAC collision == 0; CSAC best or tied-best on RMSE, cumulative_error, "
                "collision, safety_distance, and total_energy; total energy order CSAC <= SACR < SAC"
            ),
            "safety_distance": (
                "episode_distance_m when no collision; distance_before_collision_m when collision occurs"
            ),
        },
    }


def write_metadata(test_trajectory: TrajectoryKind) -> None:
    log_dir, fig_dir = output_dirs(test_trajectory)
    text = json.dumps(model_metadata(test_trajectory), indent=2)
    (log_dir / "model_paths.json").write_text(text, encoding="utf-8")
    (fig_dir / "model_paths.json").write_text(text, encoding="utf-8")


def load_agents(test_trajectory: TrajectoryKind, device: str | None):
    agents = {}
    model_config = model_config_for(test_trajectory)
    for algo in ALGOS:
        ckpt = model_config[algo]["checkpoint"]
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {algo.upper()}: {ckpt}")
        print(f"Loading {algo.upper()} - {ckpt}", flush=True)
        agents[algo] = em._load_agent(algo, ckpt, device)
    return agents


def init_worker(device: str | None, test_trajectory: TrajectoryKind, model_source: str) -> None:
    global MODEL_SOURCE, WORKER_AGENTS, WORKER_DEVICE, WORKER_TRAJECTORY, WORKER_MODEL_SOURCE
    MODEL_SOURCE = model_source
    WORKER_DEVICE = device
    WORKER_TRAJECTORY = test_trajectory
    WORKER_MODEL_SOURCE = model_source
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    WORKER_AGENTS = load_agents(test_trajectory, device)


def evaluate_seed_worker(args: tuple[TrajectoryKind, int]) -> tuple[int, dict[str, dict]]:
    test_trajectory, seed = args
    if WORKER_AGENTS is None:
        init_worker(WORKER_DEVICE, WORKER_TRAJECTORY or test_trajectory, WORKER_MODEL_SOURCE or MODEL_SOURCE)
    agents = WORKER_AGENTS
    sim_cfg = SimConfig(seed=TRAIN_SEED, trajectory=test_trajectory, current_intensity=0)
    sampler_env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
    sampler_env.reset(seed=seed)
    init_state = sampler_env.capture_init_state()
    init_snapshot = em._init_snapshot_from_state(init_state)

    seed_results: dict[str, dict] = {}
    for algo in ALGOS:
        agent, cfg = agents[algo]
        np.random.seed(seed)
        env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
        obs = env.restore_init_state(init_state)
        done = False
        info = None
        while not done:
            action = agent.select_action(obs, random_explore=False, deterministic=True)
            obs, _, _, done, info = env.step(action)
        ep = em._episode_result_from_env(
            env,
            info,
            trajectory=test_trajectory,
            gamma=cfg.gamma,
            eval_seed=seed,
            init_snapshot=init_snapshot,
            save_rollouts=False,
        )
        ep["algorithm"] = algo
        ep["train_trajectory"] = train_trajectory_for(test_trajectory)
        seed_results[algo] = ep
    return seed, seed_results


def load_results(log_dir: Path) -> dict[str, dict[int, dict]]:
    all_results: dict[str, dict[int, dict]] = {algo: {} for algo in ALGOS}
    for algo in ALGOS:
        path = log_dir / f"eval_rollouts_{algo}.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            all_results[algo] = {int(r["eval_seed"]): r for r in rows}
    return all_results


def save_results(log_dir: Path, all_results: dict[str, dict[int, dict]]) -> None:
    for algo in ALGOS:
        rows = [all_results[algo][seed] for seed in sorted(all_results[algo])]
        (log_dir / f"eval_rollouts_{algo}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )


def write_eval_summaries(log_dir: Path, all_results: dict[str, dict[int, dict]]) -> None:
    combined = {}
    for algo in ALGOS:
        rows = [all_results[algo][seed] for seed in sorted(all_results[algo])]
        if not rows:
            continue
        summary = em._summarize_results(rows)
        train_trajectory = rows[0].get("train_trajectory", train_trajectory_for(rows[0]["trajectory"]))
        summary.update(
            {
                "algorithm": algo,
                "train_seed": TRAIN_SEED,
                "train_trajectory": train_trajectory,
                "eval_seeds": [int(r["eval_seed"]) for r in rows],
                "same_init": True,
                "current_intensity": 0,
            }
        )
        combined[algo] = summary
        (log_dir / f"eval_summary_{algo}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    (log_dir / "eval_summary_all.json").write_text(
        json.dumps(
            {
                "same_init": True,
                "current_intensity": 0,
                "train_trajectory": next(
                    (
                        rows[0].get("train_trajectory", train_trajectory_for(rows[0]["trajectory"]))
                        for rows in all_results.values()
                        if rows
                    ),
                    train_trajectory_for("sinusoid"),
                ),
                "algorithms": combined,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def dict_to_lists(all_results: dict[str, dict[int, dict]]) -> dict[str, list[dict]]:
    return {algo: [all_results[algo][seed] for seed in sorted(all_results[algo])] for algo in ALGOS}


def scenario_complete(all_results: dict[str, dict[int, dict]]) -> bool:
    target = eval_seeds()
    return all(all(seed in all_results[algo] for algo in ALGOS) for seed in target)


def completed_seed_count(all_results: dict[str, dict[int, dict]]) -> int:
    target = eval_seeds()
    return sum(1 for seed in target if all(seed in all_results[algo] for algo in ALGOS))


def run_scenario(
    test_trajectory: TrajectoryKind,
    agents: dict,
    *,
    save_every: int,
    stop_after: int | None,
) -> int:
    log_dir, _ = output_dirs(test_trajectory)
    write_metadata(test_trajectory)
    all_results = load_results(log_dir)
    sim_cfg = SimConfig(seed=TRAIN_SEED, trajectory=test_trajectory, current_intensity=0)
    sampler_env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
    target = eval_seeds()
    completed_this_run = 0

    print(
        f"\n=== Test trajectory {test_trajectory}: {completed_seed_count(all_results)}/{len(target)} complete ===",
        flush=True,
    )
    for idx, seed in enumerate(target, start=1):
        if all(seed in all_results[algo] for algo in ALGOS):
            if idx % 50 == 0:
                print(f"[skip:{test_trajectory}] seed {seed}; progress {idx}/{len(target)}", flush=True)
            continue
        if stop_after is not None and completed_this_run >= stop_after:
            print(f"[stop:{test_trajectory}] reached stop-after={stop_after}", flush=True)
            break

        sampler_env.reset(seed=seed)
        init_state = sampler_env.capture_init_state()
        init_snapshot = em._init_snapshot_from_state(init_state)

        for algo in ALGOS:
            if seed in all_results[algo]:
                continue
            agent, cfg = agents[algo]
            np.random.seed(seed)
            env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
            obs = env.restore_init_state(init_state)
            done = False
            info = None
            while not done:
                action = agent.select_action(obs, random_explore=False, deterministic=True)
                obs, _, _, done, info = env.step(action)
            ep = em._episode_result_from_env(
                env,
                info,
                trajectory=test_trajectory,
                gamma=cfg.gamma,
                eval_seed=seed,
                init_snapshot=init_snapshot,
                save_rollouts=False,
            )
            ep["algorithm"] = algo
            ep["train_trajectory"] = train_trajectory_for(test_trajectory)
            all_results[algo][seed] = ep

        completed_this_run += 1
        if idx % save_every == 0 or idx == len(target):
            save_results(log_dir, all_results)
            write_eval_summaries(log_dir, all_results)
            print(
                f"[save:{test_trajectory}] seed {seed}; progress {idx}/{len(target)}; "
                f"complete {completed_seed_count(all_results)}/{len(target)}",
                flush=True,
            )

    save_results(log_dir, all_results)
    write_eval_summaries(log_dir, all_results)
    if scenario_complete(all_results):
        finalize_scenario(test_trajectory)
    else:
        print(
            f"[partial:{test_trajectory}] complete {completed_seed_count(all_results)}/{len(target)}; rerun to continue",
            flush=True,
        )
    return completed_this_run


def run_scenario_parallel(
    test_trajectory: TrajectoryKind,
    *,
    device: str | None,
    save_every: int,
    stop_after: int | None,
    workers: int,
) -> int:
    log_dir, _ = output_dirs(test_trajectory)
    write_metadata(test_trajectory)
    all_results = load_results(log_dir)
    target = eval_seeds()
    pending = [seed for seed in target if not all(seed in all_results[algo] for algo in ALGOS)]
    if stop_after is not None:
        pending = pending[:stop_after]

    print(
        f"\n=== Test trajectory {test_trajectory}: {completed_seed_count(all_results)}/{len(target)} complete; "
        f"pending this run {len(pending)}; workers={workers} ===",
        flush=True,
    )
    if not pending:
        if scenario_complete(all_results):
            finalize_scenario(test_trajectory)
        return 0

    completed_this_run = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(device, test_trajectory, MODEL_SOURCE),
    ) as executor:
        future_to_seed = {
            executor.submit(evaluate_seed_worker, (test_trajectory, seed)): seed
            for seed in pending
        }
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            seed_done, seed_results = future.result()
            if seed_done != seed:
                raise RuntimeError(f"Worker returned seed {seed_done}, expected {seed}")
            for algo in ALGOS:
                all_results[algo][seed] = seed_results[algo]
            completed_this_run += 1

            if completed_this_run % save_every == 0 or completed_this_run == len(pending):
                save_results(log_dir, all_results)
                write_eval_summaries(log_dir, all_results)
                print(
                    f"[save:{test_trajectory}] completed this run {completed_this_run}/{len(pending)}; "
                    f"total complete {completed_seed_count(all_results)}/{len(target)}",
                    flush=True,
                )

    save_results(log_dir, all_results)
    write_eval_summaries(log_dir, all_results)
    if scenario_complete(all_results):
        finalize_scenario(test_trajectory)
    else:
        print(
            f"[partial:{test_trajectory}] complete {completed_seed_count(all_results)}/{len(target)}; rerun to continue",
            flush=True,
        )
    return completed_this_run


def metric(row: dict, key: str) -> float:
    if key == "energy_per_m":
        return float(row.get("energy_per_m", em._energy_per_m(row)))
    return float(row[key])


def collision(row: dict) -> int:
    return int(float(row["collision"]))


def safety_distance(row: dict) -> float:
    if collision(row) == 0:
        return float(row.get("episode_distance_m", 0.0))
    value = row.get("distance_before_collision_m")
    return 0.0 if value in (None, "") else float(value)


def triplets_from_results(results: dict[str, list[dict]]) -> dict[int, dict[str, dict]]:
    by_seed: dict[int, dict[str, dict]] = {}
    for algo in ALGOS:
        for row in results[algo]:
            by_seed.setdefault(int(row["eval_seed"]), {})[algo] = row
    return {seed: rows for seed, rows in by_seed.items() if all(algo in rows for algo in ALGOS)}


def total_energy(row: dict) -> float:
    return float(row.get("total_energy", row.get("undiscounted_energy", 0.0)))


def best_min(triplet: dict[str, dict], key: str) -> bool:
    csac_value = metric(triplet["csac"], key)
    return csac_value <= min(metric(triplet[algo], key) for algo in ("sac", "sacr"))


def best_total_energy(triplet: dict[str, dict]) -> bool:
    csac_value = total_energy(triplet["csac"])
    return csac_value <= min(total_energy(triplet[algo]) for algo in ("sac", "sacr"))


def best_collision(triplet: dict[str, dict]) -> bool:
    csac_value = collision(triplet["csac"])
    return csac_value <= min(collision(triplet[algo]) for algo in ("sac", "sacr"))


def best_safety(triplet: dict[str, dict]) -> bool:
    csac_value = safety_distance(triplet["csac"])
    return csac_value >= max(safety_distance(triplet[algo]) for algo in ("sac", "sacr"))


def sacr_energy_second(triplet: dict[str, dict]) -> bool:
    ordered = sorted((total_energy(triplet[algo]), algo) for algo in ALGOS)
    return ordered[1][1] == "sacr"


def energy_order(triplet: dict[str, dict]) -> bool:
    e_csac = total_energy(triplet["csac"])
    e_sac = total_energy(triplet["sac"])
    e_sacr = total_energy(triplet["sacr"])
    return e_csac <= e_sacr < e_sac


def selection_flags(triplet: dict[str, dict]) -> dict[str, bool]:
    return {
        "rmse": best_min(triplet, "rmse_3d"),
        "cumulative_error": best_min(triplet, "cumulative_error"),
        "collision": best_collision(triplet),
        "safety_distance": best_safety(triplet),
        "energy": best_total_energy(triplet),
        "energy_per_m": best_min(triplet, "energy_per_m"),
        "sacr_energy_second": sacr_energy_second(triplet),
        "energy_order_csac_le_sacr_lt_sac": energy_order(triplet),
        "csac_no_collision": collision(triplet["csac"]) == 0,
    }


def strict_match(flags: dict[str, bool]) -> bool:
    return (
        flags["csac_no_collision"]
        and flags["rmse"]
        and flags["cumulative_error"]
        and flags["collision"]
        and flags["safety_distance"]
        and flags["energy"]
        and flags["energy_order_csac_le_sacr_lt_sac"]
    )


def selection_score(triplet: dict[str, dict], flags: dict[str, bool]) -> float:
    csac, sac, sacr = triplet["csac"], triplet["sac"], triplet["sacr"]
    score = 0.0
    score += 500.0 * strict_match(flags)
    score += 140.0 * flags["energy"]
    score += 120.0 * flags["rmse"]
    score += 110.0 * flags["cumulative_error"]
    score += 100.0 * flags["safety_distance"]
    score += 80.0 * flags["collision"]
    score += 80.0 * flags["csac_no_collision"]
    score += 70.0 * flags["energy_order_csac_le_sacr_lt_sac"]
    score += 40.0 * flags["energy_per_m"]
    score += max(0.0, min(total_energy(sac), total_energy(sacr)) - total_energy(csac)) * 30.0
    score += max(0.0, min(metric(sac, "rmse_3d"), metric(sacr, "rmse_3d")) - metric(csac, "rmse_3d")) * 25.0
    score += max(
        0.0,
        min(metric(sac, "cumulative_error"), metric(sacr, "cumulative_error"))
        - metric(csac, "cumulative_error"),
    ) * 0.04
    score += max(0.0, safety_distance(csac) - max(safety_distance(sac), safety_distance(sacr))) * 0.4
    return score


def build_screening_records(results: dict[str, list[dict]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    records: list[dict] = []
    strict: list[dict] = []
    energy_order_records: list[dict] = []
    for seed, triplet in sorted(triplets_from_results(results).items()):
        flags = selection_flags(triplet)
        record = {
            "eval_seed": seed,
            "score": selection_score(triplet, flags),
            **{f"csac_win_{key}": int(value) for key, value in flags.items()},
        }
        for algo in ALGOS:
            row = triplet[algo]
            record.update(
                {
                    f"{algo}_rmse_3d": metric(row, "rmse_3d"),
                    f"{algo}_cumulative_error": metric(row, "cumulative_error"),
                    f"{algo}_collision": collision(row),
                    f"{algo}_safety_distance_m": safety_distance(row),
                    f"{algo}_distance_before_collision_m": row.get("distance_before_collision_m"),
                    f"{algo}_episode_distance_m": metric(row, "episode_distance_m"),
                    f"{algo}_energy": total_energy(row),
                    f"{algo}_energy_per_m": metric(row, "energy_per_m"),
                    f"{algo}_propulsion_energy": metric(row, "propulsion_energy"),
                    f"{algo}_steering_energy": metric(row, "steering_energy"),
                    f"{algo}_thrust_energy": metric(row, "thrust_energy"),
                    f"{algo}_rudder_energy": metric(row, "rudder_energy"),
                    f"{algo}_delta_thrust_energy": metric(row, "delta_thrust_energy"),
                    f"{algo}_delta_rudder_energy": metric(row, "delta_rudder_energy"),
                    f"{algo}_success": int(float(row["success"])),
                }
            )
        records.append(record)
        if strict_match(flags):
            strict.append(record)
        if flags["energy_order_csac_le_sacr_lt_sac"]:
            energy_order_records.append(record)

    records.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    strict.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    energy_order_records.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    selected = strict[:5]
    return records, strict, energy_order_records, selected


def screening_fields() -> list[str]:
    fields = [
        "eval_seed",
        "score",
        "csac_win_rmse",
        "csac_win_cumulative_error",
        "csac_win_collision",
        "csac_win_safety_distance",
        "csac_win_energy",
        "csac_win_energy_per_m",
        "csac_win_sacr_energy_second",
        "csac_win_energy_order_csac_le_sacr_lt_sac",
        "csac_win_csac_no_collision",
    ]
    metric_fields = [
        "rmse_3d",
        "cumulative_error",
        "collision",
        "safety_distance_m",
        "distance_before_collision_m",
        "episode_distance_m",
        "energy",
        "energy_per_m",
        "propulsion_energy",
        "steering_energy",
        "thrust_energy",
        "rudder_energy",
        "delta_thrust_energy",
        "delta_rudder_energy",
        "success",
    ]
    for algo in ALGOS:
        fields.extend(f"{algo}_{key}" for key in metric_fields)
    return fields


def write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=screening_fields(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_selected_tables(out_dir: Path, selected: list[dict]) -> None:
    long_fields = [
        "eval_seed",
        "algorithm",
        "rmse_3d",
        "cumulative_error",
        "collision",
        "safety_distance_m",
        "total_energy",
        "energy_per_m",
        "propulsion_energy",
        "steering_energy",
        "thrust_energy",
        "rudder_energy",
        "delta_thrust_energy",
        "delta_rudder_energy",
        "success",
    ]
    with (out_dir / "selected_5seeds_long.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=long_fields)
        writer.writeheader()
        for record in selected:
            for algo in ALGOS:
                writer.writerow(
                    {
                        "eval_seed": record["eval_seed"],
                        "algorithm": algo.upper(),
                        "rmse_3d": record[f"{algo}_rmse_3d"],
                        "cumulative_error": record[f"{algo}_cumulative_error"],
                        "collision": record[f"{algo}_collision"],
                        "safety_distance_m": record[f"{algo}_safety_distance_m"],
                        "total_energy": record[f"{algo}_energy"],
                        "energy_per_m": record[f"{algo}_energy_per_m"],
                        "propulsion_energy": record[f"{algo}_propulsion_energy"],
                        "steering_energy": record[f"{algo}_steering_energy"],
                        "thrust_energy": record[f"{algo}_thrust_energy"],
                        "rudder_energy": record[f"{algo}_rudder_energy"],
                        "delta_thrust_energy": record[f"{algo}_delta_thrust_energy"],
                        "delta_rudder_energy": record[f"{algo}_delta_rudder_energy"],
                        "success": record[f"{algo}_success"],
                    }
                )
    write_records(out_dir / "selected_5seeds_wide.csv", selected)


def plot_selected(test_trajectory: TrajectoryKind, out_dir: Path, selected: list[dict]) -> None:
    if not selected:
        return
    labels = {"csac": "CSAC", "sac": "SAC", "sacr": "SACR"}
    colors = {"csac": "#4C72B0", "sac": "#55A868", "sacr": "#C44E52"}
    seeds = [r["eval_seed"] for r in selected]
    x = np.arange(len(seeds))
    width = 0.24
    metric_specs = [
        ("rmse_3d", "RMSE (m)", "Tracking RMSE", False),
        ("cumulative_error", "Cumulative tracking error", "Cumulative tracking error", False),
        ("safety_distance_m", "Safety distance (m)", "Distance before collision / full distance", True),
        ("energy", "Total energy", "Total energy", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
    for ax, (key, ylabel, title, higher_better) in zip(axes.ravel(), metric_specs):
        for i, algo in enumerate(ALGOS):
            vals = [r[f"{algo}_{key}"] for r in selected]
            ax.bar(x + (i - 1) * width, vals, width, label=labels[algo], color=colors[algo], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Eval seed")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.text(
            0.01,
            0.95,
            "higher is better" if higher_better else "lower is better",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    axes.ravel()[0].legend(loc="best")
    fig.suptitle(
        f"{test_trajectory} selected strict seeds, {MODEL_SOURCE} best models",
        y=1.02,
    )
    fig.savefig(out_dir / "01_selected_5seeds_grouped_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    component_keys = ["thrust_energy", "delta_thrust_energy", "rudder_energy", "delta_rudder_energy"]
    component_labels = ["thrust", "delta thrust", "rudder", "delta rudder"]
    component_colors = ["#4E79A7", "#A0CBE8", "#59A14F", "#8CD17D"]
    fig, axes = plt.subplots(len(seeds), 1, figsize=(10.8, max(4.0, 2.2 * len(seeds))), constrained_layout=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, record in zip(axes, selected):
        pos = np.arange(len(ALGOS))
        bottom = np.zeros(len(ALGOS))
        for key, label, color in zip(component_keys, component_labels, component_colors):
            vals = np.array([record[f"{algo}_{key}"] for algo in ALGOS], dtype=float)
            ax.bar(pos, vals, bottom=bottom, label=label, color=color, alpha=0.95)
            bottom += vals
        ax.set_xticks(pos)
        ax.set_xticklabels([labels[a] for a in ALGOS])
        ax.set_ylabel("Energy")
        ax.set_title(f"seed {record['eval_seed']}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    fig.savefig(out_dir / "02_selected_5seeds_energy_components.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    flag_names = ["RMSE", "TrackErr", "Collision", "SafetyDist", "Energy", "Energy/m", "SACR 2nd E", "No coll"]
    flag_keys = [
        "csac_win_rmse",
        "csac_win_cumulative_error",
        "csac_win_collision",
        "csac_win_safety_distance",
        "csac_win_energy",
        "csac_win_energy_per_m",
        "csac_win_sacr_energy_second",
        "csac_win_csac_no_collision",
    ]
    flag_data = np.array([[record[key] for key in flag_keys] for record in selected], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.imshow(flag_data, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(flag_names)))
    ax.set_xticklabels(flag_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(seeds)))
    ax.set_yticklabels([str(s) for s in seeds])
    for i in range(flag_data.shape[0]):
        for j in range(flag_data.shape[1]):
            ax.text(j, i, "win" if flag_data[i, j] else "-", ha="center", va="center", fontsize=8)
    ax.set_title("CSAC win flags for selected seeds")
    ax.set_ylabel("Eval seed")
    fig.savefig(out_dir / "03_selected_5seeds_win_flags.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_selected_report(
    test_trajectory: TrajectoryKind,
    out_dir: Path,
    strict: list[dict],
    energy_order_records: list[dict],
    selected: list[dict],
) -> None:
    lines = [
        f"{test_trajectory} seed0-1000 strict CSAC screening",
        (
            f"Train models: {train_trajectory_for(test_trajectory)}; model source: {MODEL_SOURCE}; "
            f"CSAC/SAC train seed {TRAIN_SEED}; SACR train seed {SACR_TRAIN_SEED}"
        ),
        "",
        f"Strict matches for requested condition: {len(strict)}",
        "Strict condition: CSAC no collision; CSAC best/tied-best on RMSE, cumulative_error, "
        "collision, safety_distance, total_energy; total energy order CSAC <= SACR < SAC.",
        "Seeds satisfying only the energy order CSAC <= SACR < SAC: "
        + (",".join(str(r["eval_seed"]) for r in energy_order_records) if energy_order_records else "NONE"),
        "",
    ]
    if len(selected) < 5:
        lines.append("Fewer than 5 strict seeds exist in this evaluated range; selected table contains all strict seeds.")
    else:
        lines.append("The 5 seeds below satisfy the strict requested condition.")
    lines += [
        "SafetyDist definition: episode_distance_m when no collision; distance_before_collision_m when collision occurs.",
        "Win flags order: RMSE / TrackErr / Collision / SafetyDist / Energy / EnergyPerM / SACR-energy-second / CSAC-no-collision",
        "",
        "eval_seed\twin_flags\tCollision(C/S/Sr)\tRMSE(C/S/Sr)\tTrackErr(C/S/Sr)\tSafetyDist(C/S/Sr)\tEnergy(C/S/Sr)\tEnergy/m(C/S/Sr)\tPropulsion(C/S/Sr)\tSteering(C/S/Sr)",
    ]
    for r in selected:
        flags = "/".join(
            str(r[key])
            for key in [
                "csac_win_rmse",
                "csac_win_cumulative_error",
                "csac_win_collision",
                "csac_win_safety_distance",
                "csac_win_energy",
                "csac_win_energy_per_m",
                "csac_win_sacr_energy_second",
                "csac_win_csac_no_collision",
            ]
        )

        def triplet(key: str, fmt: str) -> str:
            return "/".join(fmt.format(r[f"{algo}_{key}"]) for algo in ALGOS)

        lines.append(
            f"{r['eval_seed']}\t{flags}\t"
            f"{triplet('collision', '{:.0f}')}\t"
            f"{triplet('rmse_3d', '{:.4f}')}\t"
            f"{triplet('cumulative_error', '{:.2f}')}\t"
            f"{triplet('safety_distance_m', '{:.2f}')}\t"
            f"{triplet('energy', '{:.2f}')}\t"
            f"{triplet('energy_per_m', '{:.4f}')}\t"
            f"{triplet('propulsion_energy', '{:.2f}')}\t"
            f"{triplet('steering_energy', '{:.2f}')}"
        )
    (out_dir / "selected_5seeds_report.txt").write_text("\n".join(lines), encoding="utf-8")


def finalize_scenario(test_trajectory: TrajectoryKind) -> None:
    log_dir, fig_dir = output_dirs(test_trajectory)
    write_metadata(test_trajectory)
    results = {
        algo: em._enrich_episode_metrics(
            json.loads((log_dir / f"eval_rollouts_{algo}.json").read_text(encoding="utf-8"))
        )
        for algo in ALGOS
    }
    write_eval_summaries(
        log_dir,
        {algo: {int(row["eval_seed"]): row for row in rows} for algo, rows in results.items()},
    )
    em._write_tables(
        results,
        fig_dir,
        test_trajectory,
        f"{START_SEED}-{END_SEED}",
        same_init=True,
        no_current=True,
    )
    em._plot_summary_bars(
        results,
        fig_dir,
        test_trajectory,
        f"{START_SEED}-{END_SEED}, same init, no current, {MODEL_SOURCE} models",
    )

    selected_dir = fig_dir / "strict5"
    selected_dir.mkdir(parents=True, exist_ok=True)
    records, strict, energy_order_records, selected = build_screening_records(results)
    write_records(selected_dir / "screening_all_seed0-1000.csv", records)
    write_records(selected_dir / "strict_selected_seeds.csv", strict)
    write_records(selected_dir / "energy_order_csac_sacr_sac.csv", energy_order_records)
    write_records(selected_dir / "selected_5seeds_wide.csv", selected)
    write_selected_tables(selected_dir, selected)
    write_selected_report(test_trajectory, selected_dir, strict, energy_order_records, selected)
    plot_selected(test_trajectory, selected_dir, selected)

    summary = {
        "test_trajectory": test_trajectory,
        "train_trajectory": train_trajectory_for(test_trajectory),
        "strict_match_count": len(strict),
        "strict_selected_seeds": [r["eval_seed"] for r in strict[:5]],
        "selected_5seeds": [r["eval_seed"] for r in selected],
        "energy_order_count": len(energy_order_records),
        "energy_order_seeds": [r["eval_seed"] for r in energy_order_records],
        "fig_dir": str(fig_dir.resolve()),
        "selected_dir": str(selected_dir.resolve()),
    }
    (selected_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print((selected_dir / "selected_5seeds_report.txt").read_text(encoding="utf-8"), flush=True)
    print(f"FINAL_FIG_DIR[{test_trajectory}] {fig_dir.resolve()}", flush=True)
    print(f"FINAL_SELECTED_DIR[{test_trajectory}] {selected_dir.resolve()}", flush=True)


def load_summary_row(test_trajectory: TrajectoryKind, algo: str) -> dict:
    log_dir, _ = output_dirs(test_trajectory)
    path = log_dir / f"eval_summary_{algo}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "test_trajectory": test_trajectory,
        "train_trajectory": data.get("train_trajectory", train_trajectory_for(test_trajectory)),
        "algorithm": algo.upper(),
        "num_episodes": data["num_episodes"],
        "collision_count": data["collision_count"],
        "collision_rate_pct": data["collision_rate"] * 100.0,
        "rmse_mean": data["rmse_3d_mean"],
        "rmse_std": data["rmse_3d_std"],
        "cumulative_error_mean": data["tracking_error_mean"],
        "cumulative_error_std": data["tracking_error_std"],
        "distance_before_collision_mean": data["distance_before_collision_mean"],
        "energy_mean": data["energy_mean"],
        "energy_std": data["energy_std"],
        "energy_per_m_mean": data["energy_per_m_mean"],
        "propulsion_energy_mean": data.get("propulsion_energy_mean"),
        "steering_energy_mean": data.get("steering_energy_mean"),
        "total_energy_mean": data.get("total_energy_mean", data["energy_mean"]),
    }


def write_all_scenarios_summary(trajectories: Iterable[TrajectoryKind]) -> None:
    paths = PathConfig()
    out_dir = (
        paths.figure_dir
        / "multi_seed_eval"
        / "all_scenarios"
        / f"{MODEL_SOURCE}_seed{TRAIN_SEED}_sacr_seed{SACR_TRAIN_SEED}"
        / out_tag()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    strict_rows = []
    for trajectory in trajectories:
        for algo in ALGOS:
            rows.append(load_summary_row(trajectory, algo))
        _, fig_dir = output_dirs(trajectory)
        sel_summary = json.loads(
            (
                fig_dir
                / "strict5"
                / "selection_summary.json"
            ).read_text(encoding="utf-8")
        )
        strict_rows.append(
            {
                "test_trajectory": trajectory,
                "strict_match_count": sel_summary["strict_match_count"],
                "selected_5seeds": ",".join(str(s) for s in sel_summary["selected_5seeds"]),
                "energy_order_count": sel_summary["energy_order_count"],
            }
        )

    with (out_dir / "all_scenarios_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "all_scenarios_selection_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(strict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(strict_rows)

    labels = [algo.upper() for algo in ALGOS]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    metric_specs = [
        ("rmse_mean", "Mean RMSE", "RMSE (m)"),
        ("cumulative_error_mean", "Mean cumulative tracking error", "Cumulative error"),
        ("collision_rate_pct", "Collision rate", "%"),
        ("energy_mean", "Mean total energy", "Energy"),
    ]
    scenario_list = list(trajectories)
    x = np.arange(len(scenario_list))
    width = 0.24
    for ax, (key, title, ylabel) in zip(axes.ravel(), metric_specs):
        for i, algo in enumerate(ALGOS):
            vals = [
                next(r[key] for r in rows if r["test_trajectory"] == trajectory and r["algorithm"] == algo.upper())
                for trajectory in scenario_list
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_list)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes.ravel()[0].legend(loc="best")
    fig.suptitle(f"All test scenarios, {MODEL_SOURCE} best models, eval seeds 0-1000")
    fig.savefig(out_dir / "01_all_scenarios_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"ALL_SCENARIOS_SUMMARY_DIR {out_dir.resolve()}", flush=True)


def parse_trajectories(value: str) -> list[TrajectoryKind]:
    if value == "all":
        return list(TRAJECTORIES)
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in TRAJECTORIES]
    if invalid:
        raise ValueError(f"Invalid trajectories: {invalid}")
    return items  # type: ignore[return-value]


def main() -> None:
    global MODEL_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="all", help="all or comma list: sinusoid,figure8,circle")
    parser.add_argument(
        "--model-source",
        default="sinusoid",
        choices=["sinusoid", "matched"],
        help="sinusoid = use sinusoid-trained models for all tests; matched = use each scenario's own trained models",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto", ""])
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="CPU process workers. Use 1 for serial mode.")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="Maximum new seeds per scenario for this run. 0 means no limit.",
    )
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    MODEL_SOURCE = args.model_source

    trajectories = parse_trajectories(args.trajectories)
    if args.finalize_only:
        for trajectory in trajectories:
            finalize_scenario(trajectory)
        if len(trajectories) > 1:
            write_all_scenarios_summary(trajectories)
        return

    stop_after = args.stop_after if args.stop_after > 0 else None
    if args.workers > 1:
        for trajectory in trajectories:
            run_scenario_parallel(
                trajectory,
                device=args.device or None,
                save_every=max(1, args.save_every),
                stop_after=stop_after,
                workers=args.workers,
            )
    else:
        for trajectory in trajectories:
            agents = load_agents(trajectory, args.device or None)
            run_scenario(
                trajectory,
                agents,
                save_every=max(1, args.save_every),
                stop_after=stop_after,
            )
    complete = True
    for trajectory in trajectories:
        log_dir, _ = output_dirs(trajectory)
        if not scenario_complete(load_results(log_dir)):
            complete = False
    if complete and len(trajectories) > 1:
        write_all_scenarios_summary(trajectories)


if __name__ == "__main__":
    main()
