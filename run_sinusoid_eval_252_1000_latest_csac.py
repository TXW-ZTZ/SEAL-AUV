from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import evaluate_multi_seed as em
from config import PathConfig, SimConfig
from env import AUVTrackingCMDPEnv


TRAJECTORY = "sinusoid"
TRAIN_SEED = 666
START_SEED = 252
END_SEED = 1000
SUFFIX = "latest_csac_20260617"
LATEST_CSAC_CKPT = Path(
    "checkpoints/csac_sinusoid_no_current_ep2200_eval100_seed666/"
    "sinusoid_seed666/csac_actor_seed666_best.pt"
)
LATEST_CSAC_TAG = "csac_sinusoid_no_current_ep2200_eval100_seed666/sinusoid_seed666"
ALGOS = ("csac", "sac", "sacr")


def eval_seeds() -> list[int]:
    return list(range(START_SEED, END_SEED + 1))


def out_tag() -> str:
    return f"eval_seeds_{START_SEED}-{END_SEED}_same_init_{SUFFIX}"


def output_dirs() -> tuple[Path, Path]:
    paths = PathConfig()
    log_dir = paths.log_dir / "multi_seed_eval" / TRAJECTORY / f"train_seed{TRAIN_SEED}" / out_tag()
    fig_dir = paths.figure_dir / "multi_seed_eval" / TRAJECTORY / f"train_seed{TRAIN_SEED}" / out_tag()
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, fig_dir


def algo_config() -> dict:
    cfg = em._algo_config(TRAJECTORY)
    cfg["csac"] = dict(cfg["csac"])
    cfg["csac"]["train_tag"] = LATEST_CSAC_TAG
    cfg["csac"]["checkpoint"] = str(LATEST_CSAC_CKPT)
    return cfg


def write_metadata(log_dir: Path, fig_dir: Path) -> None:
    cfg = algo_config()
    metadata = {
        "trajectory": TRAJECTORY,
        "train_seed": TRAIN_SEED,
        "eval_seeds": eval_seeds(),
        "same_init": True,
        "episode_steps": 1000,
        "current_intensity": 0,
        "models": {
            algo: {
                "train_tag": cfg[algo]["train_tag"],
                "checkpoint": str(Path(cfg[algo]["checkpoint"]).resolve()),
            }
            for algo in ALGOS
        },
    }
    text = json.dumps(metadata, indent=2)
    (log_dir / "model_paths.json").write_text(text, encoding="utf-8")
    (fig_dir / "model_paths.json").write_text(text, encoding="utf-8")


def load_results(log_dir: Path) -> dict[str, dict[int, dict]]:
    loaded: dict[str, dict[int, dict]] = {a: {} for a in ALGOS}
    for algo in ALGOS:
        path = log_dir / f"eval_rollouts_{algo}.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            loaded[algo] = {int(r["eval_seed"]): r for r in rows}
    return loaded


def save_results(log_dir: Path, all_results: dict[str, dict[int, dict]]) -> None:
    for algo in ALGOS:
        rows = [all_results[algo][s] for s in sorted(all_results[algo])]
        (log_dir / f"eval_rollouts_{algo}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )


def write_eval_summaries(log_dir: Path, all_results: dict[str, dict[int, dict]]) -> None:
    combined = {}
    for algo in ALGOS:
        rows = [all_results[algo][s] for s in sorted(all_results[algo])]
        if not rows:
            continue
        summary = em._summarize_results(rows)
        summary["algorithm"] = algo
        summary["train_seed"] = TRAIN_SEED
        summary["eval_seeds"] = [int(r["eval_seed"]) for r in rows]
        summary["same_init"] = True
        summary["current_intensity"] = 0
        combined[algo] = summary
        (log_dir / f"eval_summary_{algo}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    (log_dir / "eval_summary_all.json").write_text(
        json.dumps({"same_init": True, "current_intensity": 0, "algorithms": combined}, indent=2),
        encoding="utf-8",
    )


def complete_dict_to_lists(all_results: dict[str, dict[int, dict]]) -> dict[str, list[dict]]:
    return {algo: [all_results[algo][s] for s in sorted(all_results[algo])] for algo in ALGOS}


def load_agents(device: str | None):
    agents = {}
    for algo, cfg in algo_config().items():
        ckpt = Path(cfg["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {algo}: {ckpt}")
        print(f"Loading {algo.upper()} - {ckpt}", flush=True)
        agents[algo] = em._load_agent(algo, ckpt, device)
    return agents


def run_eval(device: str | None, save_every: int, stop_after: int | None = None) -> None:
    log_dir, fig_dir = output_dirs()
    write_metadata(log_dir, fig_dir)
    all_results = load_results(log_dir)
    agents = load_agents(device)
    sim_cfg = SimConfig(seed=TRAIN_SEED, trajectory=TRAJECTORY, current_intensity=0)
    sampler_env = AUVTrackingCMDPEnv(sim_cfg=sim_cfg, randomize_start=True)
    target = eval_seeds()

    completed_this_run = 0
    for idx, seed in enumerate(target, start=1):
        if all(seed in all_results[a] for a in ALGOS):
            if idx % 25 == 0:
                print(f"[skip] seed {seed}; progress {idx}/{len(target)}", flush=True)
            continue
        if stop_after is not None and completed_this_run >= stop_after:
            print(f"[stop] reached stop-after={stop_after}; progress {idx - 1}/{len(target)}", flush=True)
            break

        sampler_env.reset(seed=seed)
        init_state = sampler_env.capture_init_state()
        init_snapshot = em._init_snapshot_from_state(init_state)
        print(f"[eval] seed {seed} ({idx}/{len(target)})", flush=True)

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
                trajectory=TRAJECTORY,
                gamma=cfg.gamma,
                eval_seed=seed,
                init_snapshot=init_snapshot,
                save_rollouts=False,
            )
            ep["algorithm"] = algo
            all_results[algo][seed] = ep
            coll = "COLL" if ep["collision"] else "OK"
            print(
                f"  {algo.upper()}: rmse={ep['rmse_3d']:.3f}, "
                f"track={ep['cumulative_error']:.1f}, energy={ep['undiscounted_energy']:.3f}, {coll}",
                flush=True,
            )

        if idx % save_every == 0 or idx == len(target):
            save_results(log_dir, all_results)
            write_eval_summaries(log_dir, all_results)
            print(f"[save] progress {idx}/{len(target)}", flush=True)
        completed_this_run += 1

    save_results(log_dir, all_results)
    write_eval_summaries(log_dir, all_results)
    if all(all(seed in all_results[a] for a in ALGOS) for seed in target):
        finalize_outputs()
    else:
        done = sum(1 for seed in target if all(seed in all_results[a] for a in ALGOS))
        print(f"[partial] completed {done}/{len(target)} seeds; rerun to continue", flush=True)


def metric_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def collision(row: dict) -> int:
    return int(float(row["collision"]))


def safety_distance(row: dict) -> float:
    if collision(row) == 0:
        return float(row["episode_distance_m"])
    value = metric_float(row, "distance_before_collision_m")
    return 0.0 if value is None else value


def build_screening_records(csv_path: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    by_seed: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["eval_seed"]), {})[row["algorithm"]] = row

    def best_min(triplet: dict[str, dict], key: str) -> bool:
        c = metric_float(triplet["csac"], key)
        return c <= min(metric_float(triplet[a], key) for a in ("sac", "sacr"))

    def best_collision(triplet: dict[str, dict]) -> bool:
        c = collision(triplet["csac"])
        return c <= min(collision(triplet[a]) for a in ("sac", "sacr"))

    def best_safety(triplet: dict[str, dict]) -> bool:
        c = safety_distance(triplet["csac"])
        return c >= max(safety_distance(triplet[a]) for a in ("sac", "sacr"))

    def sacr_energy_second(triplet: dict[str, dict]) -> bool:
        ordered = sorted((metric_float(triplet[a], "undiscounted_energy"), a) for a in ALGOS)
        return ordered[1][1] == "sacr"

    def energy_order(triplet: dict[str, dict]) -> bool:
        e_csac = metric_float(triplet["csac"], "undiscounted_energy")
        e_sac = metric_float(triplet["sac"], "undiscounted_energy")
        e_sacr = metric_float(triplet["sacr"], "undiscounted_energy")
        return e_csac <= e_sacr < e_sac

    def score(triplet: dict[str, dict], flags: dict[str, bool]) -> float:
        s = 0.0
        s += 160.0 * flags["energy"]
        s += 150.0 * flags["rmse"]
        s += 130.0 * flags["safety_distance"]
        s += 120.0 * flags["cumulative_error"]
        s += 90.0 * flags["collision"]
        s += 80.0 * flags["energy_order_csac_le_sacr_lt_sac"]
        s += 70.0 * flags["sacr_energy_second"]
        s += 100.0 * flags["csac_no_collision"]
        if not flags["csac_no_collision"]:
            s -= 160.0
        s += max(
            0.0,
            min(metric_float(triplet[a], "undiscounted_energy") for a in ("sac", "sacr"))
            - metric_float(triplet["csac"], "undiscounted_energy"),
        ) * 24.0
        s += max(
            0.0,
            min(metric_float(triplet[a], "rmse_3d") for a in ("sac", "sacr"))
            - metric_float(triplet["csac"], "rmse_3d"),
        ) * 20.0
        s += max(
            0.0,
            min(metric_float(triplet[a], "cumulative_error") for a in ("sac", "sacr"))
            - metric_float(triplet["csac"], "cumulative_error"),
        ) * 0.03
        s += max(
            0.0,
            safety_distance(triplet["csac"]) - max(safety_distance(triplet[a]) for a in ("sac", "sacr")),
        ) * 0.3
        return s

    records: list[dict] = []
    strict: list[dict] = []
    energy_order_records: list[dict] = []
    for seed in sorted(by_seed):
        triplet = by_seed[seed]
        flags = {
            "rmse": best_min(triplet, "rmse_3d"),
            "cumulative_error": best_min(triplet, "cumulative_error"),
            "collision": best_collision(triplet),
            "safety_distance": best_safety(triplet),
            "energy": best_min(triplet, "undiscounted_energy"),
            "energy_per_m": best_min(triplet, "energy_per_m"),
            "sacr_energy_second": sacr_energy_second(triplet),
            "energy_order_csac_le_sacr_lt_sac": energy_order(triplet),
            "csac_no_collision": collision(triplet["csac"]) == 0,
        }
        record = {
            "eval_seed": seed,
            "score": score(triplet, flags),
            **{f"csac_win_{k}": int(v) for k, v in flags.items()},
        }
        for algo in ALGOS:
            row = triplet[algo]
            record.update(
                {
                    f"{algo}_rmse_3d": metric_float(row, "rmse_3d"),
                    f"{algo}_cumulative_error": metric_float(row, "cumulative_error"),
                    f"{algo}_collision": collision(row),
                    f"{algo}_safety_distance_m": safety_distance(row),
                    f"{algo}_distance_before_collision_m": metric_float(row, "distance_before_collision_m"),
                    f"{algo}_episode_distance_m": metric_float(row, "episode_distance_m"),
                    f"{algo}_energy": metric_float(row, "undiscounted_energy"),
                    f"{algo}_energy_per_m": metric_float(row, "energy_per_m"),
                    f"{algo}_success": int(float(row["success"])),
                }
            )
        records.append(record)
        if (
            flags["rmse"]
            and flags["cumulative_error"]
            and flags["collision"]
            and flags["safety_distance"]
            and flags["energy"]
        ):
            strict.append(record)
        if flags["energy_order_csac_le_sacr_lt_sac"]:
            energy_order_records.append(record)

    strict.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    energy_order_records.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    selected = select_showcase(records, strict, energy_order_records)
    return records, strict, energy_order_records, selected


def select_showcase(records: list[dict], strict: list[dict], energy_order_records: list[dict]) -> list[dict]:
    no_collision_rmse_safety_energy = [
        r
        for r in records
        if r["csac_win_csac_no_collision"]
        and r["csac_win_rmse"]
        and r["csac_win_safety_distance"]
        and r["csac_win_energy"]
    ]
    no_collision_rmse_safety_energy.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    no_collision_rmse_safety = [
        r for r in records if r["csac_win_csac_no_collision"] and r["csac_win_rmse"] and r["csac_win_safety_distance"]
    ]
    no_collision_rmse_safety.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    energy_track = [r for r in records if r["csac_win_energy"] and r["csac_win_cumulative_error"]]
    energy_track.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    energy_best = [r for r in records if r["csac_win_energy"]]
    energy_best.sort(key=lambda r: (-r["score"], r["eval_seed"]))
    other_best = sorted(records, key=lambda r: (-r["score"], r["eval_seed"]))
    selected: list[dict] = []
    for pool in (
        strict,
        no_collision_rmse_safety_energy,
        no_collision_rmse_safety,
        energy_track,
        energy_order_records,
        energy_best,
        other_best,
    ):
        for record in pool:
            if record["eval_seed"] not in [r["eval_seed"] for r in selected]:
                selected.append(record)
            if len(selected) == 5:
                return selected
    return selected


def write_records(path: Path, records: list[dict]) -> None:
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
        "csac_rmse_3d",
        "sac_rmse_3d",
        "sacr_rmse_3d",
        "csac_cumulative_error",
        "sac_cumulative_error",
        "sacr_cumulative_error",
        "csac_collision",
        "sac_collision",
        "sacr_collision",
        "csac_safety_distance_m",
        "sac_safety_distance_m",
        "sacr_safety_distance_m",
        "csac_energy",
        "sac_energy",
        "sacr_energy",
        "csac_energy_per_m",
        "sac_energy_per_m",
        "sacr_energy_per_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_selected(out_dir: Path, selected: list[dict]) -> None:
    selected_seeds = [r["eval_seed"] for r in selected]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    labels = ["CSAC", "SAC", "SACR"]
    metric_plots = [
        ("rmse_3d", "RMSE (m)", "Tracking RMSE"),
        ("cumulative_error", "Cumulative tracking error", "Cumulative tracking error"),
        ("safety_distance_m", "Safety distance (m)", "Distance before collision / full distance"),
        ("energy", "Energy", "Total energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    axes = axes.ravel()
    x = np.arange(len(selected_seeds))
    width = 0.24
    for ax, (key, ylabel, title) in zip(axes, metric_plots):
        for i, algo in enumerate(ALGOS):
            vals = [r[f"{algo}_{key}"] for r in selected]
            ax.bar(x + (i - 1) * width, vals, width, label=labels[i], color=colors[i], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in selected_seeds])
        ax.set_xlabel("Eval seed")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle("Sinusoid selected seed252-1000 comparison (best models, train seed 666)")
    fig.tight_layout()
    fig.savefig(out_dir / "01_selected_5seeds_grouped_metrics.png", dpi=160)
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
    flag_data = np.array([[r[k] for k in flag_keys] for r in selected], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.imshow(flag_data, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(flag_names)))
    ax.set_xticklabels(flag_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(selected_seeds)))
    ax.set_yticklabels([str(s) for s in selected_seeds])
    for i in range(flag_data.shape[0]):
        for j in range(flag_data.shape[1]):
            ax.text(j, i, "win" if flag_data[i, j] else "-", ha="center", va="center", fontsize=8)
    ax.set_title("CSAC win flags for selected seeds")
    ax.set_ylabel("Eval seed")
    fig.tight_layout()
    fig.savefig(out_dir / "02_selected_5seeds_win_flags.png", dpi=160)
    plt.close(fig)


def write_selected_report(out_dir: Path, strict: list[dict], energy_order_records: list[dict], selected: list[dict]) -> None:
    lines = [
        "Sinusoid seed252-1000 CSAC advantage screening (latest CSAC, train seed 666)",
        "",
        f"Strict matches for all requested advantages: {len(strict)}",
        "Seeds satisfying energy order CSAC <= SACR < SAC: "
        + (",".join(str(r["eval_seed"]) for r in energy_order_records) if energy_order_records else "NONE"),
    ]
    if len(strict) < 5:
        lines.append(
            "Because fewer than 5 strict matches exist, the 5 seeds below are display-oriented candidates; "
            "each row explicitly marks which metrics CSAC wins."
        )
    else:
        lines.append("The 5 seeds below satisfy the strict requested condition.")
    lines += [
        "",
        "SafetyDist definition: episode_distance_m when no collision; distance_before_collision_m when collision occurs.",
        "Win flags order: RMSE / TrackErr / Collision / SafetyDist / Energy / EnergyPerM / SACR-energy-second / CSAC-no-collision",
        "",
        "eval_seed\twin_flags\tCollision(C/S/Sr)\tRMSE(C/S/Sr)\tTrackErr(C/S/Sr)\tSafetyDist(C/S/Sr)\tEnergy(C/S/Sr)\tEnergy/m(C/S/Sr)",
    ]
    for r in selected:
        flags = "/".join(
            str(r[k])
            for k in [
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
        lines.append(
            f"{r['eval_seed']}\t{flags}\t"
            f"{r['csac_collision']}/{r['sac_collision']}/{r['sacr_collision']}\t"
            f"{r['csac_rmse_3d']:.4f}/{r['sac_rmse_3d']:.4f}/{r['sacr_rmse_3d']:.4f}\t"
            f"{r['csac_cumulative_error']:.2f}/{r['sac_cumulative_error']:.2f}/{r['sacr_cumulative_error']:.2f}\t"
            f"{r['csac_safety_distance_m']:.2f}/{r['sac_safety_distance_m']:.2f}/{r['sacr_safety_distance_m']:.2f}\t"
            f"{r['csac_energy']:.2f}/{r['sac_energy']:.2f}/{r['sacr_energy']:.2f}\t"
            f"{r['csac_energy_per_m']:.4f}/{r['sac_energy_per_m']:.4f}/{r['sacr_energy_per_m']:.4f}"
        )
    (out_dir / "selected_5seeds_report.txt").write_text("\n".join(lines), encoding="utf-8")


def finalize_outputs() -> None:
    log_dir, fig_dir = output_dirs()
    all_results = em._enrich_episode_metrics(json.loads((log_dir / "eval_rollouts_csac.json").read_text(encoding="utf-8")))
    del all_results
    results_by_algo = {
        algo: em._enrich_episode_metrics(json.loads((log_dir / f"eval_rollouts_{algo}.json").read_text(encoding="utf-8")))
        for algo in ALGOS
    }
    em._write_tables(
        results_by_algo,
        fig_dir,
        TRAJECTORY,
        f"{START_SEED}-{END_SEED}",
        same_init=True,
        no_current=True,
    )
    em._plot_summary_bars(results_by_algo, fig_dir, TRAJECTORY, f"{START_SEED}-{END_SEED}, same init, no current")

    out_dir = fig_dir / "csac_advantage_seed252-1000_5seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, strict, energy_order_records, selected = build_screening_records(fig_dir / "metrics_all.csv")
    records_sorted = sorted(records, key=lambda r: (-r["score"], r["eval_seed"]))
    write_records(out_dir / "screening_all_seed252-1000.csv", records_sorted)
    write_records(out_dir / "strict_selected_seeds.csv", strict)
    write_records(out_dir / "energy_order_csac_sacr_sac.csv", energy_order_records)
    write_records(out_dir / "selected_5seeds.csv", selected)

    with (out_dir / "selected_5seeds_long.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_seed",
                "algorithm",
                "rmse_3d",
                "cumulative_error",
                "collision",
                "safety_distance_m",
                "energy",
                "energy_per_m",
                "success",
            ],
        )
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
                        "energy": record[f"{algo}_energy"],
                        "energy_per_m": record[f"{algo}_energy_per_m"],
                        "success": record[f"{algo}_success"],
                    }
                )

    summary = {
        "eval_seeds": f"{START_SEED}-{END_SEED}",
        "strict_condition": "CSAC best on RMSE, cumulative_error, collision, safety_distance, total_energy",
        "strict_match_count": len(strict),
        "strict_selected_seeds": [r["eval_seed"] for r in strict[:5]],
        "energy_order_count": len(energy_order_records),
        "energy_order_seeds": [r["eval_seed"] for r in energy_order_records],
        "selected_5seeds": [r["eval_seed"] for r in selected],
        "selection_policy": (
            "strict if available; otherwise prioritize no-collision RMSE+safety+energy, then no-collision "
            "RMSE+safety, then energy+tracking/energy-order examples with caveats"
        ),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_selected_report(out_dir, strict, energy_order_records, selected)
    plot_selected(out_dir, selected)
    print((out_dir / "selected_5seeds_report.txt").read_text(encoding="utf-8"), flush=True)
    print(f"FINAL_FIG_DIR {fig_dir.resolve()}", flush=True)
    print(f"FINAL_SELECTED_DIR {out_dir.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto", ""])
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        finalize_outputs()
    else:
        stop_after = args.stop_after if args.stop_after > 0 else None
        run_eval(args.device or None, save_every=max(1, args.save_every), stop_after=stop_after)


if __name__ == "__main__":
    main()
