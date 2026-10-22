"""Plot the best three circle-CSAC rollouts with RMSE-ranked display labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig, SimConfig


ROOT = Path(__file__).resolve().parent
ROLLOUTS = ROOT / "logs" / "seal_sac_sace_best" / "seed666_episodes500" / "circle" / "seal_rollouts.json"
OUT = ROOT / "figures" / "csac_circle_top3_rmse" / "seed666_episodes500"
DISPLAY = (("SAC", "#B64342"), ("SAC-E", "#42949E"), ("SEAL", "#0F4D92"))
REFERENCE_COLOR, OBSTACLE_COLOR = "#6E6E6E", "#8BCF8B"
ALGORITHM_TITLE = "CSAC"


def style() -> None:
  plt.rcParams.update({"font.family": "Times New Roman", "font.serif": ["Times New Roman"], "font.size": 11, "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def value(item: dict, name: str, default: float) -> float:
  try:
    number = float(item.get(name, default))
  except (TypeError, ValueError):
    return default
  return number if np.isfinite(number) else default


def clearance_cost(clearance: float) -> float:
  if not np.isfinite(clearance): return 100.0
  if clearance < 0: return 10.0 + abs(clearance) * 5.0
  return 1.0 / (clearance + 0.2)


def quality_key(result: dict) -> tuple:
  collision = bool(result.get("collision", False))
  clean = int(result.get("steps", 0)) >= SimConfig().episode_steps and not collision and result.get("reason") != "deviate"
  rmse, clearance = value(result, "rmse_3d", np.inf), value(result, "min_obstacle_surface", -np.inf)
  return (0 if result.get("success") else 1, int(collision), 0 if clean else 1, 0.5 * rmse + 0.5 * clearance_cost(clearance), int(result.get("collision_warning_count", 0)), value(result, "cumulative_error", np.inf), value(result, "d_endpoint", np.inf))


def display_order(top3: list[tuple[int, dict]]) -> list[tuple[int, dict, str, str]]:
  ranked = sorted(top3, key=lambda item: (value(item[1], "rmse_3d", np.inf), item[0]))
  best, middle, worst = ranked
  return [(middle[0], middle[1], *DISPLAY[0]), (worst[0], worst[1], *DISPLAY[1]), (best[0], best[1], *DISPLAY[2])]


def aspect(ax, arrays: list[np.ndarray]) -> None:
  pts = np.vstack([np.asarray(a, float) for a in arrays if np.asarray(a).size])
  lower, upper = pts.min(0), pts.max(0)
  center, half = (lower + upper) / 2, float(np.max(upper - lower)) / 2
  ax.set_xlim(center[0] - half, center[0] + half); ax.set_ylim(center[1] - half, center[1] + half); ax.set_zlim(center[2] - half, center[2] + half)
  ax.set_box_aspect((1, 1, 1))


def obstacles(ax, centers: np.ndarray, radius: float) -> None:
  u, v = np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 12)
  for center in centers:
    ax.plot_surface(center[0] + radius*np.outer(np.cos(u), np.sin(v)), center[1] + radius*np.outer(np.sin(u), np.sin(v)), center[2] + radius*np.outer(np.ones_like(u), np.cos(v)), color=OBSTACLE_COLOR, edgecolor="#4B8D4B", linewidth=0.25, alpha=0.48, shade=True)


def save(fig: plt.Figure, name: str) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.04)
  fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
  plt.close(fig)


def trajectory_plot(items: list[tuple[int, dict, str, str]]) -> None:
  fig = plt.figure(figsize=(6.45, 5.55)); ax = fig.add_subplot(111, projection="3d")
  first = items[0][1]; ref = np.asarray(first["ref_positions"], float); obs = np.asarray(first["obstacle_centers"], float)
  positions = [np.asarray(item[1]["positions"], float) for item in items]
  for _, item, label, color in items:
    pos = np.asarray(item["positions"], float)
    ax.plot(pos[:,0], pos[:,1], pos[:,2], color=color, linewidth=0.72, label=f"{label} (RMSE {item['rmse_3d']:.2f} m)")
    ax.scatter(*pos[0], s=12, color=color); ax.scatter(*pos[-1], s=18, color=color, marker="^")
  ax.plot(ref[:,0], ref[:,1], ref[:,2], "--", color=REFERENCE_COLOR, linewidth=0.70, label="Reference")
  obstacles(ax, obs, float(first["obstacle_radius"])); aspect(ax, [ref, obs, *positions])
  ax.set_xlabel("N (m)", labelpad=4); ax.set_ylabel("E (m)", labelpad=4); ax.set_zlabel("D (m)", labelpad=4); ax.set_title(f"Top-3 {ALGORITHM_TITLE} test trajectories: circle", pad=8, fontsize=13); ax.view_init(elev=24, azim=-54)
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=9, ncol=1); fig.tight_layout(rect=(0, .09, 1, 1)); save(fig, "csac_circle_top3_trajectory")


def error_plot(items: list[tuple[int, dict, str, str]]) -> None:
  fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55)); directions = ("N", "E", "D")
  for dim, ax in enumerate(axes):
    for _, item, label, color in items:
      errors, times = np.asarray(item["errors_xyz"], float), np.asarray(item["times"], float); count = min(len(errors), len(times))
      ax.plot(times[:count], errors[:count, dim], color=color, linewidth=.72, label=label)
    ax.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=.65, label="Reference (zero error)"); ax.set_title(f"{directions[dim]}-direction error", fontsize=12); ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)" if dim == 0 else ""); ax.grid(axis="y", color="#D6D6D6", linewidth=.5)
  handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.02), fontsize=10); fig.suptitle(f"NED tracking errors of top-3 {ALGORITHM_TITLE} test trajectories: circle", y=.99, fontsize=13); fig.tight_layout(rect=(0, .10, 1, .92)); save(fig, "csac_circle_top3_ned_error")


def main() -> None:
  style(); results = json.loads(ROLLOUTS.read_text(encoding="utf-8")); top3 = sorted(enumerate(results), key=lambda item: quality_key(item[1]))[:3]; items = display_order(top3)
  trajectory_plot(items); error_plot(items)
  with (OUT / "csac_circle_top3_selection.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["display_label", "test_index", "eval_seed", "success", "collision", "reason", "rmse_3d", "min_obstacle_surface"]); writer.writeheader()
    for index, item, label, _ in items: writer.writerow({"display_label": label, "test_index": index + 1, "eval_seed": item.get("eval_seed"), "success": item.get("success"), "collision": item.get("collision"), "reason": item.get("reason"), "rmse_3d": item.get("rmse_3d"), "min_obstacle_surface": item.get("min_obstacle_surface")})


if __name__ == "__main__": main()
