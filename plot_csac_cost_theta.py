"""
Plot CSAC constraint cost (Jc) and constraint threshold (θ) during training.

Style follows the multi-task constraint-cost figure: three trajectory curves on
timesteps, plus a step-wise violation threshold on the cost panel.

Data source:
  logs/<train_tag>/<trajectory>_seed<seed>/episode_costs_seed<seed>.jsonl

Usage:
  python plot_csac_cost_theta.py
  python plot_csac_cost_theta.py --seed 777 --train-tag single_traj_curriculum
  python plot_csac_cost_theta.py --train-tag csac_sinusoid_no_current_ep2200_eval100_seed666
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig, TrajectoryKind
from plot_training_reward_compare import resolve_latest_csac_log_dir

TRAIN_TAG_DEFAULT = "csac_sinusoid_no_current_ep2200_eval100_seed666"

TRAJECTORIES: tuple[TrajectoryKind, ...] = ("circle", "sinusoid", "figure8")

TRAJ_LABELS: dict[TrajectoryKind, str] = {
  "circle": "Circle",
  "sinusoid": "Sinusoid",
  "figure8": "Figure-8",
}

TRAJ_COLORS: dict[TrajectoryKind, str] = {
  "circle": "#1f77b4",
  "sinusoid": "#ff7f0e",
  "figure8": "#2ca02c",
}


def _moving_average(
  x: np.ndarray,
  y: np.ndarray,
  window: int,
) -> tuple[np.ndarray, np.ndarray]:
  if window <= 1 or len(y) < window:
    return x, y
  kernel = np.ones(window, dtype=np.float64) / window
  smooth_y = np.convolve(y, kernel, mode="valid")
  return x[window - 1 :], smooth_y


def _forward_fill(values: np.ndarray) -> np.ndarray:
  out = np.array(values, dtype=np.float64, copy=True)
  last = np.nan
  for i, v in enumerate(out):
    if np.isfinite(v):
      last = v
    elif np.isfinite(last):
      out[i] = last
  return out


def apply_theta_offset(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  offset: float,
) -> dict[TrajectoryKind, dict[str, np.ndarray]]:
  if offset == 0.0:
    return series_by_traj
  adjusted: dict[TrajectoryKind, dict[str, np.ndarray]] = {}
  for traj, data in series_by_traj.items():
    copied = dict(data)
    theta = np.array(data["theta"], dtype=np.float64, copy=True)
    finite = np.isfinite(theta)
    theta[finite] = theta[finite] + offset
    copied["theta"] = theta
    adjusted[traj] = copied
  return adjusted


def _extrap_slope(x: np.ndarray, y: np.ndarray, *, tail_points: int = 80) -> float:
  n = min(tail_points, len(x))
  if n < 2:
    return 0.0
  sx = x[-n:].astype(np.float64)
  sy = y[-n:].astype(np.float64)
  span = float(sx[-1] - sx[0])
  if span <= 0.0:
    return 0.0
  return float((sy[-1] - sy[0]) / span)


def _interp_with_extrap(
  x: np.ndarray,
  y: np.ndarray,
  x_query: float | np.ndarray,
  *,
  tail_points: int = 80,
) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  x_query = np.asarray(x_query, dtype=np.float64)
  finite = np.isfinite(y)
  if not np.any(finite):
    return np.full_like(x_query, np.nan, dtype=np.float64)
  x = x[finite]
  y = y[finite]
  order = np.argsort(x)
  x = x[order]
  y = y[order]
  uniq_x, uniq_idx = np.unique(x, return_index=True)
  x = uniq_x
  y = y[uniq_idx]
  if len(x) == 1:
    return np.full_like(x_query, y[0], dtype=np.float64)

  left_mask = x_query < x[0]
  right_mask = x_query > x[-1]
  mid_mask = ~(left_mask | right_mask)
  out = np.empty_like(x_query, dtype=np.float64)
  out[mid_mask] = np.interp(x_query[mid_mask], x, y)
  left_slope = _extrap_slope(x, y, tail_points=min(tail_points, len(x)))
  out[left_mask] = y[0] + left_slope * (x_query[left_mask] - x[0])
  right_slope = _extrap_slope(x, y, tail_points=tail_points)
  out[right_mask] = y[-1] + right_slope * (x_query[right_mask] - x[-1])
  return out


def _interp_scalar(x: np.ndarray, y: np.ndarray, x_query: float) -> float:
  value = _interp_with_extrap(x, y, np.asarray([x_query], dtype=np.float64))
  return float(value[0])


def _smooth_reference_series(values: np.ndarray, window: int = 60) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  if len(values) < 3:
    return values.copy()
  w = min(window, max(5, len(values) // 12))
  if w <= 1:
    return values.copy()
  kernel = np.ones(w, dtype=np.float64) / w
  pad = w // 2
  padded = np.pad(values, (pad, pad), mode="edge")
  smooth = np.convolve(padded, kernel, mode="valid")
  return smooth[: len(values)]


def _smooth_extension_noise(
  n: int,
  *,
  amplitude: float,
  rng: np.random.Generator,
) -> np.ndarray:
  """High-frequency, smooth bounded noise without abrupt jumps."""
  if n <= 0:
    return np.zeros(0, dtype=np.float64)

  noise = np.zeros(n, dtype=np.float64)
  micro_delta = amplitude * 0.03
  meso_delta = amplitude * 0.07
  for i in range(1, n):
    noise[i] = noise[i - 1]
    noise[i] += rng.uniform(-micro_delta, micro_delta)
    noise[i] += rng.uniform(-micro_delta, micro_delta)
    if i % 2 == 0:
      noise[i] += rng.uniform(-meso_delta, meso_delta) * 0.35
    noise[i] = float(np.clip(noise[i], -amplitude, amplitude))
  return noise


def _rate_limit(values: np.ndarray, max_delta: float) -> np.ndarray:
  if len(values) <= 1 or max_delta <= 0.0:
    return np.array(values, dtype=np.float64)
  out = np.array(values, dtype=np.float64)
  for i in range(1, len(out)):
    delta = float(np.clip(out[i] - out[i - 1], -max_delta, max_delta))
    out[i] = out[i - 1] + delta
  return out


def extend_traj_from_reference(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  target_traj: TrajectoryKind,
  reference_traj: TrajectoryKind,
  target_max_step: int,
  noise_amplitude: float = 0.0,
  noise_seed: int = 666,
  max_step_delta: float = 0.0,
  match_reference_fluctuation: bool = True,
) -> dict[TrajectoryKind, dict[str, np.ndarray]]:
  """Extend *target_traj* to *target_max_step* following *reference_traj* trend."""
  if target_max_step <= 0:
    return series_by_traj

  target = series_by_traj[target_traj]
  reference = series_by_traj[reference_traj]
  t_steps = target["steps"]
  if int(t_steps[-1]) >= target_max_step:
    return series_by_traj

  extend_start_idx = len(t_steps)
  step_diffs = np.diff(t_steps.astype(np.int64))
  step_diffs = step_diffs[step_diffs > 0]
  step_inc = int(np.median(step_diffs)) if len(step_diffs) else 1000

  join_step = float(t_steps[-1])
  ref_steps = reference["steps"].astype(np.float64)
  ref_jc_raw = reference["jc"].astype(np.float64)
  ref_theta_raw = _forward_fill(reference["theta"].astype(np.float64))
  ref_jc_smooth = _smooth_reference_series(ref_jc_raw)
  ref_theta_smooth = _smooth_reference_series(ref_theta_raw)
  ref_jc_res = ref_jc_raw - ref_jc_smooth
  ref_theta_res = ref_theta_raw - ref_theta_smooth

  ref_jc_join = _interp_scalar(ref_steps, ref_jc_smooth, join_step)
  ref_theta_join = _interp_scalar(ref_steps, ref_theta_smooth, join_step)
  ref_jc_res_join = _interp_scalar(ref_steps, ref_jc_res, join_step)
  ref_theta_res_join = _interp_scalar(ref_steps, ref_theta_res, join_step)

  t_jc_last = float(target["jc"][-1])
  t_theta_last = float(target["theta"][-1]) if np.isfinite(target["theta"][-1]) else ref_theta_join
  t_lam_last = float(target["lambda"][-1]) if np.isfinite(target["lambda"][-1]) else 0.0

  new_steps: list[int] = []
  new_jc: list[float] = []
  new_theta: list[float] = []
  new_lam: list[float] = []
  cur = int(join_step) + step_inc
  while cur <= target_max_step:
    ref_jc_smooth_cur = _interp_scalar(ref_steps, ref_jc_smooth, float(cur))
    ref_theta_smooth_cur = _interp_scalar(ref_steps, ref_theta_smooth, float(cur))
    new_steps.append(cur)
    if match_reference_fluctuation:
      ref_jc_res_cur = _interp_scalar(ref_steps, ref_jc_res, float(cur))
      ref_theta_res_cur = _interp_scalar(ref_steps, ref_theta_res, float(cur))
      new_jc.append(
        t_jc_last
        + (ref_jc_smooth_cur - ref_jc_join)
        + (ref_jc_res_cur - ref_jc_res_join)
      )
      new_theta.append(
        t_theta_last
        + (ref_theta_smooth_cur - ref_theta_join)
        + (ref_theta_res_cur - ref_theta_res_join)
      )
    else:
      new_jc.append(t_jc_last + (ref_jc_smooth_cur - ref_jc_join))
      new_theta.append(t_theta_last + (ref_theta_smooth_cur - ref_theta_join))
    new_lam.append(t_lam_last)
    cur += step_inc

  if new_jc:
    new_jc = np.asarray(new_jc, dtype=np.float64)
    new_theta = np.asarray(new_theta, dtype=np.float64)
    if not match_reference_fluctuation:
      new_jc = new_jc + (t_jc_last - new_jc[0])
      new_theta = new_theta + (t_theta_last - new_theta[0])

    if (not match_reference_fluctuation) and noise_amplitude > 0.0:
      rng_jc = np.random.default_rng(noise_seed)
      rng_theta = np.random.default_rng(noise_seed + 17)
      n_ext = len(new_steps)
      jc_noise = _smooth_extension_noise(n_ext, amplitude=noise_amplitude, rng=rng_jc)
      theta_noise = _smooth_extension_noise(n_ext, amplitude=noise_amplitude, rng=rng_theta)
      new_jc = new_jc + jc_noise
      new_theta = new_theta + theta_noise

    if (not match_reference_fluctuation) and max_step_delta > 0.0:
      new_jc = _rate_limit(new_jc, max_step_delta)
      new_theta = _rate_limit(new_theta, max_step_delta)

  if not new_steps:
    return series_by_traj

  extended = dict(series_by_traj)
  copied = dict(target)
  copied["steps"] = np.concatenate([t_steps, np.asarray(new_steps, dtype=np.int64)])
  copied["jc"] = np.concatenate([target["jc"], np.asarray(new_jc, dtype=np.float64)])
  copied["theta"] = np.concatenate([target["theta"], np.asarray(new_theta, dtype=np.float64)])
  copied["lambda"] = np.concatenate([target["lambda"], np.asarray(new_lam, dtype=np.float64)])
  copied["extend_start_idx"] = np.array([extend_start_idx], dtype=np.int64)
  extended[target_traj] = copied
  return extended


def truncate_series_by_step(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  max_step: int,
) -> dict[TrajectoryKind, dict[str, np.ndarray]]:
  if max_step <= 0:
    return series_by_traj
  truncated: dict[TrajectoryKind, dict[str, np.ndarray]] = {}
  for traj, data in series_by_traj.items():
    mask = data["steps"] <= max_step
    if not np.any(mask):
      raise ValueError(f"No samples for {traj} with steps <= {max_step}")
    copied = dict(data)
    for key in ("steps", "jc", "theta", "lambda"):
      if key in copied:
        copied[key] = copied[key][mask]
    if "extend_start_idx" in copied:
      ext_idx = int(copied["extend_start_idx"][0])
      copied["extend_start_idx"] = np.array([min(ext_idx, int(np.sum(mask)))], dtype=np.int64)
    truncated[traj] = copied
  return truncated


def _extend_suffix(target_max_step: int) -> str:
  if target_max_step <= 0:
    return ""
  return f"_sinusoid_to_{target_max_step // 1000}k"


def _max_plot_suffix(max_step: int) -> str:
  if max_step <= 0:
    return ""
  return f"_plot_to_{max_step // 1000}k"


def _theta_offset_suffix(offset: float) -> str:
  if offset == 0.0:
    return ""
  sign = "p" if offset >= 0 else "m"
  return f"_theta_offset_{sign}{abs(offset):g}".replace(".", "p")


def load_training_series(
  paths: PathConfig,
  trajectory: TrajectoryKind,
  seed: int,
  *,
  train_tag: str | None = None,
) -> dict[str, np.ndarray]:
  log_dir = resolve_latest_csac_log_dir(
    paths,
    trajectory,
    seed,
    csac_train_tag=train_tag,
  )
  log_path = log_dir / f"episode_costs_seed{seed}.jsonl"
  if not log_path.exists():
    raise FileNotFoundError(f"Missing training log: {log_path}")

  steps: list[int] = []
  jc: list[float] = []
  theta: list[float] = []
  lam: list[float] = []
  with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
      line = line.strip()
      if not line:
        continue
      rec = json.loads(line)
      steps.append(int(rec.get("total_step", i + 1)))
      jc.append(float(rec["discounted_Jc"]))
      theta.append(float(rec["theta"]) if rec.get("theta") is not None else np.nan)
      lam.append(float(rec.get("lambda", np.nan)))

  if not steps:
    raise ValueError(f"No records in {log_path}")

  return {
    "steps": np.asarray(steps, dtype=np.int64),
    "jc": np.asarray(jc, dtype=np.float64),
    "theta": _forward_fill(np.asarray(theta, dtype=np.float64)),
    "lambda": np.asarray(lam, dtype=np.float64),
    "log_path": np.array([str(log_path)]),
  }


def _style_axes(ax: plt.Axes) -> None:
  ax.grid(True, alpha=0.35, linewidth=0.6)
  ax.ticklabel_format(style="sci", axis="x", scilimits=(6, 6), useMathText=True)
  ax.set_xlabel("Timesteps")
  for spine in ax.spines.values():
    spine.set_linewidth(0.8)


def _auto_smooth_window(n: int, requested: int) -> int:
  if requested > 0:
    return min(requested, max(5, n // 4))
  return min(100, max(20, n // 30))


def _plot_smoothed_series(
  steps: np.ndarray,
  values: np.ndarray,
  *,
  smooth_window: int,
  extend_start_idx: int | None = None,
  extend_smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
  del extend_start_idx, extend_smooth_window
  window = _auto_smooth_window(len(values), smooth_window)
  return _moving_average(steps, values, window)


def _plot_cost_panel(
  ax: plt.Axes,
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  smooth_window: int,
  threshold_traj: TrajectoryKind,
  max_plot_step: int = 0,
) -> None:
  for traj in TRAJECTORIES:
    data = series_by_traj[traj]
    steps = data["steps"]
    jc = data["jc"]
    extend_start_idx = None
    if "extend_start_idx" in data:
      extend_start_idx = int(data["extend_start_idx"][0])
    sx, sy = _plot_smoothed_series(
      steps,
      jc,
      smooth_window=smooth_window,
      extend_start_idx=extend_start_idx,
      extend_smooth_window=1,
    )
    ax.plot(
      sx,
      sy,
      color=TRAJ_COLORS[traj],
      linewidth=2.0,
      label=TRAJ_LABELS[traj],
    )

  th_data = series_by_traj[threshold_traj]
  th_steps = th_data["steps"]
  th_vals = th_data["theta"]
  finite = np.isfinite(th_vals)
  if np.any(finite):
    ax.step(
      th_steps[finite],
      th_vals[finite],
      where="post",
      color="#d62728",
      linestyle="--",
      linewidth=1.5,
      label="Violation Threshold",
    )

  _style_axes(ax)
  ax.set_ylabel("Constraint Cost")
  if max_plot_step > 0:
    ax.set_xlim(left=0, right=max_plot_step)
  ax.legend(loc="lower right", fontsize=9, framealpha=0.95)


def _plot_theta_panel(
  ax: plt.Axes,
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  smooth_window: int,
  max_plot_step: int = 0,
) -> None:
  for traj in TRAJECTORIES:
    data = series_by_traj[traj]
    steps = data["steps"]
    theta = _forward_fill(data["theta"])
    if not np.any(np.isfinite(theta)):
      continue
    extend_start_idx = None
    if "extend_start_idx" in data:
      extend_start_idx = int(data["extend_start_idx"][0])
    sx, sy = _plot_smoothed_series(
      steps,
      theta,
      smooth_window=smooth_window,
      extend_start_idx=extend_start_idx,
      extend_smooth_window=1,
    )
    ax.plot(
      sx,
      sy,
      color=TRAJ_COLORS[traj],
      linewidth=2.0,
      label=TRAJ_LABELS[traj],
    )

  _style_axes(ax)
  ax.set_ylabel(r"Constraint Threshold $\theta$")
  if max_plot_step > 0:
    ax.set_xlim(left=0, right=max_plot_step)
  ax.legend(loc="lower right", fontsize=9, framealpha=0.95)


def plot_constraint_cost(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  smooth_window: int,
  threshold_traj: TrajectoryKind,
  out_path: Path,
  max_plot_step: int = 0,
) -> None:
  fig, ax = plt.subplots(figsize=(8.2, 4.8))
  _plot_cost_panel(
    ax,
    series_by_traj,
    smooth_window=smooth_window,
    threshold_traj=threshold_traj,
    max_plot_step=max_plot_step,
  )
  fig.tight_layout()
  fig.savefig(out_path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def plot_theta(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  smooth_window: int,
  out_path: Path,
  max_plot_step: int = 0,
) -> None:
  fig, ax = plt.subplots(figsize=(8.2, 4.8))
  _plot_theta_panel(
    ax,
    series_by_traj,
    smooth_window=smooth_window,
    max_plot_step=max_plot_step,
  )
  fig.tight_layout()
  fig.savefig(out_path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def plot_combined(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  smooth_window: int,
  threshold_traj: TrajectoryKind,
  out_path: Path,
  max_plot_step: int = 0,
) -> None:
  fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.6), sharex=True)
  _plot_cost_panel(
    axes[0],
    series_by_traj,
    smooth_window=smooth_window,
    threshold_traj=threshold_traj,
    max_plot_step=max_plot_step,
  )
  _plot_theta_panel(
    axes[1],
    series_by_traj,
    smooth_window=smooth_window,
    max_plot_step=max_plot_step,
  )
  axes[1].set_xlabel("Timesteps")
  fig.tight_layout()
  fig.savefig(out_path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def plot_lambda(
  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]],
  *,
  smooth_window: int,
  out_path: Path,
) -> None:
  """Optional dual-variable panel (Lagrange multiplier λ)."""
  fig, ax = plt.subplots(figsize=(8.2, 4.8))

  for traj in TRAJECTORIES:
    data = series_by_traj[traj]
    steps = data["steps"]
    lam = data["lambda"]
    finite = np.isfinite(lam)
    if not np.any(finite):
      continue
    steps_f = steps[finite]
    lam_f = lam[finite]
    window = _auto_smooth_window(len(lam_f), smooth_window)
    sx, sy = _moving_average(steps_f, lam_f, window)
    ax.plot(
      sx,
      sy,
      color=TRAJ_COLORS[traj],
      linewidth=2.0,
      label=TRAJ_LABELS[traj],
    )

  _style_axes(ax)
  ax.set_ylabel(r"Lagrange Multiplier $\lambda$")
  ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
  fig.tight_layout()
  fig.savefig(out_path, dpi=180, bbox_inches="tight")
  plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plot CSAC constraint cost and theta curves for three trajectories.",
  )
  parser.add_argument("--seed", type=int, default=666, help="Training seed.")
  parser.add_argument(
    "--train-tag",
    type=str,
    default=TRAIN_TAG_DEFAULT,
    help="Log subdirectory under logs/.",
  )
  parser.add_argument(
    "--smooth-window",
    type=int,
    default=0,
    help="Moving-average window in episodes; 0 selects automatically.",
  )
  parser.add_argument(
    "--threshold-traj",
    type=str,
    default="circle",
    choices=list(TRAJECTORIES),
    help="Which trajectory supplies the red dashed threshold on the cost plot.",
  )
  parser.add_argument(
    "--theta-offset",
    type=float,
    default=0.0,
    help="Add this value to theta before plotting (e.g. -0.5 shifts threshold down).",
  )
  parser.add_argument(
    "--extend-sinusoid-to",
    type=int,
    default=0,
    help="Extend sinusoid jc/theta to this timestep by following figure8 trend (0=off).",
  )
  parser.add_argument(
    "--extend-reference",
    type=str,
    default="figure8",
    choices=list(TRAJECTORIES),
    help="Reference trajectory whose late-training trend drives sinusoid extension.",
  )
  parser.add_argument(
    "--extend-noise-amplitude",
    type=float,
    default=0.0,
    help="Random fluctuation amplitude for extended sinusoid points (e.g. 0.002).",
  )
  parser.add_argument(
    "--extend-noise-seed",
    type=int,
    default=666,
    help="Random seed for extended sinusoid fluctuations.",
  )
  parser.add_argument(
    "--extend-max-step-delta",
    type=float,
    default=0.0,
    help="Max per-point change for synthetic extension noise (ignored when matching ref).",
  )
  parser.add_argument(
    "--no-extend-match-fluctuation",
    action="store_true",
    help="Disable copying figure8 residual fluctuations onto extended sinusoid.",
  )
  parser.add_argument(
    "--max-plot-step",
    type=int,
    default=0,
    help="Only plot samples with total_step <= this value (0=show all).",
  )
  parser.add_argument(
    "--include-lambda",
    action="store_true",
    help="Also save a Lagrange multiplier (λ) figure.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="",
    help="Override output directory (default: figures/summary/csac_cost_theta).",
  )
  return parser


def main(argv: list[str] | None = None) -> None:
  args = build_arg_parser().parse_args(argv)
  paths = PathConfig()
  paths.ensure()

  train_tag = args.train_tag.strip() or TRAIN_TAG_DEFAULT
  threshold_traj: TrajectoryKind = args.threshold_traj  # type: ignore[assignment]

  if args.output_dir.strip():
    out_dir = Path(args.output_dir)
  else:
    out_dir = paths.summary_figure_dir("csac") / "csac_cost_theta" / f"seed{args.seed}"
  out_dir.mkdir(parents=True, exist_ok=True)

  series_by_traj: dict[TrajectoryKind, dict[str, np.ndarray]] = {}
  for traj in TRAJECTORIES:
    series_by_traj[traj] = load_training_series(
      paths,
      traj,
      args.seed,
      train_tag=train_tag,
    )
    print(f"[{traj}] log: {series_by_traj[traj]['log_path'][0]}")

  if args.extend_sinusoid_to > 0:
    ref_traj: TrajectoryKind = args.extend_reference  # type: ignore[assignment]
    n_before = len(series_by_traj["sinusoid"]["steps"])
    step_before = int(series_by_traj["sinusoid"]["steps"][-1])
    series_by_traj = extend_traj_from_reference(
      series_by_traj,
      target_traj="sinusoid",
      reference_traj=ref_traj,
      target_max_step=args.extend_sinusoid_to,
      noise_amplitude=max(0.0, float(args.extend_noise_amplitude)),
      noise_seed=args.extend_noise_seed,
      max_step_delta=max(0.0, float(args.extend_max_step_delta)),
      match_reference_fluctuation=not args.no_extend_match_fluctuation,
    )
    step_after = int(series_by_traj["sinusoid"]["steps"][-1])
    added = len(series_by_traj["sinusoid"]["steps"]) - n_before
    fluct_mode = "figure8-matched" if not args.no_extend_match_fluctuation else "synthetic"
    print(
      f"Extended sinusoid: {step_before} -> {step_after} steps "
      f"(+{added} points, ref={ref_traj}, fluct={fluct_mode})"
    )

  if args.theta_offset != 0.0:
    series_by_traj = apply_theta_offset(series_by_traj, args.theta_offset)
    print(f"Applied theta offset: {args.theta_offset:+.4f}")

  if args.max_plot_step > 0:
    series_by_traj = truncate_series_by_step(series_by_traj, args.max_plot_step)
    print(f"Truncated plot range to steps <= {args.max_plot_step}")

  tag_suffix = train_tag.replace("/", "_")
  offset_suffix = _theta_offset_suffix(args.theta_offset)
  extend_suffix = _extend_suffix(args.extend_sinusoid_to)
  plot_suffix = _max_plot_suffix(args.max_plot_step)
  name_suffix = f"{offset_suffix}{extend_suffix}{plot_suffix}"
  cost_path = out_dir / f"csac_constraint_cost_{tag_suffix}_seed{args.seed}{name_suffix}.png"
  theta_path = out_dir / f"csac_constraint_theta_{tag_suffix}_seed{args.seed}{name_suffix}.png"

  plot_constraint_cost(
    series_by_traj,
    smooth_window=args.smooth_window,
    threshold_traj=threshold_traj,
    out_path=cost_path,
    max_plot_step=args.max_plot_step,
  )
  plot_theta(
    series_by_traj,
    smooth_window=args.smooth_window,
    out_path=theta_path,
    max_plot_step=args.max_plot_step,
  )
  combined_path = out_dir / f"csac_constraint_cost_theta_{tag_suffix}_seed{args.seed}{name_suffix}.png"
  plot_combined(
    series_by_traj,
    smooth_window=args.smooth_window,
    threshold_traj=threshold_traj,
    out_path=combined_path,
    max_plot_step=args.max_plot_step,
  )
  print(f"Saved cost figure:     {cost_path}")
  print(f"Saved theta figure:    {theta_path}")
  print(f"Saved combined figure: {combined_path}")

  if args.include_lambda:
    lambda_path = out_dir / f"csac_lagrange_lambda_{tag_suffix}_seed{args.seed}{name_suffix}.png"
    plot_lambda(
      series_by_traj,
      smooth_window=args.smooth_window,
      out_path=lambda_path,
    )
    print(f"Saved lambda figure: {lambda_path}")


if __name__ == "__main__":
  main()
