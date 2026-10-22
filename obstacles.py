"""沿参考轨迹布置球形障碍物（球心位于参考路径上，默认每回合固定）。"""

from dataclasses import dataclass
import math
import numpy as np

from trajectory import BaseTrajectory, make_trajectory
from config import TrajectoryKind


@dataclass
class ObstacleField:
  radius: float
  centers: np.ndarray
  arc_parameters: np.ndarray | None = None

  @property
  def num(self) -> int:
    return int(self.centers.shape[0])

  def distances_to_centers(self, position: np.ndarray) -> np.ndarray:
    p = np.asarray(position).reshape(1, 3)
    return np.linalg.norm(self.centers - p, axis=1)

  def nearest(self, position: np.ndarray) -> tuple[float, float, float, int]:
    dists = self.distances_to_centers(position)
    idx = int(np.argmin(dists))
    center_dist = float(dists[idx])
    c = self.centers[idx]
    p = np.asarray(position).reshape(3)
    dis_xy = math.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)
    psi = math.atan2(p[1] - c[1], p[0] - c[0])
    theta = math.atan2(p[2] - c[2], dis_xy + 1e-8)
    return center_dist, psi, theta, idx

  def surface_distance(self, position: np.ndarray) -> float:
    center_dist, _, _, _ = self.nearest(position)
    return center_dist - self.radius


def _arc_length_parameterization(
  traj: BaseTrajectory,
  t_horizon: float,
  n_samples: int = 1200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """离散参考轨迹，返回 (t 采样, 位置点, 累积弧长)。"""
  t_grid = np.linspace(0.0, t_horizon, n_samples)
  pts = np.stack([traj.position(float(ti)) for ti in t_grid], axis=0)
  seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
  cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
  return t_grid, pts, cum_s


def _time_at_arc_length(t_grid: np.ndarray, cum_s: np.ndarray, target_s: float) -> float:
  idx = int(np.searchsorted(cum_s, target_s, side="left"))
  idx = min(max(idx, 1), len(cum_s) - 1)
  s0, s1 = float(cum_s[idx - 1]), float(cum_s[idx])
  t0, t1 = float(t_grid[idx - 1]), float(t_grid[idx])
  if s1 <= s0 + 1e-12:
    return t0
  alpha = (target_s - s0) / (s1 - s0)
  return t0 + alpha * (t1 - t0)


def _offset_margin(position_offset_max: float, randomize_offset: bool) -> float:
  """Worst-case center shift toward an endpoint when applying per-axis offsets."""
  if not randomize_offset or position_offset_max <= 0.0:
    return 0.0
  return math.sqrt(3.0) * position_offset_max


def _center_endpoint_exclusion(
  endpoint_surface_clearance: float,
  ball_radius: float,
  position_offset_max: float,
  randomize_offset: bool,
) -> float:
  return endpoint_surface_clearance + ball_radius + _offset_margin(position_offset_max, randomize_offset)


def _sample_path_fractions(
  rng: np.random.Generator,
  obstacle_num: int,
  path_fraction_min: float,
  path_fraction_max: float,
  min_gap: float,
) -> np.ndarray:
  """在弧长比例区间内随机采样，保证相邻障碍最小间隔。"""
  span = path_fraction_max - path_fraction_min
  min_gap = min(min_gap, span / max(obstacle_num, 1) * 0.9)
  for _ in range(200):
    fracs = np.sort(rng.uniform(path_fraction_min, path_fraction_max, size=obstacle_num))
    if obstacle_num <= 1 or np.all(np.diff(fracs) >= min_gap):
      return fracs
  # 回退：均匀分布 + 小抖动
  base = np.linspace(path_fraction_min, path_fraction_max, obstacle_num)
  jitter = rng.uniform(-0.02, 0.02, size=obstacle_num)
  fracs = np.sort(np.clip(base + jitter, path_fraction_min, path_fraction_max))
  return fracs


def generate_obstacles_on_trajectory(
  traj: BaseTrajectory,
  obstacle_num: int,
  ball_radius: float,
  path_fraction_min: float,
  path_fraction_max: float,
  endpoint_exclusion_radius: float,
  t_horizon: float,
  start_exclusion_radius: float | None = None,
  end_exclusion_radius: float | None = None,
  rng: np.random.Generator | None = None,
  randomize_positions: bool = False,
  min_arc_fraction_gap: float = 0.08,
  arc_discretization: int = 1200,
  endpoint_surface_clearance: float | None = None,
  position_offset_max: float = 0.0,
  randomize_position_offset: bool = False,
) -> ObstacleField:
  """
  在参考轨迹上放置 obstacle_num 个球形障碍。
  基准球心为 traj.position(t_i)；randomize_position_offset=True 时可施加各轴随机偏移。
  randomize_positions=False 且 randomize_position_offset=False 时，每 episode 障碍位置完全相同。
  """
  if obstacle_num < 1:
    raise ValueError("obstacle_num must be >= 1")
  if not (0.0 <= path_fraction_min < path_fraction_max <= 1.0):
    raise ValueError("path_fraction_min/max must satisfy 0 <= min < max <= 1")
  if endpoint_exclusion_radius < 0.0:
    raise ValueError("endpoint_exclusion_radius must be >= 0")
  if position_offset_max < 0.0:
    raise ValueError("position_offset_max must be >= 0")
  surface_clearance = (
    float(endpoint_surface_clearance)
    if endpoint_surface_clearance is not None
    else float(endpoint_exclusion_radius)
  )
  if surface_clearance < 0.0:
    raise ValueError("endpoint_surface_clearance must be >= 0")
  center_excl = _center_endpoint_exclusion(
    surface_clearance,
    ball_radius,
    position_offset_max,
    randomize_position_offset,
  )
  start_excl = float(center_excl if start_exclusion_radius is None else start_exclusion_radius)
  end_excl = float(center_excl if end_exclusion_radius is None else end_exclusion_radius)
  if start_excl < center_excl - 1e-9 or end_excl < center_excl - 1e-9:
    start_excl = end_excl = center_excl
  if start_excl < 0.0 or end_excl < 0.0:
    raise ValueError("start/end exclusion radius must be >= 0")

  t_grid, pts, cum_s = _arc_length_parameterization(traj, t_horizon, arc_discretization)
  total_len = float(cum_s[-1])
  fracs = cum_s / max(total_len, 1e-12)
  start_pt = pts[0]
  end_pt = pts[-1]
  d_start = np.linalg.norm(pts - start_pt.reshape(1, 3), axis=1)
  d_end = np.linalg.norm(pts - end_pt.reshape(1, 3), axis=1)
  valid = (
    (fracs >= path_fraction_min)
    & (fracs <= path_fraction_max)
    & (d_start >= start_excl)
    & (d_end >= end_excl)
  )
  valid_idx = np.where(valid)[0]
  if valid_idx.size < obstacle_num:
    raise ValueError(
      "Not enough valid obstacle positions after endpoint exclusion. "
      "Please reduce obstacle_num or exclusion radius."
    )
  valid_s = cum_s[valid_idx]
  valid_t = t_grid[valid_idx]
  valid_fracs = valid_s / max(total_len, 1e-12)

  if randomize_positions:
    if rng is None:
      raise ValueError("randomize_positions requires rng")
    chosen = None
    for _ in range(300):
      pick = np.sort(rng.choice(valid_idx.size, size=obstacle_num, replace=False))
      if obstacle_num <= 1 or np.all(np.diff(valid_fracs[pick]) >= min_arc_fraction_gap):
        chosen = pick
        break
    if chosen is None:
      targets = _sample_path_fractions(
        rng, obstacle_num, float(valid_fracs[0]), float(valid_fracs[-1]), min_arc_fraction_gap
      )
      pick = np.searchsorted(valid_fracs, targets, side="left")
      pick = np.clip(pick, 0, valid_idx.size - 1)
      # 保证严格递增，避免重复索引
      for i in range(1, pick.size):
        if pick[i] <= pick[i - 1]:
          pick[i] = min(valid_idx.size - 1, pick[i - 1] + 1)
      chosen = pick
  else:
    chosen = np.floor(np.linspace(0, valid_idx.size - 1, obstacle_num)).astype(int)

  t_vals = valid_t[chosen]
  centers = np.stack([traj.position(float(t)) for t in t_vals], axis=0)
  if randomize_position_offset:
    if rng is None:
      raise ValueError("randomize_position_offset requires rng")
    if position_offset_max > 0.0:
      centers = centers + rng.uniform(-position_offset_max, position_offset_max, size=centers.shape)
  # 生成后强校验：障碍球面距起/终点 ≥ endpoint_surface_clearance。
  center_d_start = np.linalg.norm(centers - start_pt.reshape(1, 3), axis=1)
  center_d_end = np.linalg.norm(centers - end_pt.reshape(1, 3), axis=1)
  surface_d_start = center_d_start - ball_radius
  surface_d_end = center_d_end - ball_radius
  if np.any(surface_d_start < surface_clearance - 1e-9) or np.any(surface_d_end < surface_clearance - 1e-9):
    raise RuntimeError("Obstacle surfaces violate endpoint clearance constraints.")
  return ObstacleField(radius=ball_radius, centers=centers, arc_parameters=t_vals)


def generate_obstacles(
  trajectory: TrajectoryKind,
  obstacle_num: int,
  ball_radius: float,
  path_fraction_min: float,
  path_fraction_max: float,
  endpoint_exclusion_radius: float,
  t_horizon: float,
  start_exclusion_radius: float | None = None,
  end_exclusion_radius: float | None = None,
  los_delta: float = 7.14,
  rng: np.random.Generator | None = None,
  randomize_positions: bool = False,
  min_arc_fraction_gap: float = 0.08,
  endpoint_surface_clearance: float | None = None,
  position_offset_max: float = 0.0,
  randomize_position_offset: bool = False,
) -> ObstacleField:
  traj = make_trajectory(trajectory, los_delta=los_delta)
  return generate_obstacles_on_trajectory(
    traj,
    obstacle_num,
    ball_radius,
    path_fraction_min,
    path_fraction_max,
    endpoint_exclusion_radius,
    t_horizon,
    start_exclusion_radius=start_exclusion_radius,
    end_exclusion_radius=end_exclusion_radius,
    rng=rng,
    randomize_positions=randomize_positions,
    min_arc_fraction_gap=min_arc_fraction_gap,
    endpoint_surface_clearance=endpoint_surface_clearance,
    position_offset_max=position_offset_max,
    randomize_position_offset=randomize_position_offset,
  )
