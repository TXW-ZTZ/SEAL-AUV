"""
AUV 轨迹跟踪 + 避障 CMDP 环境。

CMDP (Tao et al. 式 15–18):
  max_π E[Σ γ^t r_t]   s.t.   E[Σ γ^t c_t] ≤ θ

观测量 14 维（Zhang et al. + 最近障碍方位），回报不含能耗。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from config import (
  SimConfig,
  ActuatorConfig,
  ObstacleConfig,
  RewardConfig,
  CostConfig,
  TrajectoryKind,
)
from trajectory import make_trajectory, BaseTrajectory
from obstacles import generate_obstacles_on_trajectory, ObstacleField
from dynamics import AUVSimulator
from ocean_current import OceanCurrentModel
from utils import radlimit, denormalize_action, normalize_action


@dataclass
class StepInfo:
  d_path: float
  energy_cost: float
  energy_components: dict | None
  closest_center_dist: float
  surface_dist: float
  success: bool
  reason: str
  collision: bool = False
  collision_state: str = "安全"
  collision_warning: bool = False
  collision_warning_count: int = 0
  d_endpoint: float = float("nan")
  rmse_3d: float = float("nan")
  min_obstacle_surface: float = float("nan")
  action_exec_norm: np.ndarray | None = None


@dataclass
class EpisodeLog:
  """单 episode 记录，供评估与绘图。"""
  positions: list = field(default_factory=list)
  ref_positions: list = field(default_factory=list)
  errors_xyz: list = field(default_factory=list)
  actions_phys: list = field(default_factory=list)
  step_costs: list = field(default_factory=list)
  step_energy_components: list = field(default_factory=list)
  heading_errors: list = field(default_factory=list)
  pitch_errors: list = field(default_factory=list)
  times: list = field(default_factory=list)


@dataclass
class EpisodeInitState:
  """Episode 初始条件快照，用于多算法公平对比（同一 eval seed 下初始状态一致）。"""
  x: np.ndarray
  u: np.ndarray
  start: np.ndarray
  obstacles: ObstacleField
  random_factor_array: np.ndarray
  prev_action_phys: np.ndarray
  rng_state: dict

class AUVTrackingCMDPEnv:
  ENERGY_COMPONENT_KEYS = (
    "thrust_energy",
    "rudder_energy",
    "delta_thrust_energy",
    "delta_rudder_energy",
    "propulsion_energy",
    "steering_energy",
    "total_energy",
  )

  def __init__(
    self,
    trajectory: TrajectoryKind | None = None,
    sim_cfg: SimConfig | None = None,
    act_cfg: ActuatorConfig | None = None,
    obs_cfg: ObstacleConfig | None = None,
    rew_cfg: RewardConfig | None = None,
    cost_cfg: CostConfig | None = None,
    randomize_start: bool = True,
    fuzzy_dynamics: bool = False,
    record_episode_log: bool = True,
  ):
    self.sim = sim_cfg or SimConfig()
    if trajectory is not None:
      self.sim.trajectory = trajectory
    self.act = act_cfg or ActuatorConfig()
    self.obs_cfg = obs_cfg or ObstacleConfig()
    self.rew_cfg = rew_cfg or RewardConfig()
    self.cost_cfg = cost_cfg or CostConfig()
    self.randomize_start = randomize_start
    self.fuzzy_dynamics = fuzzy_dynamics
    self.record_episode_log = record_episode_log

    self.traj: BaseTrajectory = make_trajectory(self.sim.trajectory, self.sim.los_lookahead)
    self.plant = AUVSimulator(dt=self.sim.dt)
    self.ocean_current = OceanCurrentModel(intensity=self.sim.current_intensity)
    self.rng = np.random.default_rng(self.sim.seed)

    self.obs_dim = 14
    self.action_dim = 3
    self.episode_log = EpisodeLog()
    self._init_counters()
    self.reset()

  def _init_counters(self):
    self.episode_counter = 0
    self.total_steps = 0
    self.collision_count = 0
    self.collision_warning_count = 0
    self.success_count = 0

  def reset(self, seed: int | None = None) -> np.ndarray:
    if seed is not None:
      self.rng = np.random.default_rng(seed)

    self.t = 0.0
    self.t_horizon = self.sim.dt * (self.sim.episode_steps - 1)
    self.step_counter = 0
    self.cumulative_error = 0.0
    self.episode_reward = 0.0
    self.episode_cost = 0.0
    self.episode_energy_components = self._zero_energy_components()
    self.success = False
    self.min_obstacle_surface = float("inf")
    self.episode_collision_warning_count = 0
    self.episode_log = EpisodeLog()
    self.prev_action_phys = np.array(
      [(self.act.thrust_min + self.act.thrust_max) * 0.5, 0.0, 0.0],
      dtype=np.float64,
    )

    ref0 = self.traj.position(0.0)
    if self.randomize_start:
      start_pos = self._sample_start_near_reference(ref0)
    else:
      start_pos = ref0.copy()
    self.x = np.zeros((6, 1), dtype=np.float64)
    self.x[:3, 0] = start_pos
    self.u = np.zeros((6, 1), dtype=np.float64)

    if self.randomize_start:
      self._randomize_state()

    self.ocean_current.reset(self.rng)

    if self.fuzzy_dynamics:
      self.random_factor_array = self.rng.uniform(0.8, 1.2, 25)
    else:
      self.random_factor_array = np.ones(25, dtype=np.float64)

    t_horizon = self.sim.dt * (self.sim.episode_steps - 1)
    self.obstacles: ObstacleField = generate_obstacles_on_trajectory(
      self.traj,
      self.obs_cfg.obstacle_num,
      self.obs_cfg.ball_radius,
      self.obs_cfg.path_fraction_min,
      self.obs_cfg.path_fraction_max,
      endpoint_exclusion_radius=self.obs_cfg.endpoint_exclusion_radius,
      t_horizon=t_horizon,
      start_exclusion_radius=self.obs_cfg.start_exclusion_radius,
      end_exclusion_radius=self.obs_cfg.end_exclusion_radius,
      rng=self.rng,
      randomize_positions=self.obs_cfg.randomize_positions,
      min_arc_fraction_gap=self.obs_cfg.min_arc_fraction_gap,
      arc_discretization=self.obs_cfg.arc_discretization,
      endpoint_surface_clearance=self.obs_cfg.endpoint_surface_clearance,
      position_offset_max=self.obs_cfg.position_offset_max,
      randomize_position_offset=self.obs_cfg.randomize_position_offset,
    )
    self.start = self.x[:3].copy()
    self._reset_episode_metrics()
    return self._get_obs()

  def capture_init_state(self) -> EpisodeInitState:
    arc = self.obstacles.arc_parameters
    return EpisodeInitState(
      x=self.x.copy(),
      u=self.u.copy(),
      start=self.start.copy(),
      obstacles=ObstacleField(
        radius=float(self.obstacles.radius),
        centers=self.obstacles.centers.copy(),
        arc_parameters=None if arc is None else arc.copy(),
      ),
      random_factor_array=self.random_factor_array.copy(),
      prev_action_phys=self.prev_action_phys.copy(),
      rng_state=self.rng.bit_generator.state,
    )

  def restore_init_state(self, state: EpisodeInitState) -> np.ndarray:
    self.t = 0.0
    self.t_horizon = self.sim.dt * (self.sim.episode_steps - 1)
    self.step_counter = 0
    self.cumulative_error = 0.0
    self.episode_reward = 0.0
    self.episode_cost = 0.0
    self.episode_energy_components = self._zero_energy_components()
    self.success = False
    self.min_obstacle_surface = float("inf")
    self.episode_collision_warning_count = 0
    self.episode_log = EpisodeLog()
    self.x = state.x.copy()
    self.u = state.u.copy()
    self.start = state.start.copy()
    self.obstacles = ObstacleField(
      radius=float(state.obstacles.radius),
      centers=state.obstacles.centers.copy(),
      arc_parameters=(
        None
        if state.obstacles.arc_parameters is None
        else state.obstacles.arc_parameters.copy()
      ),
    )
    self.random_factor_array = state.random_factor_array.copy()
    self.prev_action_phys = state.prev_action_phys.copy()
    self.rng.bit_generator.state = state.rng_state
    self.ocean_current.reset(self.rng)
    self._reset_episode_metrics()
    return self._get_obs()

  def _reset_episode_metrics(self) -> None:
    """轻量 episode 统计（record_episode_log=False 时用于距离/RMSE/成功判据）。"""
    self._prev_position = np.asarray(self.start, dtype=np.float64).reshape(3)
    self.episode_distance_m = 0.0
    self.distance_before_collision_m: float | None = None
    self._rmse_sq_sum = 0.0
    self._rmse_step_count = 0
    self._path_error_steps_ok = 0

  def _sample_start_near_reference(self, ref0: np.ndarray) -> np.ndarray:
    """在参考轨迹起点 ref0 的球形邻域内均匀采样（半径 start_position_radius）。"""
    r_max = self.sim.start_position_radius
    direction = self.rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
      direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
      direction /= norm
    radius = r_max * (self.rng.random() ** (1.0 / 3.0))
    return ref0 + radius * direction

  def _randomize_state(self):
    self.x[4, 0] = 0.0
    self.x[5, 0] = float(self.rng.uniform(-np.pi, np.pi))
    sign = self.rng.integers(0, 2)
    self.u[0, 0] = 1.0 + self.rng.random() * 0.5 * (1 if sign else -1)
    for idx in (1, 5):
      sign = self.rng.integers(0, 2)
      self.u[idx, 0] += self.rng.random() * 0.1 * (1 if sign else -1)

  def _reference(self):
    ref = self.traj.los_reference(self.t, self.x[:3, 0])
    self.x_desire, self.y_desire, self.z_desire = ref.position
    self.desired_yaw = ref.desired_yaw
    self.desired_pitch = ref.desired_pitch
    return ref

  def _get_obs(self) -> np.ndarray:
    ref = self._reference()
    delta = self.x[:3, 0] - ref.position
    d_path = float(np.linalg.norm(delta))
    self.d_path = d_path
    self.d_endpoint = self.traj.endpoint_distance(self.x[:3, 0], self.t_horizon)

    center_dist, psi_obs, theta_obs, _ = self.obstacles.nearest(self.x[:3, 0])
    self.closest_dis_to_obstacle = center_dist
    surface = max(center_dist - self.obstacles.radius, 0.0)

    obs = np.zeros(14, dtype=np.float32)
    obs[0:3] = delta
    obs[3] = d_path
    obs[4:9] = self.u[[0, 1, 2, 4, 5], 0]
    obs[9] = radlimit(self.desired_pitch - self.x[4, 0])
    obs[10] = radlimit(self.desired_yaw - self.x[5, 0])
    obs[11] = np.clip(surface / self.sim.detection_radius, 0.0, 1.0)
    obs[12] = psi_obs
    obs[13] = theta_obs
    return obs

  def _task_reward(
    self,
    d_path: float,
    heading_error: float,
    pitch_error: float,
    surface_dist: float,
  ) -> float:
    lo, hi = self.rew_cfg.path_log_clip

    # 1) 路径 log 项：由原始 shaping 统一映射到 [-1, 1]
    # raw = -clip(log2(d), lo, hi)，其理论范围为 [-hi, -lo]
    path_log_raw = -np.clip(np.log2(d_path + 1e-20), lo, hi)
    path_log_min, path_log_max = -hi, -lo
    path_log_norm = self._normalize_symmetric(path_log_raw, path_log_min, path_log_max)

    # 2) 路径线性补充项：统一为 [-1, 0] 单侧惩罚（0 最好，-1 最差）
    linear_span = max(self.sim.max_path_error - self.rew_cfg.path_linear_threshold, 1e-6)
    path_linear_excess = max(d_path - self.rew_cfg.path_linear_threshold, 0.0)
    path_linear_norm = -np.clip(path_linear_excess / linear_span, 0.0, 1.0)

    # 3) 航向与俯仰项：cos 本身已在 [-1, 1]
    heading_norm = float(np.cos(heading_error))
    pitch_norm = float(np.cos(pitch_error))

    # 4) 避障项：安全距离内按 log 比例惩罚，归一化到 [-1, 0]
    obstacle_norm = 0.0
    safe_dist = max(self.sim.safe_surface_distance, 1e-6)
    if surface_dist < safe_dist:
      surface_ratio = np.clip(surface_dist / safe_dist, 1e-2, 1.0)
      obstacle_norm = float(np.clip(np.log10(surface_ratio), -2.0, 0.0) / 2.0)

    r = 0.02
    r += self.rew_cfg.w_path_log * path_log_norm
    r += self.rew_cfg.w_path_linear * path_linear_norm
    r += self.rew_cfg.w_heading * heading_norm
    r += self.rew_cfg.w_pitch * pitch_norm
    r += self.rew_cfg.w_obstacle_log * obstacle_norm
    return float(r)

  @staticmethod
  def _normalize_symmetric(x: float, x_min: float, x_max: float) -> float:
    """将 [x_min, x_max] 线性映射到 [-1, 1]。"""
    if x_max <= x_min:
      return 0.0
    x_clip = float(np.clip(x, x_min, x_max))
    return 2.0 * (x_clip - x_min) / (x_max - x_min) - 1.0

  def _collision_state(self, surface_dist: float) -> tuple[str, bool, bool]:
    if surface_dist <= self.sim.danger_surface_distance:
      return "危险", False, True
    if surface_dist < self.sim.safe_surface_distance:
      return "预警", True, False
    return "安全", False, False

  def _zero_energy_components(self) -> dict[str, float]:
    return {key: 0.0 for key in self.ENERGY_COMPONENT_KEYS}

  def _energy_cost_components(self, action_phys: np.ndarray) -> dict[str, float]:
    f_n = action_phys[0] / self.act.thrust_max
    d0_n = action_phys[1] / self.act.rudder_max_rad
    d2_n = action_phys[2] / self.act.rudder_max_rad
    beta = self.cost_cfg.beta_eng
    thrust_energy = beta * self.cost_cfg.w_thrust * f_n ** 2
    rudder_energy = beta * self.cost_cfg.w_rudder * (d0_n ** 2 + d2_n ** 2)
    # 叠加动作变化率平滑项，抑制高频抖动。
    thrust_span = max(self.act.thrust_max - self.act.thrust_min, 1e-9)
    d_f = (action_phys[0] - self.prev_action_phys[0]) / thrust_span
    rudder_span = max(2.0 * self.act.rudder_max_rad, 1e-9)
    d_r0 = (action_phys[1] - self.prev_action_phys[1]) / rudder_span
    d_r2 = (action_phys[2] - self.prev_action_phys[2]) / rudder_span
    delta_thrust_energy = beta * self.cost_cfg.w_delta_thrust * d_f ** 2
    delta_rudder_energy = beta * self.cost_cfg.w_delta_rudder * (d_r0 ** 2 + d_r2 ** 2)
    propulsion_energy = thrust_energy + delta_thrust_energy
    steering_energy = rudder_energy + delta_rudder_energy
    total_energy = propulsion_energy + steering_energy
    return {
      "thrust_energy": float(thrust_energy),
      "rudder_energy": float(rudder_energy),
      "delta_thrust_energy": float(delta_thrust_energy),
      "delta_rudder_energy": float(delta_rudder_energy),
      "propulsion_energy": float(propulsion_energy),
      "steering_energy": float(steering_energy),
      "total_energy": float(total_energy),
    }

  def _energy_cost(self, action_phys: np.ndarray) -> float:
    return self._energy_cost_components(action_phys)["total_energy"]

  def _smoothed_and_limited_action(self, action_phys_raw: np.ndarray) -> np.ndarray:
    alpha = float(np.clip(self.act.action_lowpass_alpha, 1e-6, 1.0))
    action_cmd = (1.0 - alpha) * self.prev_action_phys + alpha * action_phys_raw

    thrust_rate = float(self.act.thrust_rate_limit_per_s)
    if thrust_rate > 0.0:
      max_df = thrust_rate * self.sim.dt
      action_cmd[0] = np.clip(
        action_cmd[0],
        self.prev_action_phys[0] - max_df,
        self.prev_action_phys[0] + max_df,
      )

    rudder_rate = float(self.act.rudder_rate_limit_rad_per_s)
    if rudder_rate > 0.0:
      max_dr = rudder_rate * self.sim.dt
      action_cmd[1] = np.clip(
        action_cmd[1],
        self.prev_action_phys[1] - max_dr,
        self.prev_action_phys[1] + max_dr,
      )
      action_cmd[2] = np.clip(
        action_cmd[2],
        self.prev_action_phys[2] - max_dr,
        self.prev_action_phys[2] + max_dr,
      )

    action_cmd[0] = np.clip(action_cmd[0], self.act.thrust_min, self.act.thrust_max)
    action_cmd[1] = np.clip(action_cmd[1], -self.act.rudder_max_rad, self.act.rudder_max_rad)
    action_cmd[2] = np.clip(action_cmd[2], -self.act.rudder_max_rad, self.act.rudder_max_rad)
    return action_cmd.astype(np.float64, copy=False)

  def step(self, action_norm: np.ndarray) -> tuple[np.ndarray, float, float, bool, StepInfo]:
    action_phys_raw = denormalize_action(
      action_norm,
      self.act.thrust_min,
      self.act.thrust_max,
      self.act.rudder_max_rad,
    )
    action_phys = self._smoothed_and_limited_action(action_phys_raw)
    ref = self._reference()
    d_path_before = float(np.linalg.norm(self.x[:3, 0] - ref.position))
    snap = (
      self.sim.snap_on_path_enabled
      and d_path_before <= self.sim.snap_path_threshold
      and self.obstacles.surface_distance(self.x[:3, 0]) >= self.sim.collision_margin
    )

    self.x, self.u, _ = self.plant.step(
      self.x,
      self.u,
      action_phys,
      self.t,
      self.random_factor_array,
      snap_on_path=snap,
      desired_pose=(self.desired_yaw, self.desired_pitch, ref.position),
      ocean_current=self.ocean_current,
    )

    self.t += self.sim.dt
    self.step_counter += 1
    self.total_steps += 1

    obs = self._get_obs()
    heading_error = radlimit(self.desired_yaw - self.x[5, 0])
    pitch_error = radlimit(self.desired_pitch - self.x[4, 0])
    surface_dist = self.obstacles.surface_distance(self.x[:3, 0])

    reward = self._task_reward(self.d_path, heading_error, pitch_error, surface_dist)
    energy_components = self._energy_cost_components(action_phys)
    cost = energy_components["total_energy"]

    if self.record_episode_log:
      self.episode_log.positions.append(self.x[:3, 0].copy())
      self.episode_log.ref_positions.append(ref.position.copy())
      self.episode_log.errors_xyz.append(self.x[:3, 0] - ref.position)
      self.episode_log.actions_phys.append(action_phys.copy())
      self.episode_log.step_costs.append(cost)
      self.episode_log.step_energy_components.append(dict(energy_components))
      self.episode_log.heading_errors.append(heading_error)
      self.episode_log.pitch_errors.append(pitch_error)
      self.episode_log.times.append(self.t)
    else:
      pos = self.x[:3, 0].copy()
      self.episode_distance_m += float(np.linalg.norm(pos - self._prev_position))
      self._prev_position = pos
      err = pos - np.asarray(ref.position, dtype=np.float64).reshape(3)
      d_path_step = float(np.linalg.norm(err))
      self._rmse_sq_sum += d_path_step ** 2
      self._rmse_step_count += 1
      if d_path_step <= self.sim.path_error_per_step_limit:
        self._path_error_steps_ok += 1

    self.cumulative_error += self.d_path
    self.min_obstacle_surface = min(self.min_obstacle_surface, surface_dist)
    self.episode_reward += reward
    self.episode_cost += cost
    for key, value in energy_components.items():
      self.episode_energy_components[key] += value

    collision_state, collision_warning, dangerous_collision = self._collision_state(surface_dist)
    if collision_warning:
      self.collision_warning_count += 1
      self.episode_collision_warning_count += 1

    done, reason, terminal_bonus, collision = self._terminal_check(dangerous_collision)
    if collision_warning and not done:
      reason = "碰撞预警"
    reward += terminal_bonus
    self.episode_reward += terminal_bonus

    if done:
      self.episode_counter += 1
      if self.success:
        self.success_count += 1
      if collision:
        self.collision_count += 1

    ep_rmse = self._episode_rmse() if done else float("nan")
    info = StepInfo(
      d_path=float(self.d_path),
      energy_cost=cost,
      energy_components=dict(energy_components),
      closest_center_dist=float(self.closest_dis_to_obstacle),
      surface_dist=float(surface_dist),
      success=self.success,
      reason=reason,
      collision=collision,
      collision_state=collision_state,
      collision_warning=collision_warning,
      collision_warning_count=int(self.episode_collision_warning_count),
      d_endpoint=float(self.d_endpoint),
      rmse_3d=ep_rmse,
      min_obstacle_surface=float(self.min_obstacle_surface),
    )
    self.prev_action_phys = action_phys
    info.action_exec_norm = normalize_action(
      action_phys,
      self.act.thrust_min,
      self.act.thrust_max,
      self.act.rudder_max_rad,
    )
    return obs, reward, cost, done, info

  def _episode_rmse(self) -> float:
    if not self.record_episode_log:
      if self._rmse_step_count <= 0:
        return float("inf")
      return float(np.sqrt(self._rmse_sq_sum / self._rmse_step_count))
    errs = self.episode_log.errors_xyz
    if not errs:
      return float("inf")
    arr = np.asarray(errs, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(arr ** 2, axis=1))))

  def _path_error_fraction_ok(self) -> bool:
    if not self.record_episode_log:
      if self._rmse_step_count <= 0:
        return False
      frac = float(self._path_error_steps_ok / self._rmse_step_count)
      return frac >= self.sim.path_error_step_fraction
    errs = self.episode_log.errors_xyz
    if not errs:
      return False
    d_paths = np.linalg.norm(np.asarray(errs, dtype=np.float64), axis=1)
    frac = float(np.mean(d_paths <= self.sim.path_error_per_step_limit))
    return frac >= self.sim.path_error_step_fraction

  def check_success(self) -> bool:
    """权威 episode 成功判据（训练/评估仅应使用 info.success / 本方法）。"""
    if self.step_counter < self.sim.episode_steps:
      return False
    if self.min_obstacle_surface < self.sim.min_obstacle_surface_clearance - 1e-9:
      return False
    if self._episode_rmse() > self.sim.rmse_success_threshold:
      return False
    if not self._path_error_fraction_ok():
      return False
    if self.d_endpoint >= self.sim.terminal_success_dist:
      return False
    if self.cumulative_error > self.sim.cumulative_error_threshold:
      return False
    return True

  def _terminal_check(self, dangerous_collision: bool) -> tuple[bool, str, float, bool]:
    bonus = 0.0
    reason = ""
    done = False
    collision = False

    end_timeout = self.step_counter >= self.sim.episode_steps
    end_deviate = self.d_path >= self.sim.max_path_error
    end_collision = dangerous_collision
    remaining_frac = float(
      np.clip(
        (self.sim.episode_steps - self.step_counter) / max(self.sim.episode_steps, 1),
        0.0,
        1.0,
      )
    )

    if end_deviate:
      done = True
      reason = "deviate"
      bonus -= self.rew_cfg.terminal_fail_penalty
      bonus -= (
        self.rew_cfg.terminal_fail_penalty
        * self.rew_cfg.early_fail_scale_deviate
        * remaining_frac
      )
      self.success = False
    if end_collision:
      done = True
      reason = "危险碰撞"
      bonus -= self.rew_cfg.terminal_fail_penalty
      bonus -= (
        self.rew_cfg.terminal_fail_penalty
        * self.rew_cfg.early_fail_scale_collision
        * remaining_frac
      )
      self.success = False
      collision = True
      if not self.record_episode_log:
        self.distance_before_collision_m = float(self.episode_distance_m)
    if end_timeout and not done:
      done = True
      if self.check_success():
        reason = "success"
        bonus += self.rew_cfg.terminal_success_bonus
        self.success = True
      else:
        reason = "timeout"
        bonus -= self.rew_cfg.terminal_fail_penalty * self.rew_cfg.timeout_fail_scale
        bonus += self.rew_cfg.horizon_completion_bonus
        self.success = False

    return done, reason, bonus, collision
