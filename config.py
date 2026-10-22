"""
统一配置：AUV 三维轨迹跟踪 + 避障 CMDP + Constrained SAC。

论文对应：
- Zhang et al. (2025) Ocean Engineering — 观测量、回报 shaping、RT1/RT2/RT3
- Tao et al. (2026) CACAIE — CMDP 能耗约束与拉格朗日 CSAC
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ThetaCurriculumMode = Literal["absolute", "relative"]

TrajectoryKind = Literal["circle", "sinusoid", "figure8"]
CurrentIntensity = Literal[0, 1, 2, 3]
AlgorithmKind = Literal["csac", "sac", "sacr"]

@dataclass
class SimConfig:
  dt: float = 0.1
  episode_steps: int = 1000
  detection_radius: float = 10.0
  cumulative_error_threshold: float = 4500.0
  max_path_error: float = 40.0
  collision_margin: float = 0.5
  safe_surface_distance: float = 3.0
  danger_surface_distance: float = 0.1
  snap_on_path_enabled: bool = False
  snap_path_threshold: float = 1.0
  terminal_success_dist: float = 5.0
  rmse_success_threshold: float = 3.0
  min_obstacle_surface_clearance: float = 0.1
  path_error_per_step_limit: float = 5.0
  path_error_step_fraction: float = 0.90
  start_position_radius: float = 1.0
  los_lookahead: float = 2.38 * 3.0
  seed: int = 7
  trajectory: TrajectoryKind = "circle"  # 训练默认：螺旋线 (circle)
  current_intensity: CurrentIntensity = 0


@dataclass
class ActuatorConfig:
  thrust_min: float = 0.0
  thrust_max: float = 500.0
  rudder_max_rad: float = 25.0 * 3.141592653589793 / 180.0
  # 一阶低通系数 α：a_exec = (1-α)*a_prev + α*a_raw，取值 (0,1]；1 表示关闭低通
  action_lowpass_alpha: float = 0.35
  thrust_rate_limit_per_s: float = 300.0
  rudder_rate_limit_rad_per_s: float = 0.4


@dataclass
class ObstacleConfig:
  obstacle_num: int = 3
  ball_radius: float = 1.5
  path_fraction_min: float = 0.12
  path_fraction_max: float = 0.88
  randomize_positions: bool = True
  min_arc_fraction_gap: float = 0.08
  arc_discretization: int = 1200
  endpoint_surface_clearance: float = 10.0
  start_exclusion_radius: float = 10.0
  end_exclusion_radius: float = 10.0
  endpoint_exclusion_radius: float = 10.0
  position_offset_max: float = 0.0
  randomize_position_offset: bool = False


@dataclass
class RewardConfig:
  """任务回报（式 14–17，Zhang et al.）+ 避障项；不含能耗。

  说明：环境中会先将各分量做统一归一化（主区间 [-1, 1]），再按以下权重加权。
  """
  w_path_log: float = 1.2
  w_heading: float = 1.0
  w_pitch: float = 1.5
  w_obstacle_log: float = 2.5
  path_log_clip: tuple[float, float] = (-2.0, 1.0)
  path_linear_threshold: float = 1.4
  w_path_linear: float = 0.1
  terminal_success_bonus: float = 100.0
  terminal_fail_penalty: float = 200.0
  horizon_completion_bonus: float = 50.0
  early_fail_scale_deviate: float = 0.30
  early_fail_scale_collision: float = 0.30
  timeout_fail_scale: float = 0.10


@dataclass
class CostConfig:
  """瞬时能耗代价 c_t（Tao et al. 式 35 在 AUV 作动上的代理）。"""
  beta_eng: float = 0.1
  w_thrust: float = 0.1
  w_rudder: float = 0.1
  w_delta_thrust: float = 0.1
  w_delta_rudder: float = 0.1


@dataclass
class CSACConfig:
  obs_dim: int = 14
  action_dim: int = 3
  hidden_dim: int = 256
  num_hidden_layers: int = 3
  gamma: float = 0.99
  tau: float = 0.005
  actor_lr: float = 1e-4
  critic_lr: float = 3e-4
  alpha_lr: float = 3e-4
  lambda_init: float = 0.05
  lambda_min: float = 0.0
  lambda_max: float = 100.0
  lambda_dual_step: float = 0.0005
  # λ 预热：前段 actor 不受能耗约束，对偶变量也不更新，先学跟踪
  lambda_warmup_start_frac: float = 0.0
  lambda_warmup_end_frac: float = 0.30
  adam_eps: float = 1e-4
  grad_clip_norm: float = 10.0
  target_entropy_scale: float = 1.0
  target_entropy_final_scale: float = 0.75
  target_entropy_decay_start_frac: float = 0.75
  target_entropy_decay_end_frac: float = 0.95
  alpha_freeze_start_frac: float = 1.0
  energy_budget_theta: float = 85.7
  # 能耗约束课程：θ 从宽松到最终目标，每 1/4 总 episode 允许降一档
  # relative：档位为满程 Jc 中位数的倍数（推荐，避免 SAC/CSAC 尺度失配）
  # absolute：固定能耗阈值（可由 calibrate_energy_theta.py 标定）
  theta_curriculum_enabled: bool = True
  theta_curriculum_mode: ThetaCurriculumMode = "relative"
  theta_curriculum_levels: tuple[float, ...] = (1.20, 1.10, 1.05, 1.00)
  theta_ref_min_samples: int = 50
  theta_satisfaction_rate_threshold: float = 0.90
  theta_curriculum_clean_full_rate_threshold: float = 0.80
  theta_curriculum_success_rate_threshold: float = 0.15
  theta_curriculum_replay_trim_frac: float = 0.50
  # SAC with rEnergy（SACr）：r_shaped = r_task − w_energy · c_t（固定权重，非 CMDP 约束）
  energy_reward_penalty: float = 0.1
  batch_size: int = 256
  buffer_size: int = 100_000
  initial_random_steps: int = 10_000
  update_after: int = 10_000
  update_every: int = 4
  utd_ratio: int = 1
  use_amp: bool = True
  train_episodes: int = 2200
  train_frames: int = 2_200_000
  num_seeds: int = 6
  log_interval: int = 200_000
  eval_interval_episodes: int = 100
  eval_seeds: tuple[int, ...] = (101, 202, 303, 404, 505)
  curriculum_window_episodes: int = 100
  curriculum_clean_full_rate_threshold: float = 0.80
  curriculum_current_levels: tuple[int, ...] = (0, 1, 2)
  cost_log_flush_episodes: int = 10
  train_log_print_episodes: int = 10
  device: str = "cuda"
  training_profile: str = "default"


def apply_fast_training_profile(cfg: CSACConfig) -> CSACConfig:
  """快速训练配置：更小网络 + 更少更新，适合调试与冒烟测试。"""
  cfg.training_profile = "fast"
  cfg.hidden_dim = 128
  cfg.num_hidden_layers = 2
  cfg.batch_size = 128
  cfg.update_every = 4
  cfg.utd_ratio = 1
  cfg.use_amp = True
  cfg.initial_random_steps = 5_000
  cfg.update_after = 5_000
  return cfg


def apply_weak_baseline_training_profile(cfg: CSACConfig) -> CSACConfig:
  """Mildly weaker profile for SAC/SACr baseline sensitivity runs."""
  cfg.training_profile = "weak_baseline"
  cfg.actor_lr = 8e-5
  cfg.critic_lr = 2.5e-4
  cfg.alpha_lr = 2.5e-4
  cfg.initial_random_steps = 12_000
  cfg.update_after = 12_000
  cfg.update_every = 5
  cfg.target_entropy_final_scale = 0.90
  cfg.target_entropy_decay_start_frac = 0.85
  cfg.target_entropy_decay_end_frac = 0.98
  cfg.energy_reward_penalty = 0.12
  return cfg


@dataclass
class PathConfig:
  root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
  checkpoint_dir: Path = field(init=False)
  log_dir: Path = field(init=False)
  figure_dir: Path = field(init=False)

  def __post_init__(self):
    self.checkpoint_dir = self.root / "checkpoints"
    self.log_dir = self.root / "logs"
    self.figure_dir = self.root / "figures"

  def ensure(self):
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.figure_dir.mkdir(parents=True, exist_ok=True)

  def traj_checkpoint_dir(self, trajectory: TrajectoryKind) -> Path:
    d = self.checkpoint_dir / trajectory
    d.mkdir(parents=True, exist_ok=True)
    return d

  def traj_figure_dir(self, trajectory: TrajectoryKind) -> Path:
    d = self.figure_dir / trajectory
    d.mkdir(parents=True, exist_ok=True)
    return d

  def algorithm_log_root(self, algorithm: AlgorithmKind = "csac") -> Path:
    if algorithm == "sac":
      return self.log_dir / "sac"
    if algorithm == "sacr":
      return self.log_dir / "sacr"
    return self.log_dir

  def algorithm_figure_root(self, algorithm: AlgorithmKind = "csac") -> Path:
    if algorithm == "sac":
      return self.figure_dir / "sac"
    if algorithm == "sacr":
      return self.figure_dir / "sacr"
    return self.figure_dir

  def traj_eval_log_dir(
    self,
    trajectory: TrajectoryKind,
    algorithm: AlgorithmKind = "csac",
    current_label: str | None = None,
  ) -> Path:
    d = self.algorithm_log_root(algorithm) / trajectory
    if current_label:
      d = d / current_label
    d.mkdir(parents=True, exist_ok=True)
    return d

  def traj_eval_figure_dir(self, trajectory: TrajectoryKind, algorithm: AlgorithmKind = "csac") -> Path:
    d = self.algorithm_figure_root(algorithm) / trajectory
    d.mkdir(parents=True, exist_ok=True)
    return d

  def summary_figure_dir(self, algorithm: AlgorithmKind = "csac") -> Path:
    d = self.algorithm_figure_root(algorithm) / "summary"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def eval_summary_all_path(self, seed: int, algorithm: AlgorithmKind = "csac") -> Path:
    return self.algorithm_log_root(algorithm) / f"eval_summary_all_seed{seed}.json"

  def cross_eval_log_dir(self, train_trajectory: TrajectoryKind, seed: int) -> Path:
    d = self.log_dir / "cross_eval" / train_trajectory / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d

  def cross_eval_figure_dir(self, train_trajectory: TrajectoryKind, seed: int) -> Path:
    d = self.figure_dir / "cross_eval" / train_trajectory / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d
