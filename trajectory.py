"""参考轨迹与三维 LOS 期望姿态（Zhang et al. RT1/RT2/RT3）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import numpy as np

from config import TrajectoryKind
from utils import radlimit


@dataclass
class ReferenceState:
  position: np.ndarray
  velocity_hint: np.ndarray
  desired_yaw: float
  desired_pitch: float


class BaseTrajectory(ABC):
  def __init__(self, los_delta: float = 7.14):
    self.los_delta = los_delta

  @abstractmethod
  def position(self, t: float) -> np.ndarray:
    ...

  @abstractmethod
  def name(self) -> str:
    ...

  def velocity_hint(self, t: float, eps: float = 1e-3) -> np.ndarray:
    p0 = self.position(t)
    p1 = self.position(t + eps)
    return (p1 - p0) / eps

  def endpoint_distance(self, position: np.ndarray, t_horizon: float) -> float:
    """当前位置到轨迹终点 ref(T) 的欧氏距离。"""
    p = np.asarray(position).reshape(3)
    return float(np.linalg.norm(p - self.position(t_horizon)))

  def terminal_distance(self, position: np.ndarray, t_horizon: float | None = None) -> float:
    """兼容旧名；未给 t_horizon 时退化为到 ref(0)（不推荐用于成功判据）。"""
    if t_horizon is not None:
      return self.endpoint_distance(position, t_horizon)
    p = np.asarray(position).reshape(3)
    return float(np.linalg.norm(p - self.position(0.0)))

  def los_reference(self, t: float, position: np.ndarray | None = None) -> ReferenceState:
    eps = 0.1
    x_d, y_d, z_d = self.velocity_hint(t, eps)
    alpha_k = math.atan2(y_d, x_d + 1e-8)
    pos = self.position(t)
    if position is None:
      delta_x = delta_y = delta_z = 0.0
    else:
      p = np.asarray(position).reshape(3)
      delta_x = float(p[0] - pos[0])
      delta_y = float(p[1] - pos[1])
      delta_z = float(p[2] - pos[2])
    e = -delta_x * math.sin(alpha_k) + delta_y * math.cos(alpha_k)
    desired_yaw = radlimit(alpha_k + math.atan((-e) / self.los_delta))

    beta_k = math.atan(-z_d / (math.sqrt(x_d ** 2 + y_d ** 2) + 1e-8))
    r1 = np.array([
      [math.cos(alpha_k) * math.cos(beta_k), math.sin(alpha_k) * math.cos(beta_k), -math.sin(beta_k)],
      [-math.sin(alpha_k), math.cos(alpha_k), 0.0],
      [math.cos(alpha_k) * math.sin(beta_k), math.sin(alpha_k) * math.sin(beta_k), math.cos(beta_k)],
    ])
    # 俯仰 LOS 修正使用“位置误差在路径坐标系垂向轴上的投影”。
    err_vec = np.array([delta_x, delta_y, delta_z], dtype=np.float64)
    ze = float(np.dot(r1, err_vec)[2])
    desired_pitch = radlimit(beta_k + math.atan((-ze) / self.los_delta))
    return ReferenceState(
      position=pos,
      velocity_hint=np.array([x_d, y_d, z_d], dtype=np.float64),
      desired_yaw=desired_yaw,
      desired_pitch=desired_pitch,
    )


class CircleTrajectory(BaseTrajectory):
  """圆形螺旋 RT1；默认参数在 t∈[0,99.9]s 下弧长约 300 m。"""

  def __init__(self, radius: float = 23.84, z_slope: float = 0.2, z_offset: float = -20.0, los_delta: float = 7.14):
    super().__init__(los_delta)
    self.radius = radius
    self.z_slope = z_slope
    self.z_offset = z_offset
    self.omega = math.pi / 25.0

  def name(self) -> str:
    return "circle"

  def position(self, t: float) -> np.ndarray:
    return np.array([
      self.radius * math.sin(self.omega * t),
      self.radius * math.cos(self.omega * t),
      self.z_slope * t + self.z_offset,
    ], dtype=np.float64)


class SinusoidTrajectory(BaseTrajectory):
  """三维正弦轨迹 RT2（Zhang et al. 式 31）；默认参数弧长约 300 m，R_min ≥ 8 m。"""

  def __init__(
    self,
    x_speed: float = 2.0847,
    y_amplitude: float = 17.80,
    y_offset: float = 30.0,
    # 约束：t=0 时 z=0，t=99.9s 时 z=-40
    z_slope: float = -40.0 / 99.9,
    z_offset: float = 0.0,
    # 标定：T=99.9s 弧长≈300 m，最小转弯半径≈8.1 m（原 pi/25 约 6.1 m）
    omega: float = 0.17712,
    los_delta: float = 7.14,
  ):
    super().__init__(los_delta)
    self.x_speed = x_speed
    self.y_amplitude = y_amplitude
    self.y_offset = y_offset
    self.z_slope = z_slope
    self.z_offset = z_offset
    self.omega = omega

  def name(self) -> str:
    return "sinusoid"

  def position(self, t: float) -> np.ndarray:
    return np.array([
      self.x_speed * t,
      self.y_amplitude * math.sin(self.omega * t) + self.y_offset,
      self.z_slope * t + self.z_offset,
    ], dtype=np.float64)


class Figure8Trajectory(BaseTrajectory):
  """斜 Lissajous 八字（2:1）；XZ 为直线，T=99.9s 下 L≈300 m、R_min≥8 m。"""

  T_HORIZON: float = 99.9

  def __init__(
    self,
    x_amplitude: float = 45.0,
    y_amplitude: float = 56.15,
    # 约束：t=0 时 z=-20（x=0），t=T 时 z=-15；z = z_slope_x * x + z_offset
    z_offset: float = -20.0,
    z_end: float = -15.0,
    omega_x: float = 0.08582,
    omega_y: float = 0.04291,
    los_delta: float = 7.14,
  ):
    super().__init__(los_delta)
    self.x_amplitude = x_amplitude
    self.y_amplitude = y_amplitude
    self.z_offset = z_offset
    self.z_end = z_end
    self.omega_x = omega_x
    self.omega_y = omega_y
    x_T = x_amplitude * math.sin(omega_x * self.T_HORIZON)
    if abs(x_T) < 1.0:
      raise ValueError(
        f"Figure8 x(T)≈0 at T={self.T_HORIZON}s; cannot set straight XZ with z_end={z_end}"
      )
    self.z_slope_x = (z_end - z_offset) / x_T

  def name(self) -> str:
    return "figure8"

  def position(self, t: float) -> np.ndarray:
    x = self.x_amplitude * math.sin(self.omega_x * t)
    y = self.y_amplitude * math.cos(self.omega_y * t)
    z = self.z_slope_x * x + self.z_offset
    return np.array([x, y, z], dtype=np.float64)


def make_trajectory(kind: TrajectoryKind, los_delta: float = 7.14) -> BaseTrajectory:
  if kind == "circle":
    return CircleTrajectory(los_delta=los_delta)
  if kind == "sinusoid":
    return SinusoidTrajectory(los_delta=los_delta)
  if kind == "figure8":
    return Figure8Trajectory(los_delta=los_delta)
  raise ValueError(f"Unknown trajectory kind: {kind}")
