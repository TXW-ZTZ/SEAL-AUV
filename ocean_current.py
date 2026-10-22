"""
洋流干扰模型（NED：x 北 / y 东 / z 下）。

物理依据：
  - 惯性系位置更新：ṗ = R(η)·ν + v_c(p, t)
  - 体坐标系相对流速：ν_rel = ν − R^T·v_c，用于附加流体力（简化二次阻尼）
  - 水平流速随深度剪切；叠加潮汐与低频脉动分量
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

CurrentIntensity = Literal[0, 1, 2, 3]
CURRENT_LABELS: dict[int, str] = {
  0: "none",
  1: "mild",
  2: "moderate",
  3: "strong",
}


@dataclass(frozen=True)
class CurrentSpec:
  """各强度档位的标称水平流速与扰动比例。"""
  speed: float
  gust_fraction: float
  tide_fraction: float
  shear_factor: float
  drag_scale: float


INTENSITY_SPECS: dict[int, CurrentSpec] = {
  0: CurrentSpec(0.0, 0.0, 0.0, 0.0, 0.0),
  1: CurrentSpec(0.05, 0.25, 0.15, 0.10, 0.6),
  2: CurrentSpec(0.12, 0.30, 0.20, 0.15, 1.2),
  3: CurrentSpec(0.25, 0.35, 0.25, 0.20, 2.0),
}


def parse_current_intensity(value: str | int) -> CurrentIntensity:
  if isinstance(value, int):
    if value in INTENSITY_SPECS:
      return value  # type: ignore[return-value]
    raise ValueError(f"current intensity must be 0–3, got {value}")
  key = str(value).strip().lower()
  if key.isdigit():
    return parse_current_intensity(int(key))
  name_to_id = {v: k for k, v in CURRENT_LABELS.items()}
  if key in name_to_id:
    return name_to_id[key]  # type: ignore[return-value]
  raise ValueError(
    f"Unknown current intensity '{value}'. Use 0–3 or none/mild/moderate/strong."
  )


def body_to_ned_rotation(phi: float, theta: float, psi: float) -> np.ndarray:
  cpsi, spsi = math.cos(psi), math.sin(psi)
  cth, sth = math.cos(theta), math.sin(theta)
  cphi, sphi = math.cos(phi), math.sin(phi)
  return np.array([
    [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
    [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
    [-sth, cth * sphi, cth * cphi],
  ], dtype=np.float64)


class OceanCurrentModel:
  """
  生成 NED 系洋流速度 v_c 及由相对流速引起的体坐标系广义力扰动。
  每个 episode 随机固定主流向，步内随时间与深度变化。
  """

  def __init__(
    self,
    intensity: CurrentIntensity = 0,
    tide_period: float = 200.0,
    rho_water: float = 1070.0,
    ref_length: float = 2.38,
  ):
    self.intensity = parse_current_intensity(intensity)
    self.spec = INTENSITY_SPECS[self.intensity]
    self.tide_period = tide_period
    self.rho_water = rho_water
    self.ref_length = ref_length
    self._heading_rad: float = 0.0
    self._gust_phase: float = 0.0

  @property
  def label(self) -> str:
    return CURRENT_LABELS[self.intensity]

  @property
  def active(self) -> bool:
    return self.intensity > 0

  def set_intensity(self, intensity: CurrentIntensity | str | int) -> None:
    self.intensity = parse_current_intensity(intensity)
    self.spec = INTENSITY_SPECS[self.intensity]

  def reset(self, rng: np.random.Generator) -> None:
    if not self.active:
      return
    self._heading_rad = float(rng.uniform(0.0, 2.0 * math.pi))
    self._gust_phase = float(rng.uniform(0.0, 2.0 * math.pi))

  def velocity_ned(self, t: float, position_ned: np.ndarray) -> np.ndarray:
    """返回 v_c = [u_n, u_e, u_d] (m/s)。"""
    if not self.active:
      return np.zeros(3, dtype=np.float64)

    spec = self.spec
    z_down = float(max(position_ned[2], 0.0))
    shear = 1.0 + spec.shear_factor * min(z_down / 25.0, 1.5)

    beta = self._heading_rad
    e_n = math.cos(beta)
    e_e = math.sin(beta)

    omega_t = 2.0 * math.pi / self.tide_period
    v_mean = spec.speed * shear
    v_tide = spec.speed * spec.tide_fraction * math.sin(omega_t * t + self._gust_phase)
    v_gust = spec.speed * spec.gust_fraction * math.sin(0.35 * t + 2.0 * self._gust_phase)
    v_horiz = v_mean + v_tide + v_gust

    v_d = 0.05 * spec.speed * math.sin(0.12 * t + self._gust_phase)
    return np.array([v_horiz * e_n, v_horiz * e_e, v_d], dtype=np.float64)

  def drag_wrench_body(
    self,
    x: np.ndarray,
    u: np.ndarray,
    v_c_ned: np.ndarray,
  ) -> np.ndarray:
    """
    由相对流速产生的简化流体力（体坐标 6 维广义力，主要作用于平动自由度）。
    F_i ≈ -0.5·ρ·A_ref·C_d·|ν_rel,i|·ν_rel,i
    """
    wrench = np.zeros((6, 1), dtype=np.float64)
    if not self.active:
      return wrench

    phi = float(x[3, 0])
    theta = float(x[4, 0])
    psi = float(x[5, 0])
    r_bn = body_to_ned_rotation(phi, theta, psi)
    v_c_body = r_bn.T @ np.asarray(v_c_ned, dtype=np.float64).reshape(3)
    v_body = np.array([u[0, 0], u[1, 0], u[2, 0]], dtype=np.float64)
    v_rel = v_body - v_c_body

    area_ref = self.ref_length ** 2
    scale = self.spec.drag_scale
    for i in range(3):
      vi = v_rel[i]
      wrench[i, 0] = -scale * 0.5 * self.rho_water * area_ref * vi * abs(vi)
    return wrench
