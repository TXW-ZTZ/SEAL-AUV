"""AUV 5-DOF 动力学封装（RK4），底层 AUV_model_fuzzy.py；可选洋流干扰。"""

import numpy as np
import AUV_model_fuzzy as model
from utils import radlimit
from ocean_current import OceanCurrentModel


class AUVSimulator:
  def __init__(self, dt: float = 0.1):
    self.dt = dt

  def step(
    self,
    x: np.ndarray,
    u: np.ndarray,
    action_phys: np.ndarray,
    t: float,
    random_factor_array: np.ndarray,
    snap_on_path: bool = False,
    desired_pose: tuple[float, float, np.ndarray] | None = None,
    ocean_current: OceanCurrentModel | None = None,
  ) -> tuple[np.ndarray, np.ndarray, bool]:
    f = np.zeros((6, 1), dtype=np.float64)
    delta = np.zeros((3, 1), dtype=np.float64)
    f[0, 0] = action_phys[0]
    delta[0, 0] = action_phys[1]
    delta[2, 0] = action_phys[2]

    v_c_ned = np.zeros(3, dtype=np.float64)
    if ocean_current is not None and ocean_current.active:
      v_c_ned = ocean_current.velocity_ned(t, x[:3, 0])
      f = f + ocean_current.drag_wrench_body(x, u, v_c_ned)

    auv = model.AuvModel(x, u, delta, f, t, random_factor_array)
    k1 = auv.get_du(x, u, delta, f)
    k2 = auv.get_du(x, u + 0.5 * self.dt * k1, delta, f)
    k3 = auv.get_du(x, u + 0.5 * self.dt * k2, delta, f)
    k4 = auv.get_du(x, u + self.dt * k3, delta, f)
    u_next = u + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    u_next[3, 0] = 0.0

    trans = model.CoordinateTrans(x, u_next)
    dx = trans.mov_to_fix(x, u_next)
    if ocean_current is not None and ocean_current.active:
      dx[0, 0] += v_c_ned[0]
      dx[1, 0] += v_c_ned[1]
      dx[2, 0] += v_c_ned[2]

    x_next = dx * self.dt + x

    if snap_on_path and desired_pose is not None:
      desired_yaw, desired_pitch, ref_pos = desired_pose
      x_next[0, 0] = ref_pos[0]
      x_next[1, 0] = ref_pos[1]
      x_next[2, 0] = ref_pos[2]
      if abs(desired_yaw - x_next[5, 0]) <= 0.1:
        x_next[5, 0] = desired_yaw
      if abs(desired_pitch - x_next[4, 0]) <= 0.1:
        x_next[4, 0] = desired_pitch

    x_next[3, 0] = 0.0
    x_next[4, 0] = radlimit(x_next[4, 0])
    x_next[5, 0] = radlimit(x_next[5, 0])
    return x_next, u_next, False
