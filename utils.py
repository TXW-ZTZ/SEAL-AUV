"""通用数值与动作工具。"""

from __future__ import annotations

import numpy as np
import torch


ENERGY_COMPONENT_KEYS = (
  "thrust_energy",
  "rudder_energy",
  "delta_thrust_energy",
  "delta_rudder_energy",
  "propulsion_energy",
  "steering_energy",
  "total_energy",
)


def radlimit(rad: float) -> float:
  rad = float(rad)
  for _ in range(1000):
    if rad > np.pi:
      rad -= 2.0 * np.pi
    elif rad < -np.pi:
      rad += 2.0 * np.pi
    else:
      break
  return rad


def denormalize_action(
  action_norm: np.ndarray,
  thrust_min: float,
  thrust_max: float,
  rudder_max_rad: float,
) -> np.ndarray:
  a = np.asarray(action_norm, dtype=np.float64).reshape(3)
  scale = (thrust_max - thrust_min) / 2.0
  reloc = thrust_max - scale
  thrust = a[0] * scale + reloc
  delta0 = a[1] * rudder_max_rad
  delta2 = a[2] * rudder_max_rad
  out = np.array([thrust, delta0, delta2], dtype=np.float64)
  out[0] = np.clip(out[0], thrust_min, thrust_max)
  out[1] = np.clip(out[1], -rudder_max_rad, rudder_max_rad)
  out[2] = np.clip(out[2], -rudder_max_rad, rudder_max_rad)
  return out


def normalize_action(
  action_phys: np.ndarray,
  thrust_min: float,
  thrust_max: float,
  rudder_max_rad: float,
) -> np.ndarray:
  """与 denormalize_action 互逆，输出 [-1, 1] 归一化动作。"""
  a = np.asarray(action_phys, dtype=np.float64).reshape(3)
  scale = (thrust_max - thrust_min) / 2.0
  reloc = thrust_max - scale
  thrust_norm = (a[0] - reloc) / max(scale, 1e-9)
  rudder_norm = a[1] / max(rudder_max_rad, 1e-9)
  rudder2_norm = a[2] / max(rudder_max_rad, 1e-9)
  out = np.array([thrust_norm, rudder_norm, rudder2_norm], dtype=np.float32)
  out[0] = np.clip(out[0], -1.0, 1.0)
  out[1] = np.clip(out[1], -1.0, 1.0)
  out[2] = np.clip(out[2], -1.0, 1.0)
  return out


def discounted_sum(values: list[float], gamma: float) -> float:
  if not values:
    return 0.0
  w = np.power(gamma, np.arange(len(values), dtype=np.float64))
  return float(np.dot(w, np.asarray(values, dtype=np.float64)))


def episode_energy_fields(env) -> dict[str, float]:
  """Return cumulative energy components for an evaluated episode."""
  components = getattr(env, "episode_energy_components", None) or {}
  total = float(getattr(env, "episode_cost", components.get("total_energy", 0.0)))
  fields = {key: float(components.get(key, 0.0)) for key in ENERGY_COMPONENT_KEYS}
  fields["total_energy"] = total
  return fields


def episode_distance_m(start, positions) -> float:
  """AUV 沿实际路径累计前进距离 (m)，含起点到首帧位置。"""
  start_pt = np.asarray(start, dtype=np.float64).reshape(3)
  pts = [start_pt]
  for p in positions:
    pts.append(np.asarray(p, dtype=np.float64).reshape(3))
  if len(pts) < 2:
    return 0.0
  arr = np.vstack(pts)
  return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def distance_before_collision_m(start, positions, collision: bool) -> float | None:
  """碰撞 episode 中，碰撞发生前 AUV 累计前进距离 (m)；未碰撞返回 None。"""
  if not collision:
    return None
  return episode_distance_m(start, positions)


def resolve_torch_device(name: str = "cuda") -> torch.device:
  """解析训练/推理设备：默认优先 CUDA。"""
  key = (name or "cuda").strip().lower()
  if key in ("auto", "default"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if key.startswith("cuda"):
    if not torch.cuda.is_available():
      raise RuntimeError(
        "已请求 CUDA，但当前 PyTorch 未检测到可用 GPU。"
        "请安装带 CUDA 的 PyTorch，例如："
        "pip install torch --index-url https://download.pytorch.org/whl/cu124"
      )
    return torch.device(key)
  return torch.device(key)


def setup_torch_runtime(device: torch.device) -> None:
  """在 GPU 上启用 cuDNN 与 TF32 等加速选项。"""
  if device.type != "cuda":
    return
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  if hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True
  if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = True


def describe_torch_device(device: torch.device) -> str:
  if device.type != "cuda":
    return (
      f"{device}（未使用 GPU；若本机有 NVIDIA 显卡，请安装 CUDA 版 PyTorch 后重试）"
    )
  index = device.index if device.index is not None else torch.cuda.current_device()
  name = torch.cuda.get_device_name(index)
  cap = torch.cuda.get_device_capability(index)
  return f"{device} — {name} (compute {cap[0]}.{cap[1]})"
