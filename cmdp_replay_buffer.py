"""CMDP 回放：同时存储任务回报 r 与能耗代价 c。"""

import numpy as np
from typing import Dict


class CMDPReplayBuffer:
  def __init__(self, obs_dim: int, act_dim: int, size: int):
    self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
    self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
    self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.cost_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.max_size = size
    self.ptr = 0
    self.size = 0

  def store(self, obs, act, rew, cost, next_obs, done):
    self.obs_buf[self.ptr] = obs
    self.acts_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.cost_buf[self.ptr] = cost
    self.next_obs_buf[self.ptr] = next_obs
    self.done_buf[self.ptr] = float(done)
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
    idx = np.random.choice(self.size, size=batch_size, replace=False)
    return {
      "obs": self.obs_buf[idx],
      "acts": self.acts_buf[idx],
      "rews": self.rew_buf[idx],
      "costs": self.cost_buf[idx],
      "next_obs": self.next_obs_buf[idx],
      "done": self.done_buf[idx],
    }

  def trim_oldest(self, fraction: float) -> int:
    """丢弃最早 fraction 比例的样本，减轻早期高 λ 轨迹对 Q 网络的持续影响。"""
    fraction = float(np.clip(fraction, 0.0, 0.95))
    drop = int(self.size * fraction)
    if drop <= 0:
      return 0
    keep = self.size - drop
    bufs = (
      self.obs_buf,
      self.next_obs_buf,
      self.acts_buf,
      self.rew_buf,
      self.cost_buf,
      self.done_buf,
    )
    if self.size < self.max_size:
      for buf in bufs:
        buf[:keep] = buf[drop:self.size]
    else:
      idx = (self.ptr + np.arange(self.size, dtype=np.int64)) % self.max_size
      keep_idx = idx[drop:]
      for buf in bufs:
        buf[:keep] = buf[keep_idx]
    self.size = keep
    self.ptr = keep % self.max_size
    return drop

  def __len__(self) -> int:
    return self.size
