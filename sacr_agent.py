"""
SAC with rEnergy (SACr) — Tao et al. (2026) Section 5.2 reward-shaping baseline.

与三种算法的能耗处理方式
--------------------------
1. SAC（sac_agent.py，论文 *SAC without rEnergy*）
   - 回放与 Bellman 目标仅使用任务回报 r_task（env 返回的 reward）。
   - 能耗 c_t 仍由环境计算并写入 buffer，但 **不参与** Q 网络更新。
   - 优化目标：max E[Σ γ^t r_task]。

2. SACr（本文件，论文 *SAC with rEnergy*）
   - 将瞬时能耗以 **固定权重** 并入回报（reward shaping）：
       r_shaped = r_task − w_energy · c_t
   - 回放中 rews 存 r_shaped；c_t 仍单独存 cost_buf 供日志与 Jc 统计。
   - 仍是标准 SAC 结构（Twin Q，无 Q_c，无拉格朗日 λ）。
   - 优化目标：max E[Σ γ^t (r_task − w_energy · c_t)]。
   - w_energy 默认 0.1，与 CSAC 初始 λ（lambda_init=0.05）同量级，可通过 --w-energy 覆盖。

3. CSAC（csac_agent.py，论文 *Constrained SAC*）
   - r_task 与 c_t **分离**；Qr 学任务回报，Qc 学能耗代价。
   - Actor：min E[α log π − Qr + λ Qc]；λ 对偶上升使 Jc ≤ θ。
   - 优化目标：max Jr(π) s.t. Jc(π) ≤ θ（CMDP 显式约束）。

实现说明
--------
- 网络结构、超参数、探索调度与 SACAgent 完全一致。
- 与 SAC 的唯一算法差异：train_sacr.py 在 replay.store 前调用 shape_reward()。
- 评估与绘图仍报告 **纯任务回报** 与 **独立能耗指标**，便于三方公平对比。
"""

from __future__ import annotations

from pathlib import Path
import pickle

import torch

from sac_agent import SACAgent


class SACrAgent(SACAgent):
  """Standard SAC trained on shaped reward r_task − w_energy · c."""

  def __init__(self, cfg, device: str | None = None):
    super().__init__(cfg, device=device)
    print(f"SACr w_energy={float(cfg.energy_reward_penalty):.4f} (r_shaped = r_task - w*c)")

  @staticmethod
  def shape_reward(task_reward: float, energy_cost: float, w_energy: float) -> float:
    return float(task_reward) - float(w_energy) * float(energy_cost)

  def save(self, path: Path, seed: int, scores: list, trajectories: list):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "actor": self.actor.state_dict(),
        "q1": self.q1.state_dict(),
        "q2": self.q2.state_dict(),
        "cfg": self.cfg,
        "algorithm": "sacr",
        "w_energy": float(self.cfg.energy_reward_penalty),
      },
      path / f"sacr_actor_seed{seed}.pt",
    )
    with open(path / f"scores_SACr_{seed}.pkl", "wb") as f:
      pickle.dump(scores, f)
    with open(path / f"trajectory_SACr_{seed}.pkl", "wb") as f:
      pickle.dump(trajectories, f)

  def load(self, checkpoint: Path):
    state = torch.load(checkpoint, map_location=self.device, weights_only=False)
    self.actor.load_state_dict(state["actor"])
    if "q1" in state:
      self.q1.load_state_dict(state["q1"])
    if "q2" in state:
      self.q2.load_state_dict(state["q2"])
