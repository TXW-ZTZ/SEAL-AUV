"""
Standard Soft Actor-Critic (SAC) — unconstrained baseline for CSAC comparison.

- Twin Q critics (Qr only; no cost critic / Lagrange multiplier)
- Actor: min E[ α log π - Qr ]
- Automatic entropy tuning (same schedule hooks as CSACAgent)
"""

from collections import deque
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import CSACConfig
from cmdp_replay_buffer import CMDPReplayBuffer
from rl_networks import GaussianActor, QCritic
from utils import describe_torch_device, resolve_torch_device, setup_torch_runtime


class SACAgent:
  def __init__(self, cfg: CSACConfig, device: str | None = None):
    self.cfg = cfg
    device_name = device if device is not None else cfg.device
    try:
      self.device = resolve_torch_device(device_name)
    except RuntimeError as exc:
      print(f"警告: {exc}")
      print("回退到 CPU 继续运行。")
      self.device = torch.device("cpu")
    self._use_cuda = self.device.type == "cuda"
    setup_torch_runtime(self.device)
    print("SAC device:", describe_torch_device(self.device))

    net_kw = dict(
      hidden_dim=cfg.hidden_dim,
      num_hidden_layers=cfg.num_hidden_layers,
    )
    self.actor = GaussianActor(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.q1 = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.q2 = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)

    self.q1_t = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.q2_t = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    for tgt, src in [(self.q1_t, self.q1), (self.q2_t, self.q2)]:
      tgt.load_state_dict(src.state_dict())

    adam_kw = dict(eps=cfg.adam_eps)
    self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, **adam_kw)
    self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg.critic_lr, **adam_kw)
    self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg.critic_lr, **adam_kw)

    self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
    self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr, **adam_kw)
    self.target_entropy_scale = float(cfg.target_entropy_scale)
    self.target_entropy = -float(cfg.action_dim) * self.target_entropy_scale
    self.alpha_frozen = False

    self.replay = CMDPReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.buffer_size)
    self.use_amp = bool(cfg.use_amp and self._use_cuda)
    self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
    self.total_step = 0
    self.episodes = 0
    self.success = 0
    self.success_rate_window = deque(maxlen=100)
    self.trajectory_demo = []
    self.scores = []

  @property
  def alpha(self):
    return self.log_alpha.exp()

  def _numpy_to_device(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if self._use_cuda:
      return t.pin_memory().to(self.device, dtype=dtype, non_blocking=True)
    return t.to(self.device, dtype=dtype)

  def set_target_entropy_scale(self, scale: float):
    self.target_entropy_scale = float(scale)
    self.target_entropy = -float(self.cfg.action_dim) * self.target_entropy_scale

  def set_alpha_frozen(self, frozen: bool):
    self.alpha_frozen = bool(frozen)

  def select_action(
    self,
    obs: np.ndarray,
    random_explore: bool = False,
    deterministic: bool | None = None,
  ) -> np.ndarray:
    if random_explore:
      return np.random.uniform(-1.0, 1.0, size=self.cfg.action_dim).astype(np.float32)
    if deterministic is None:
      deterministic = True
    obs_t = self._numpy_to_device(obs).unsqueeze(0)
    with torch.no_grad():
      action, _ = self.actor(obs_t, deterministic=deterministic)
    return action.cpu().numpy()[0].astype(np.float32)

  def update(self) -> dict:
    batch = self.replay.sample(self.cfg.batch_size)
    obs = self._numpy_to_device(batch["obs"])
    acts = self._numpy_to_device(batch["acts"])
    rews = self._numpy_to_device(batch["rews"]).unsqueeze(1)
    next_obs = self._numpy_to_device(batch["next_obs"])
    done = self._numpy_to_device(batch["done"]).unsqueeze(1)
    mask = 1.0 - done
    amp = torch.autocast(device_type=self.device.type, enabled=self.use_amp)

    with torch.no_grad(), amp:
      next_action, next_logp = self.actor(next_obs)
      q_next = torch.min(self.q1_t(next_obs, next_action), self.q2_t(next_obs, next_action))
      q_target = rews + mask * self.cfg.gamma * (q_next - self.alpha * next_logp)

    with amp:
      q1_loss = F.smooth_l1_loss(self.q1(obs, acts), q_target)
      q2_loss = F.smooth_l1_loss(self.q2(obs, acts), q_target)

    for module, opt, loss in [
      (self.q1, self.q1_opt, q1_loss),
      (self.q2, self.q2_opt, q2_loss),
    ]:
      opt.zero_grad(set_to_none=True)
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(opt)
      nn.utils.clip_grad_norm_(module.parameters(), self.cfg.grad_clip_norm)
      self.scaler.step(opt)

    with amp:
      new_action, logp = self.actor(obs)
      q_val = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
      actor_loss = (self.alpha.detach() * logp - q_val).mean()
      alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

    self.actor_opt.zero_grad(set_to_none=True)
    self.scaler.scale(actor_loss).backward()
    self.scaler.unscale_(self.actor_opt)
    nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
    self.scaler.step(self.actor_opt)

    if not self.alpha_frozen:
      self.alpha_opt.zero_grad(set_to_none=True)
      self.scaler.scale(alpha_loss).backward()
      self.scaler.step(self.alpha_opt)
    self.scaler.update()

    self._soft_update(self.q1, self.q1_t)
    self._soft_update(self.q2, self.q2_t)

    return {
      "actor_loss": float(actor_loss.detach().cpu()),
      "q1_loss": float(q1_loss.detach().cpu()),
      "alpha": float(self.alpha.detach().cpu()),
    }

  def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module):
    tau = self.cfg.tau
    with torch.no_grad():
      for p, pt in zip(src.parameters(), tgt.parameters()):
        pt.data.mul_(1.0 - tau)
        pt.data.add_(tau * p.data)

  def save(self, path: Path, seed: int, scores: list, trajectories: list):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "actor": self.actor.state_dict(),
        "q1": self.q1.state_dict(),
        "q2": self.q2.state_dict(),
        "cfg": self.cfg,
        "algorithm": "sac",
      },
      path / f"sac_actor_seed{seed}.pt",
    )
    with open(path / f"scores_SAC_{seed}.pkl", "wb") as f:
      pickle.dump(scores, f)
    with open(path / f"trajectory_SAC_{seed}.pkl", "wb") as f:
      pickle.dump(trajectories, f)

  def load(self, checkpoint: Path):
    state = torch.load(checkpoint, map_location=self.device, weights_only=False)
    self.actor.load_state_dict(state["actor"])
    if "q1" in state:
      self.q1.load_state_dict(state["q1"])
    if "q2" in state:
      self.q2.load_state_dict(state["q2"])
