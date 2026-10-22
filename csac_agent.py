"""
Constrained Soft Actor-Critic (CSAC) — Tao et al. Section 3.3–3.4

- Twin Qr, Twin Qc
- Actor: min E[ α log π - Qr + λ Qc ]  (式 25)
- Dual ascent: λ ← max(0, λ + ηλ (Jc - θ))  (式 26)
"""

from collections import deque
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import CSACConfig, PathConfig
from cmdp_replay_buffer import CMDPReplayBuffer
from rl_networks import GaussianActor, QCritic
from utils import describe_torch_device, resolve_torch_device, setup_torch_runtime


class CSACAgent:
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
    print("CSAC device:", describe_torch_device(self.device))

    net_kw = dict(
      hidden_dim=cfg.hidden_dim,
      num_hidden_layers=cfg.num_hidden_layers,
    )
    self.actor = GaussianActor(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qr1 = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qr2 = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qc1 = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qc2 = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)

    self.qr1_t = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qr2_t = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qc1_t = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    self.qc2_t = QCritic(cfg.obs_dim, cfg.action_dim, **net_kw).to(self.device)
    for tgt, src in [
      (self.qr1_t, self.qr1),
      (self.qr2_t, self.qr2),
      (self.qc1_t, self.qc1),
      (self.qc2_t, self.qc2),
    ]:
      tgt.load_state_dict(src.state_dict())

    adam_kw = dict(eps=cfg.adam_eps)
    self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, **adam_kw)
    self.qr1_opt = optim.Adam(self.qr1.parameters(), lr=cfg.critic_lr, **adam_kw)
    self.qr2_opt = optim.Adam(self.qr2.parameters(), lr=cfg.critic_lr, **adam_kw)
    self.qc1_opt = optim.Adam(self.qc1.parameters(), lr=cfg.critic_lr, **adam_kw)
    self.qc2_opt = optim.Adam(self.qc2.parameters(), lr=cfg.critic_lr, **adam_kw)

    self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
    self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr, **adam_kw)
    self.target_entropy_scale = float(cfg.target_entropy_scale)
    self.target_entropy = -float(cfg.action_dim) * self.target_entropy_scale
    self.alpha_frozen = False

    lam0 = float(np.clip(cfg.lambda_init, cfg.lambda_min, cfg.lambda_max))
    log_lam0 = -float("inf") if lam0 <= 0.0 else float(np.log(lam0))
    self.log_lambda = torch.tensor(
      log_lam0, device=self.device, dtype=torch.float32, requires_grad=False
    )

    self.replay = CMDPReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.buffer_size)
    self.use_amp = bool(cfg.use_amp and self._use_cuda)
    self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
    self.total_step = 0
    self.episodes = 0
    self.success = 0
    self.success_rate_window = deque(maxlen=100)
    self.trajectory_demo = []
    self.scores = []
    self.lambda_history: list[float] = []
    self.lambda_actor_scale: float = 1.0

  @property
  def alpha(self):
    return self.log_alpha.exp()

  @property
  def lam(self):
    return self.log_lambda.exp().clamp(self.cfg.lambda_min, self.cfg.lambda_max)

  def _set_lambda(self, value: float) -> float:
    """将 λ 限制在 [lambda_min, lambda_max] 并写回 log_lambda。"""
    clipped = float(np.clip(value, self.cfg.lambda_min, self.cfg.lambda_max))
    log_value = -float("inf") if clipped <= 0.0 else float(np.log(clipped))
    self.log_lambda.fill_(log_value)
    return clipped

  def update_lambda_from_episode(self, episode_jc: float) -> float:
    """
    用 episode 折扣能耗 Jc 做对偶上升（式 26）：
      λ ← clip(λ + η·(Jc − θ), λ_min, λ_max)
    """
    try:
      theta = float(self.cfg.energy_budget_theta)
    except (TypeError, ValueError):
      return float(self.lam.detach().cpu())
    if not np.isfinite(theta) or not np.isfinite(float(episode_jc)):
      return float(self.lam.detach().cpu())
    violation = float(episode_jc) - theta
    lam_new = float(self.lam.detach().cpu()) + self.cfg.lambda_dual_step * violation
    lam_new = self._set_lambda(lam_new)
    self.lambda_history.append(lam_new)
    return lam_new

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

  def set_lambda_actor_scale(self, scale: float):
    self.lambda_actor_scale = float(np.clip(scale, 0.0, 1.0))

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
    costs = self._numpy_to_device(batch["costs"]).unsqueeze(1)
    next_obs = self._numpy_to_device(batch["next_obs"])
    done = self._numpy_to_device(batch["done"]).unsqueeze(1)
    mask = 1.0 - done
    amp = torch.autocast(device_type=self.device.type, enabled=self.use_amp)

    with torch.no_grad(), amp:
      next_action, next_logp = self.actor(next_obs)
      qr_next = torch.min(self.qr1_t(next_obs, next_action), self.qr2_t(next_obs, next_action))
      qc_next = torch.min(self.qc1_t(next_obs, next_action), self.qc2_t(next_obs, next_action))
      qr_target = rews + mask * self.cfg.gamma * (qr_next - self.alpha * next_logp)
      qc_target = costs + mask * self.cfg.gamma * qc_next

    with amp:
      qr1_loss = F.smooth_l1_loss(self.qr1(obs, acts), qr_target)
      qr2_loss = F.smooth_l1_loss(self.qr2(obs, acts), qr_target)
      qc1_loss = F.smooth_l1_loss(self.qc1(obs, acts), qc_target)
      qc2_loss = F.smooth_l1_loss(self.qc2(obs, acts), qc_target)

    for module, opt, loss in [
      (self.qr1, self.qr1_opt, qr1_loss),
      (self.qr2, self.qr2_opt, qr2_loss),
      (self.qc1, self.qc1_opt, qc1_loss),
      (self.qc2, self.qc2_opt, qc2_loss),
    ]:
      opt.zero_grad(set_to_none=True)
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(opt)
      nn.utils.clip_grad_norm_(module.parameters(), self.cfg.grad_clip_norm)
      self.scaler.step(opt)

    with amp:
      new_action, logp = self.actor(obs)
      qr_val = torch.min(self.qr1(obs, new_action), self.qr2(obs, new_action))
      qc_val = torch.min(self.qc1(obs, new_action), self.qc2(obs, new_action))
      lam_actor = self.lam.detach() * float(self.lambda_actor_scale)
      actor_loss = (self.alpha.detach() * logp - qr_val + lam_actor * qc_val).mean()
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

    self._soft_update(self.qr1, self.qr1_t)
    self._soft_update(self.qr2, self.qr2_t)
    self._soft_update(self.qc1, self.qc1_t)
    self._soft_update(self.qc2, self.qc2_t)

    return {
      "actor_loss": float(actor_loss.detach().cpu()),
      "qr1_loss": float(qr1_loss.detach().cpu()),
      "lambda": float(self.lam.detach().cpu()),
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
        "qr1": self.qr1.state_dict(),
        "qc1": self.qc1.state_dict(),
        "log_lambda": self.log_lambda.detach().cpu(),
        "cfg": self.cfg,
      },
      path / f"csac_actor_seed{seed}.pt",
    )
    with open(path / f"scores_CSAC_{seed}.pkl", "wb") as f:
      pickle.dump(scores, f)
    with open(path / f"trajectory_CSAC_{seed}.pkl", "wb") as f:
      pickle.dump(trajectories, f)

  def load(self, checkpoint: Path):
    state = torch.load(checkpoint, map_location=self.device, weights_only=False)
    self.actor.load_state_dict(state["actor"])
    if "log_lambda" in state:
      lam = float(state["log_lambda"].exp().item())
      self._set_lambda(lam)
