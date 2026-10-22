"""Constrained SAC 用 Actor 与 Twin Critics（LayerNorm + ELU MLP）。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _init_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
  layer.weight.data.uniform_(-init_w, init_w)
  layer.bias.data.uniform_(-init_w, init_w)
  return layer


class MLP(nn.Module):
  """可配置深度的前馈网络，用于连续控制 SAC/CSAC。"""

  def __init__(self, in_dim: int, hidden_dim: int, num_hidden_layers: int):
    super().__init__()
    if num_hidden_layers < 1:
      raise ValueError("num_hidden_layers must be >= 1")
    layers: list[nn.Module] = []
    dim_in = in_dim
    for _ in range(num_hidden_layers):
      layers.append(nn.Linear(dim_in, hidden_dim))
      layers.append(nn.LayerNorm(hidden_dim))
      dim_in = hidden_dim
    self.net = nn.Sequential(*layers)
    self.out_dim = hidden_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.net(x)
    return F.elu(x)


class GaussianActor(nn.Module):
  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 3,
    log_std_min: float = -5.0,
    log_std_max: float = 1.0,
  ):
    super().__init__()
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.backbone = MLP(obs_dim, hidden_dim, num_hidden_layers)
    self.mu = _init_uniform(nn.Linear(self.backbone.out_dim, action_dim))
    self.log_std = _init_uniform(nn.Linear(self.backbone.out_dim, action_dim))

  def forward(self, state: torch.Tensor, deterministic: bool = False):
    x = self.backbone(state)
    mu = torch.tanh(self.mu(x))
    log_std = self.log_std(x).tanh()
    log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
    std = log_std.exp()
    dist = Normal(mu, std)
    z = mu if deterministic else dist.rsample()
    action = torch.tanh(z)
    log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
    log_prob = log_prob.sum(-1, keepdim=True)
    return action, log_prob


class QCritic(nn.Module):
  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    num_hidden_layers: int = 3,
  ):
    super().__init__()
    self.backbone = MLP(obs_dim + action_dim, hidden_dim, num_hidden_layers)
    self.out = _init_uniform(nn.Linear(self.backbone.out_dim, 1))

  def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    x = torch.cat([state, action], dim=-1)
    x = self.backbone(x)
    return self.out(x)
