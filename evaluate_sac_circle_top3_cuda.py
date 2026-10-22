"""Evaluate the circle-trained SAC actor on the circle scenario using CUDA."""

from pathlib import Path

import evaluate_custom_csac_circle_top3 as runner


ROOT = Path(__file__).resolve().parent
runner.CHECKPOINT = ROOT / "checkpoints" / "single_traj_curriculum_sac" / "circle_seed666" / "sac_actor_seed666_best.pt"
runner.OUT = ROOT / "figures" / "sac_circle_top3_cuda" / "episodes1000"
runner.LOG = ROOT / "logs" / "sac_circle_top3_cuda" / "episodes1000" / "rollouts.json"
runner.DEVICE = "cuda"
runner.figure.ALGORITHM_TITLE = "SAC"


if __name__ == "__main__":
  runner.main()
