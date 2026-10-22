"""单 episode 测试（evaluate.py 的简化入口）。"""

import argparse
from pathlib import Path
from evaluate import evaluate_episode


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--trajectory", type=str, default="circle", choices=["circle", "sinusoid", "figure8"])
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--device", type=str, default="", choices=["", "cuda", "cpu", "auto", "cuda:0"])
  parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="可选：显式指定 csac_actor checkpoint 路径",
  )
  parser.add_argument(
    "--train-tag",
    type=str,
    default="single_traj_curriculum",
    help="训练实验目录名",
  )
  parser.add_argument(
    "--fast",
    action="store_true",
    help="兼容旧 fast checkpoint（优先级低于 checkpoint 内置 cfg/结构推断）",
  )
  args = parser.parse_args()
  ckpt = Path(args.checkpoint) if args.checkpoint else None
  result = evaluate_episode(
    args.trajectory,
    args.seed,
    checkpoint=ckpt,
    device=args.device or None,
    train_tag=args.train_tag,
    fast=args.fast,
  )
  print(
    f"success={result['success']} collision={result['collision']} "
    f"RMSE={result['rmse_3d']:.4f} energy={result['undiscounted_energy']:.3f} "
    f"reason={result['reason']}"
  )
