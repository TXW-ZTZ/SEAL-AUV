"""One-off: plot three reference trajectories (circle / sinusoid / figure8)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajectory import CircleTrajectory, SinusoidTrajectory, Figure8Trajectory
from config import PathConfig

t = np.linspace(0, 99.9, 1000)
trajs = [
    (CircleTrajectory(), "Circle (RT1)", "#e74c3c"),
    (SinusoidTrajectory(), "Sinusoid (RT2)", "#2ecc71"),
    (Figure8Trajectory(), "Figure-8 (RT3)", "#3498db"),
]

fig = plt.figure(figsize=(16, 5))

ax3d = fig.add_subplot(131, projection="3d")
for traj, label, color in trajs:
    pts = np.array([traj.position(ti) for ti in t])
    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.8, label=label)
    ax3d.scatter(*pts[0], color=color, s=40, marker="o", zorder=5)
    ax3d.scatter(*pts[-1], color=color, s=40, marker="s", zorder=5)
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
ax3d.set_title("3D reference trajectories")
ax3d.legend(fontsize=8, loc="upper left")

ax_xy = fig.add_subplot(132)
for traj, label, color in trajs:
    pts = np.array([traj.position(ti) for ti in t])
    ax_xy.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8, label=label)
    ax_xy.scatter(pts[0, 0], pts[0, 1], color=color, s=30, zorder=5)
ax_xy.set_xlabel("X (m)")
ax_xy.set_ylabel("Y (m)")
ax_xy.set_title("XY projection")
ax_xy.set_aspect("equal", adjustable="box")
ax_xy.grid(True, alpha=0.3)
ax_xy.legend(fontsize=8)

ax_xz = fig.add_subplot(133)
for traj, label, color in trajs:
    pts = np.array([traj.position(ti) for ti in t])
    ax_xz.plot(pts[:, 0], pts[:, 2], color=color, lw=1.8, label=label)
    ax_xz.scatter(pts[0, 0], pts[0, 2], color=color, s=30, zorder=5)
ax_xz.set_xlabel("X (m)")
ax_xz.set_ylabel("Z (m)")
ax_xz.set_title("XZ projection")
ax_xz.grid(True, alpha=0.3)
ax_xz.legend(fontsize=8)

plt.tight_layout()
paths = PathConfig()
paths.ensure()
out = paths.figure_dir / "reference_trajectories.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)

# 分开展示，便于对比形状
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5), subplot_kw={"projection": "3d"})
for ax, (traj, label, color) in zip(axes, trajs):
    pts = np.array([traj.position(ti) for ti in t])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=2)
    ax.scatter(*pts[0], color=color, s=50, marker="o", label="start")
    ax.scatter(*pts[-1], color=color, s=50, marker="s", label="end")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(label)
    ax.legend(fontsize=7)
plt.tight_layout()
out2 = paths.figure_dir / "reference_trajectories_separate.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print("Saved:", out2)
