from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "circle" / "comparison"

METHODS = [
    {
        "label": "SAC",
        "path": ROOT / "logs" / "circle" / "eval_rollouts_seed777_v1_deterministic.json",
        "color": "#B64342",
        "zorder": 1,
    },
    {
        "label": "SAC-E",
        "path": ROOT / "logs" / "circle" / "eval_rollouts_seed777_v2_smoothed.json",
        "color": "#42949E",
        "zorder": 2,
    },
    {
        "label": "SEAL",
        "path": ROOT / "logs" / "circle" / "eval_rollouts_seed777_v2b_medium_smooth.json",
        "color": "#0F4D92",
        "zorder": 3,
    },
]

SIGNALS = [
    ("thrust", "Thrust", "Thrust (N)", lambda acts: acts[:, 0], (-10.0, 465.0)),
    (
        "vertical_rudder",
        "Vertical rudder",
        "Vertical rudder (deg)",
        lambda acts: np.rad2deg(acts[:, 1]),
        (-22.0, 22.0),
    ),
    (
        "horizontal_rudder",
        "Horizontal rudder",
        "Horizontal rudder (deg)",
        lambda acts: np.rad2deg(acts[:, 2]),
        (-22.0, 22.0),
    ),
]


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.family": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 12,
            "axes.linewidth": 1.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def load_series() -> list[dict]:
    series = []
    for method in METHODS:
        rollouts = json.loads(method["path"].read_text(encoding="utf-8"))
        if not rollouts:
            raise ValueError(f"No rollouts found in {method['path']}")
        ep = rollouts[0]
        series.append(
            {
                **method,
                "times": np.asarray(ep["times"], dtype=float),
                "actions": np.asarray(ep["actions_phys"], dtype=float),
            }
        )
    return series


def plot_signal(series: list[dict], signal: tuple[str, str, str, object, tuple[float, float]]) -> None:
    key, title, ylabel, extractor, ylim = signal
    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    ax.axhline(0.0, color="#4D4D4D", linewidth=1.2, alpha=0.55, zorder=0)
    for item in series:
        y = extractor(item["actions"])
        ax.plot(
            item["times"],
            y,
            color=item["color"],
            linewidth=1.35,
            alpha=0.95,
            label=item["label"],
            zorder=item["zorder"],
        )

    ax.set_title(title, fontsize=18, pad=42)
    ax.set_xlabel("Time (s)", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color="#CFCECE", linewidth=0.9, alpha=0.55)
    ax.tick_params(axis="both", labelsize=13, width=1.4, length=4.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=13,
        handlelength=2.0,
        columnspacing=1.4,
    )

    fig.tight_layout(pad=1.2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"04_actuator_{key}_seal_sace_sac.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    series = load_series()
    for signal in SIGNALS:
        plot_signal(series, signal)


if __name__ == "__main__":
    main()
