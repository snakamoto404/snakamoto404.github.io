"""Regenerate the post's minimal, theme-specific schedule figures.

Run from the repository root:
    uv run --with matplotlib python src/content/blogs/machine-learning/weight-norm-artifacts/generate-schedule.py

This is a schematic orthogonal-update model, not a fitted training trajectory.
Dividing the squared L2-norm recurrence by the parameter count gives
    r[t+1]^2 = (1 - lr[t] * wd)^2 * r[t]^2 + lr[t]^2 * g[t]^2.
Here r = RMS(W), g = RMS(G) = 1, and the third panel uses lr / r.
For N parameters, the post's lr / ||W||_2 is that curve divided by sqrt(N).
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = 10_000
WARMUP = STEPS // 100
WEIGHT_DECAY = 0.1
INITIAL_RMS = 0.02
GRADIENT_RMS = 1.0
PEAK_LRS = (0.0015, 0.003, 0.006)
OUTPUT = Path(__file__).resolve().parents[5] / "public" / "plots"


def schedule(step, peak):
    if step <= WARMUP:
        return peak * step / WARMUP
    return peak * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP))) / 2


def simulate(peak):
    # r[t] is the norm before update t; the zero endpoint shows the final norm.
    norm_squared = INITIAL_RMS**2
    rows = []
    for step in range(STEPS + 1):
        lr = schedule(step, peak)
        rms = math.sqrt(norm_squared)
        rows.append((step, lr, rms, lr / rms))
        if step < STEPS:
            norm_squared = (1 - lr * WEIGHT_DECAY)**2 * norm_squared + lr**2 * GRADIENT_RMS**2
    return rows


TRAJECTORIES = [simulate(peak) for peak in PEAK_LRS]
THEMES = {
    "light": {"fg": "#242629", "muted": "#686b70", "axis": "#b8bbbf", "colors": ("#1d6fd1", "#27864b", "#c74446")},
    "dark": {"fg": "#ecf0f7", "muted": "#b0b9c7", "axis": "#677487", "colors": ("#60a5fa", "#6fcf97", "#f28b82")},
}


def render(theme, palette):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": palette["fg"],
        "axes.labelcolor": palette["muted"],
        "xtick.color": palette["muted"],
        "ytick.color": palette["muted"],
        "svg.fonttype": "none",
        "svg.hashsalt": "weight-norm-schedule",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.55))
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.24, top=0.97, wspace=0.20)
    labels = ("lr", "weight rms", "effective lr")
    limits = ((0, 0.0066), (0, 0.18), (0, 0.18))
    for column, ax in enumerate(axes, start=1):
        for rows, peak, color in zip(TRAJECTORIES, PEAK_LRS, palette["colors"]):
            ax.plot([row[0] for row in rows], [row[column] for row in rows],
                    color=color, linewidth=1.65, label=f"{peak:g}", solid_capstyle="round")
        ax.set_ylabel(labels[column - 1], fontsize=11.5, labelpad=12, color=palette["fg"])
        ax.set_xlim(0, STEPS)
        ax.set_ylim(*limits[column - 1])
        ax.set_xticks((0, STEPS // 2, STEPS), ("0", "5k", "10k"))
        ax.set_yticks([])
        if column == 3:
            ax.set_xlabel("Step", labelpad=12, fontsize=10)
        ax.tick_params(length=3, width=0.6, labelsize=9.5, pad=5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(palette["axis"])
            ax.spines[side].set_linewidth(0.6)
        ax.set_facecolor("none")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center", ncol=3,
               bbox_to_anchor=(0.35, 0.075), frameon=False, fontsize=11,
               handlelength=2.2, columnspacing=2.0)
    metadata = {"Date": None, "Title": "Cosine schedule, weight RMS, and effective learning rate",
                "Description": "Schematic: 10,000 steps; 1% linear warmup then cosine decay to zero; weight decay 0.1; initial weight RMS 0.02; constant orthogonal gradient RMS 1. Blue/green/red: peak lr 0.0015/0.003/0.006. Effective lr in RMS units is lr/RMS(W)."}
    fig.savefig(OUTPUT / f"weight-norm-schedule-{theme}.svg", transparent=True, metadata=metadata)
    # Raster previews are for local visual QA, not additional blog assets.
    fig.savefig(Path("/private/tmp") / f"weight-norm-schedule-{theme}.png", dpi=160,
                facecolor="#fdfdfd" if theme == "light" else "#1e293b")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for theme, palette in THEMES.items():
        render(theme, palette)
    for peak, rows in zip(PEAK_LRS, TRAJECTORIES):
        assert rows[0][1] == rows[-1][1] == rows[-1][3] == 0
        assert rows[WARMUP][1] == peak
        assert all(math.isfinite(value) for row in rows for value in row)
        assert max(row[2] for row in rows) < 0.18
        assert max(row[3] for row in rows) < 0.18
        print(f"peak lr={peak:g}; final RMS={rows[-1][2]:.6f}; peak effective lr={max(row[3] for row in rows):.6f}")
