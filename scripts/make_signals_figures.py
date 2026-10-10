"""Generate the figures in docs/figures/signals/ for docs/signals-math.md.

Every angle and number drawn is computed by pipeline.signals itself, on synthetic points,
so the figures stay consistent with the code. Re-run after changing signals.py:

    python -m scripts.make_signals_figures

This is plotting code only; nothing in the pipeline imports it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter

from pipeline import signals

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "signals"

# Reference palette from the data-viz method: categorical slots 1-3 validated for all
# pairs, and blue/red as the two poles of a sign (sag vs pike).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
BLUE_TINT = "#cde2fb"

L = signals.LEFT

# A 2D point as the drawing helpers take it: a tuple or a NumPy array of x, y.
XY = tuple[float, float] | np.ndarray

plt.switch_backend("Agg")
plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)


# ---------------------------------------------------------------------------- helpers


def landmarks_for(points: dict[int, tuple[float, float]]) -> np.ndarray:
    """A one-frame (1, 33, 4) landmark array with only the given points set."""
    lm = np.full((1, 33, 4), np.nan)
    for index, (x, y) in points.items():
        lm[0, index, :2] = (x, y)
        lm[0, index, 3] = 1.0
    return lm


def diagram(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """A geometry panel in image coordinates: equal scale, y growing downward."""
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_aspect("equal")
    ax.axis("off")


def point(ax: plt.Axes, xy: XY, label: str = "", dx: float = 0, dy: float = 0, **text: str) -> None:
    """A joint: ink dot with a 2px surface ring, and an optional label beside it."""
    ax.plot(*xy, "o", ms=9, color=INK, mec=SURFACE, mew=2, zorder=6)
    if label:
        ax.text(xy[0] + dx, xy[1] + dy, label, color=INK_2, fontsize=10, **text)


def arc(
    ax: plt.Axes, center: XY, towards_a: XY, towards_b: XY, radius: float, color: str = INK_2
) -> float:
    """Draw the smaller arc at center between two directions; return its middle angle."""
    a1 = np.arctan2(towards_a[1], towards_a[0])
    a2 = np.arctan2(towards_b[1], towards_b[0])
    sweep = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    phi = a1 + np.linspace(0, sweep, 80)
    ax.plot(
        center[0] + radius * np.cos(phi),
        center[1] + radius * np.sin(phi),
        color=color,
        lw=1.5,
        solid_capstyle="butt",
    )
    return a1 + sweep / 2


def at_angle(center: XY, phi: float, radius: float) -> tuple[float, float]:
    return center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11, color=INK, pad=6)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote docs/figures/signals/{name}")


# ---------------------------------------------------------------------------- figures


def fig_coordinates() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    diagram(ax, (-6, 166), (-14, 96))
    ax.add_patch(Rectangle((0, 0), 160, 90, fill=False, ec=AXIS, lw=1.2))
    ax.text(160, -4, "the video frame", color=MUTED, fontsize=9, ha="right")
    point(ax, (0, 0), "(0, 0)", dx=2, dy=-4)
    arrow = dict(arrowstyle="-|>", color=INK_2, lw=1.5)
    ax.annotate("", xy=(52, 9), xytext=(8, 9), arrowprops=arrow)
    ax.text(56, 9, "x grows to the right", color=INK_2, fontsize=10, va="center")
    ax.annotate("", xy=(8, 60), xytext=(8, 13), arrowprops=arrow)
    ax.text(12, 50, "y grows\nDOWNWARD", color=INK, fontsize=10, fontweight="bold")
    point(ax, (112, 26), "y = 26\nhigher on screen", dx=5, dy=2, va="center")
    point(ax, (112, 70), "y = 70\nlower on screen", dx=5, dy=2, va="center")
    ax.plot([112, 112], [30, 66], color=AXIS, lw=1)
    save(fig, "coordinates.png")


def fig_angle() -> None:
    a, b, c = np.array([30.0, 10.0]), np.array([50.0, 70.0]), np.array([125.0, 85.0])
    theta = float(signals.angle(a, b, c))
    u, v = a - b, c - b
    dot = float(u @ v)
    norms = float(np.linalg.norm(u) * np.linalg.norm(v))

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[1.05, 1])
    diagram(ax, (0, 150), (-5, 100))
    arrow = dict(arrowstyle="-|>", lw=2.2, shrinkA=0, shrinkB=6)
    ax.annotate("", xy=a, xytext=b, arrowprops={**arrow, "color": BLUE})
    ax.annotate("", xy=c, xytext=b, arrowprops={**arrow, "color": ORANGE})
    mid = arc(ax, b, u, v, 13)
    ax.text(
        *at_angle(b, mid, 24),
        f"θ = {theta:.1f}°",
        color=INK,
        fontsize=11,
        ha="center",
        va="center",
        fontweight="bold",
    )
    point(ax, a, "a  (shoulder)", dx=5, dy=-2)
    point(ax, b, "b  (elbow)\nthe angle is measured here", dx=-4, dy=9, ha="right")
    point(ax, c, "c  (wrist)", dx=-2, dy=9, ha="center")
    ax.text(26, 45, "u = a − b", color=INK_2, fontsize=10, ha="right")
    ax.text(90, 72, "v = c − b", color=INK_2, fontsize=10)
    title(ax, "The angle at b, between the two lines leaving it")

    ax_t.axis("off")
    lines = [
        ("u = a − b", f"= ({u[0]:.0f}, {u[1]:.0f})"),
        ("v = c − b", f"= ({v[0]:.0f}, {v[1]:.0f})"),
        ("u · v", f"= ({u[0]:.0f})({v[0]:.0f}) + ({u[1]:.0f})({v[1]:.0f}) = {dot:.0f}"),
        ("|u| · |v|", f"= {np.linalg.norm(u):.1f} × {np.linalg.norm(v):.1f} = {norms:.1f}"),
        ("cos θ", f"= {dot:.0f} / {norms:.1f} = {dot / norms:.3f}"),
        ("θ", f"= arccos({dot / norms:.3f}) = {theta:.1f}°"),
    ]
    ax_t.text(
        0,
        0.97,
        "cos θ  =  (u · v) / (|u| · |v|)",
        fontsize=13,
        fontweight="bold",
        transform=ax_t.transAxes,
        va="top",
    )
    for i, (lhs, rhs) in enumerate(lines):
        y = 0.80 - i * 0.105
        ax_t.text(0.0, y, lhs, color=INK_2, transform=ax_t.transAxes, va="top")
        ax_t.text(0.22, y, rhs, color=INK, transform=ax_t.transAxes, va="top")
    ax_t.text(
        0,
        0.1,
        "np.clip keeps cos θ inside [−1, 1] before arccos: rounding can\n"
        "produce 1.0000000002, and arccos of that is NaN.",
        color=MUTED,
        fontsize=9.5,
        transform=ax_t.transAxes,
        va="top",
    )
    save(fig, "angle.png")


def fig_upper_arm() -> None:
    cases = [
        ("Top of the rep", (0.0, 100.0)),
        ("Halfway down", (70.0, 70.0)),
        ("Bottom: parallel to the floor", (100.0, 0.0)),
        ("Mirrored: same angle", (-70.0, 70.0)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, (name, elbow) in zip(axes, cases, strict=True):
        diagram(ax, (-118, 125), (-28, 118))
        lm = landmarks_for({L.shoulder: (0.0, 0.0), L.elbow: elbow})
        value = float(signals.upper_arm_angle_series(lm, L)[0])
        side = -1.0 if elbow[0] < 0 else 1.0
        ax.plot([0, side * 112], [0, 0], color=MUTED, lw=1.3, ls=(0, (5, 4)))
        lying_flat = value < 1
        if not lying_flat:
            ax.text(
                side * 122,
                -6,
                "horizontal",
                color=MUTED,
                fontsize=9,
                ha="right" if side > 0 else "left",
            )
        ax.plot([0, elbow[0]], [0, elbow[1]], color=BLUE, lw=3)
        if lying_flat:
            ax.text(50, 24, f"{value:.0f}°", fontsize=13, fontweight="bold", ha="center")
            ax.text(
                50,
                44,
                "the upper arm lies along the horizontal",
                fontsize=9,
                color=INK_2,
                ha="center",
            )
        else:
            mid = arc(ax, (0, 0), (side, 0), elbow, 30)
            ax.text(
                *at_angle((0, 0), mid, 48),
                f"{value:.0f}°",
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
            )
        if name == "Halfway down":
            ax.plot([elbow[0], elbow[0]], [0, elbow[1]], color=AXIS, lw=1.2)
            ax.text(elbow[0] + 5, elbow[1] / 2, "|dy|", color=INK_2, fontsize=10)
            ax.text(elbow[0] / 2, -7, "|dx|", color=INK_2, fontsize=10, ha="center")
        point(ax, (0, 0), "shoulder", dx=-5 * side, dy=-8, ha="right" if side > 0 else "left")
        point(ax, elbow, "elbow", dx=6 * side, dy=12, ha="left" if side > 0 else "right")
        title(ax, name)
    fig.canvas.draw()  # apply the equal aspect, so the axes know their real height
    bottom = min(ax.get_position().y0 for ax in axes)
    fig.text(
        0.5,
        bottom - 0.08,
        "upper-arm angle = atan2(|dy|, |dx|)        dx, dy = elbow − shoulder"
        "        0° = parallel to the floor,  90° = pointing straight down",
        ha="center",
        color=INK_2,
        fontsize=10.5,
    )
    save(fig, "upper_arm_angle.png")


def fig_elbow_vs_upper_arm() -> None:
    cases = [
        (
            "A · not locked out",
            (0.0, 100.0),
            (60.0, 180.0),
            "upper arm is vertical,\nbut the elbow is bent",
        ),
        (
            "B · locked out, hands forward",
            (40.0, 90.0),
            (80.0, 180.0),
            "upper arm is tilted,\nbut the arm is straight",
        ),
        ("C · locked out", (0.0, 90.0), (0.0, 180.0), "both say straight"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4))
    for ax, (name, elbow, wrist, verdict) in zip(axes, cases, strict=True):
        diagram(ax, (-60, 150), (-18, 262))
        s, e, w = np.zeros(2), np.array(elbow), np.array(wrist)
        lm = landmarks_for({L.shoulder: tuple(s), L.elbow: elbow, L.wrist: wrist})
        upper = float(signals.upper_arm_angle_series(lm, L)[0])
        bend = float(signals.elbow_angle_series(lm, L)[0])

        ax.plot([-40, 140], [186, 186], color=AXIS, lw=1.2)
        ax.text(140, 196, "floor", color=MUTED, fontsize=9, ha="right")
        straight = e + (e - s) / np.linalg.norm(e - s) * np.linalg.norm(w - e)
        ax.plot(*zip(e, straight, strict=True), color=MUTED, lw=1.5, ls=(0, (5, 4)))
        ax.plot(*zip(s, e, strict=True), color=BLUE, lw=3)
        ax.plot(*zip(e, w, strict=True), color=ORANGE, lw=3)
        if bend < 179:
            mid = arc(ax, e, s - e, w - e, 15)
            ax.text(
                *at_angle(e, mid, 30),
                f"{bend:.0f}°",
                fontsize=10,
                color=INK_2,
                ha="center",
                va="center",
            )
        point(ax, s, "shoulder", dx=8, dy=-4)
        point(ax, e, "elbow", dx=-9, dy=4, ha="right")
        point(ax, w, "wrist", dx=8, dy=-6)
        ax.text(-55, 212, "upper-arm angle", fontsize=10.5, color=INK)
        ax.text(60, 212, f"{upper:.0f}°", fontsize=10.5, color=INK, ha="right")
        ax.text(-55, 227, "elbow angle", fontsize=10.5, color=INK)
        ax.text(60, 227, f"{bend:.0f}°", fontsize=10.5, color=INK, ha="right")
        ax.text(-55, 247, verdict.replace("\n", " "), fontsize=9.5, color=INK_2)
        title(ax, name)
    handles = [
        Line2D([], [], color=BLUE, lw=3, label="upper arm (shoulder → elbow)"),
        Line2D([], [], color=ORANGE, lw=3, label="forearm (elbow → wrist)"),
        Line2D(
            [],
            [],
            color=MUTED,
            lw=1.5,
            ls=(0, (5, 4)),
            label="where the forearm would point if the arm were straight",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=10,
    )
    save(fig, "elbow_vs_upper_arm.png")


def fig_torso_tilt() -> None:
    cases = [
        ("In a push-up", (0.0, 40.0), (170.0, 55.0)),
        ("Standing", (60.0, 0.0), (70.0, 170.0)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (name, shoulder, ankle) in zip(axes, cases, strict=True):
        diagram(ax, (-25, 215), (-25, 212))
        lm = landmarks_for({L.shoulder: shoulder, L.ankle: ankle})
        tilt = float(signals.torso_tilt_series(lm, L)[0])
        s = np.array(shoulder)
        ax.plot([s[0], s[0] + 180], [s[1], s[1]], color=MUTED, lw=1.3, ls=(0, (5, 4)))
        ax.text(s[0] + 180, s[1] - 6, "horizontal", color=MUTED, fontsize=9, ha="right")
        threshold = np.radians(35)
        end = s + 150 * np.array([np.cos(threshold), np.sin(threshold)])
        ax.plot(*zip(s, end, strict=True), color=INK_2, lw=1.3, ls=(0, (1, 3)))
        ax.text(*end + np.array([4, 4]), "35° threshold\n(max_tilt_deg)", color=INK_2, fontsize=9)
        ax.plot(*zip(shoulder, ankle, strict=True), color=BLUE, lw=3)
        mid = arc(ax, s, (1, 0), np.array(ankle) - s, 42)
        if tilt < 15:
            ax.text(
                s[0] + 62, s[1] - 9, f"{tilt:.0f}°", fontsize=13, fontweight="bold", ha="center"
            )
        else:
            ax.text(
                *at_angle(s, mid, 60),
                f"{tilt:.0f}°",
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
            )
        point(ax, shoulder, "shoulder", dx=-5, dy=-8, ha="right")
        point(ax, ankle, "ankle", dx=6, dy=12)
        inside = tilt < 35
        ax.text(
            -20,
            207,
            f"{tilt:.0f}° {'<' if inside else '≥'} 35°  →  "
            f"{'in position' if inside else 'not in position'}",
            fontsize=11,
            color=INK,
            fontweight="bold",
        )
        title(ax, name)
    fig.text(
        0.5,
        0.02,
        "torso tilt = atan2(|dy|, |dx|)        dx, dy = ankle − shoulder",
        ha="center",
        color=INK_2,
        fontsize=10.5,
    )
    save(fig, "torso_tilt.png")


def fig_hip_magnitude() -> None:
    cases = [
        ("Straight body", (50.0, 0.0)),
        ("Hip dropped 10 px", (50.0, 10.0)),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.9))
    for ax, (name, hip) in zip(axes, cases, strict=True):
        diagram(ax, (-14, 190), (-17, 22))
        s, h, a = np.array([0.0, 0.0]), np.array(hip), np.array([100.0, 0.0])
        lm = landmarks_for({L.shoulder: tuple(s), L.hip: hip, L.ankle: tuple(a)})
        deviation = float(signals.hip_deviation_series(lm, L)[0])
        at_hip = float(signals.angle(s, h, a))
        if hip[1] != 0:
            ax.plot(*zip(s, a, strict=True), color=INK_2, lw=1.3, ls=(0, (5, 4)))
            ax.text(100, -5, "straight shoulder–ankle line", color=INK_2, fontsize=9, ha="right")
            half = np.degrees(np.arctan(hip[1] / 50))
            for corner, towards in ((s, a - s), (a, s - a)):
                mid = arc(ax, corner, towards, h - corner, 24, color=ORANGE)
                ax.text(
                    *at_angle(corner, mid, 34),
                    f"{half:.1f}°",
                    fontsize=9.5,
                    color=INK_2,
                    ha="center",
                    va="center",
                )
        ax.plot(*zip(s, h, a, strict=True), color=BLUE, lw=3)
        up = (-1, -1e-6), (1, -1e-6)
        arc(ax, h, *(up if hip[1] == 0 else (s - h, a - h)), 7)
        angle_text = "180°" if hip[1] == 0 else f"{at_hip:.1f}°"
        point(ax, s, "shoulder", dx=-4, dy=6, ha="right")
        point(ax, h, f"hip   (angle here: {angle_text})", dx=0, dy=9, ha="center")
        point(ax, a, "ankle", dx=4, dy=6)
        if hip[1] == 0:
            summary = f"180° − 180° = {abs(deviation):.0f}°"
            detail = "a straight body: the hip is on the line"
        else:
            summary = f"180° − {at_hip:.1f}° = {abs(deviation):.1f}°"
            detail = f"= {half:.1f}° + {half:.1f}°: each half's tilt, added"
        ax.text(116, -2, summary, fontsize=12, fontweight="bold")
        ax.text(116, 8, detail, fontsize=9.5, color=INK_2)
        title(ax, name)
    save(fig, "hip_magnitude.png")


def fig_hip_sign() -> None:
    cases = [
        ("a · level line, hip below it", (50.0, 10.0), (100.0, 0.0)),
        ("b · level line, hip above it", (50.0, -10.0), (100.0, 0.0)),
        ("c · sloped line, hip below it", (50.0, 30.0), (100.0, 40.0)),
        (
            "d · sloped line, hip above it (though lower than the shoulder)",
            (50.0, 10.0),
            (100.0, 40.0),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6))
    for ax, (name, hip, ankle) in zip(axes.flat, cases, strict=True):
        diagram(ax, (-18, 215), (-32, 66))
        s, h, a = np.array([0.0, 0.0]), np.array(hip), np.array(ankle)
        lm = landmarks_for({L.shoulder: tuple(s), L.hip: hip, L.ankle: ankle})
        deviation = float(signals.hip_deviation_series(lm, L)[0])
        t = (h[0] - s[0]) / (a[0] - s[0])
        line_y = s[1] + t * (a[1] - s[1])
        slope = (a[1] - s[1]) / (a[0] - s[0])

        xs = np.array([-12.0, 112.0])
        ax.plot(xs, slope * xs, color=INK_2, lw=1.3, ls=(0, (5, 4)))
        ax.plot(*zip(s, h, a, strict=True), color=AXIS, lw=2)
        ax.plot([h[0], h[0]], [line_y, h[1]], color=MUTED, lw=1)
        ax.plot(h[0], line_y, "o", ms=10, mfc=SURFACE, mec=INK_2, mew=2, zorder=6)
        colour = BLUE if deviation > 0 else RED
        ax.plot(*h, "o", ms=12, color=colour, mec=SURFACE, mew=2, zorder=7)
        below = h[1] > line_y
        ax.text(h[0], h[1] + (10 if below else -7), "hip", fontsize=9.5, color=INK_2, ha="center")
        ax.text(
            h[0] + (7 if below else -7),
            line_y + (-6 if below else 11),
            "line",
            fontsize=9.5,
            color=INK_2,
            ha="left" if below else "right",
        )
        point(ax, s, "shoulder", dx=-3, dy=-5, ha="right")
        point(ax, a, "ankle", dx=4, dy=-4)
        verdict = "below → sag" if below else "above → pike"
        relation = ">" if below else "<"
        ax.text(
            128,
            18,
            f"hip y = {h[1]:.0f}  {relation}  line y = {line_y:.0f}",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.text(128, 29, f"hip is {verdict}", fontsize=10, color=INK_2)
        ax.text(128, 40, f"result: {deviation:+.1f}°", fontsize=11, color=INK)
        if name.startswith("d"):
            ax.text(
                -15,
                60,
                "The hip is lower than the shoulder (10 > 0), but higher than the "
                "line (10 < 20).\nOnly the line matters.",
                fontsize=9.5,
                color=INK_2,
            )
        title(ax, name)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=11,
            color=BLUE,
            label="hip below the line → positive → sag",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=11,
            color=RED,
            label="hip above the line → negative → pike",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=10,
            mfc=SURFACE,
            mec=INK_2,
            mew=2,
            label="the line's height at the hip's x",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=10,
    )
    save(fig, "hip_sign.png")


def fig_hip_mirror() -> None:
    cases = [
        ("Facing right", (0.0, 0.0), (100.0, 0.0)),
        ("Mirrored: the direction flips, the line doesn't", (100.0, 0.0), (0.0, 0.0)),
    ]
    hip = (50.0, 10.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, (name, shoulder, ankle) in zip(axes, cases, strict=True):
        diagram(ax, (-15, 115), (-22, 42))
        s, h, a = np.array(shoulder), np.array(hip), np.array(ankle)
        lm = landmarks_for({L.shoulder: shoulder, L.hip: hip, L.ankle: ankle})
        deviation = float(signals.hip_deviation_series(lm, L)[0])
        t = (h[0] - s[0]) / (a[0] - s[0])
        cross = (a - s)[0] * (h - s)[1] - (a - s)[1] * (h - s)[0]

        ax.plot([-10, 110], [0, 0], color=INK_2, lw=1.3, ls=(0, (5, 4)))
        ax.annotate(
            "",
            xy=a + (s - a) * 0.08,
            xytext=s + (a - s) * 0.08,
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4),
        )
        ax.text(50, -8, "direction: shoulder → ankle", color=MUTED, fontsize=9, ha="center")
        ax.plot(*zip(s, h, a, strict=True), color=AXIS, lw=2)
        ax.plot(h[0], 0, "o", ms=10, mfc=SURFACE, mec=INK_2, mew=2, zorder=6)
        ax.plot(*h, "o", ms=12, color=BLUE, mec=SURFACE, mew=2, zorder=7)
        point(ax, s, "shoulder", dy=-6, ha="center")
        point(ax, a, "ankle", dy=-6, ha="center")
        ax.text(
            -12,
            26,
            f"t = {t:.1f}    line y = 0    hip y = 10    →    {deviation:+.1f}°",
            fontsize=10.5,
            color=INK,
        )
        ax.text(
            -12,
            37,
            f"a cross product would give {cross:+.0f}"
            + ("  — the sign flipped" if cross < 0 else ""),
            fontsize=9.5,
            color=INK_2,
        )
        title(ax, name)
    save(fig, "hip_mirror.png")


def fig_runs() -> None:
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    padded = np.r_[0, mask.astype(int), 0]
    diff = np.diff(padded)
    runs = signals.runs(mask)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(-8.2, 10.2)
    ax.set_ylim(-6.3, 1.3)
    ax.axis("off")

    def cell(
        col: float,
        row: float,
        text: str,
        fill: str = SURFACE,
        edge: str = AXIS,
        lw: float = 1.0,
        ls: str = "-",
    ) -> None:
        ax.add_patch(
            Rectangle((col - 0.44, row - 0.38), 0.88, 0.76, fc=fill, ec=edge, lw=lw, ls=ls)
        )
        ax.text(col, row, text, ha="center", va="center", fontsize=11, color=INK)

    def row_label(row: float, text: str) -> None:
        ax.text(-1.75, row, text, ha="right", va="center", fontsize=10.5, color=INK_2)

    for col in range(len(mask)):
        ax.text(col, 0.85, str(col), ha="center", color=MUTED, fontsize=9)
    ax.text(-1.75, 0.85, "index", ha="right", color=MUTED, fontsize=9)

    row_label(0, "mask")
    for col, value in enumerate(mask):
        cell(col, 0, str(int(value)), fill=BLUE_TINT if value else SURFACE)

    row_label(-1.3, "padded with a 0 at each end")
    for col, value in zip(range(-1, len(mask) + 1), padded, strict=True):
        added = col in (-1, len(mask))
        cell(
            col,
            -1.3,
            str(value),
            fill=BLUE_TINT if value else SURFACE,
            ls=(0, (3, 2)) if added else "-",
        )

    row_label(-2.6, "np.diff  (next − this)")
    for col, value in enumerate(diff):
        edge, lw = (BLUE, 2.5) if value == 1 else (ORANGE, 2.5) if value == -1 else (AXIS, 1)
        cell(col, -2.6, f"{value:+d}".replace("-", "−") if value else "0", edge=edge, lw=lw)

    row_label(-3.9, "runs  (start, stop)")
    for start, stop in runs:
        ax.add_patch(
            Rectangle((start - 0.44, -4.17), stop - start - 0.12, 0.54, fc=BLUE, ec=SURFACE, lw=2)
        )
        ax.text(
            (start + stop - 1) / 2,
            -4.3,
            f"({start}, {stop})",
            ha="center",
            va="top",
            fontsize=10.5,
            color=INK,
        )

    ax.text(
        -8.1,
        -6.25,
        "+1 (blue ring) is where a run starts; −1 (orange ring) is the first "
        "index after it ends.\nThe edges alternate start, stop, start, stop, so "
        "edges[::2] are the starts and edges[1::2] the stops.\nThe stop is exclusive: "
        "(1, 4) means indices 1, 2, 3, which is exactly the slice signal[1:4].",
        fontsize=9.5,
        color=INK_2,
        va="bottom",
    )
    save(fig, "runs.png")


def fig_in_position() -> None:
    fps = 30
    rng = np.random.default_rng(1)
    parts = [
        np.full(36, 82.0),  # standing
        np.full(5, 12.0),  # a stray horizontal moment while setting up
        np.full(22, 78.0),  # upright again
        np.linspace(78, 8, 9),  # getting down
        6 + 3 * np.sin(np.linspace(0, 10 * np.pi, 300)),  # ten seconds of push-ups
        np.linspace(8, 80, 12),  # getting up
        np.full(36, 82.0),  # standing
    ]
    tilt = np.concatenate(parts)
    tilt = tilt + rng.normal(0, 1.2, tilt.size)
    t = np.arange(tilt.size) / fps
    lo, hi = signals.in_position_window(tilt, max_tilt_deg=35.0)
    stray = [r for r in signals.runs(tilt < 35.0) if r != (lo, hi)]

    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvspan(t[lo], t[hi - 1], color=AQUA, alpha=0.14, lw=0)
    for start, stop in stray:
        ax.axvspan(t[start], t[stop - 1], color=MUTED, alpha=0.25, lw=0)
    ax.axhline(35, color=MUTED, lw=1.3, ls=(0, (5, 4)))
    ax.plot(t, tilt, color=BLUE, lw=2)
    ax.text(t[-1], 37.5, "35°  (max_tilt_deg)", color=INK_2, fontsize=9.5, ha="right")
    ax.text(
        (t[lo] + t[hi - 1]) / 2,
        55,
        f"kept: the longest run below 35°\nframes {lo}–{hi}  ({(hi - lo) / fps:.1f} s)",
        ha="center",
        fontsize=10.5,
        color=INK,
    )
    for start, stop in stray:
        ax.annotate(
            f"ignored\n({stop - start} frames)",
            xy=(t[start], 14),
            xytext=(0.12, 48),
            fontsize=9.5,
            color=INK_2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
        )
    ax.text(0.1, 86, "standing", fontsize=9.5, color=INK_2)
    ax.text(t[-1] - 0.1, 86, "standing", fontsize=9.5, color=INK_2, ha="right")
    ax.set_ylim(0, 96)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("torso tilt (°)")
    title(ax, "in_position_window: keep the longest stretch where the body is horizontal")
    save(fig, "in_position_window.png")


def fig_interpolate() -> None:
    x = np.arange(90)
    signal = 60 + 25 * np.sin(2 * np.pi * x / 45)
    gaps = {"start": (0, 4), "short": (24, 27), "long": (52, 66)}
    for start, stop in gaps.values():
        signal[start:stop] = np.nan
    max_gap = 5
    filled = signals.interpolate_gaps(signal, max_gap=max_gap)
    new = np.isnan(signal) & ~np.isnan(filled)

    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for start, stop in gaps.values():
        ax.axvspan(start - 0.5, stop - 0.5, color=MUTED, alpha=0.18, lw=0)
    ax.plot(x, signal, color=BLUE, lw=2, label="known values")
    ax.plot(
        x[new],
        filled[new],
        "o",
        ms=8,
        color=ORANGE,
        mec=SURFACE,
        mew=2,
        label="filled by interpolation",
    )
    notes = {
        "start": "at the start:\nleft as NaN\n(nothing before it)",
        "short": f"3 frames ≤ max_gap ({max_gap}):\nfilled",
        "long": "14 frames > max_gap:\nleft as NaN\n(don't invent data)",
    }
    for key, (start, stop) in gaps.items():
        centre = max((start + stop - 1) / 2, 3.5)
        ax.text(centre, 110, notes[key], ha="center", va="top", fontsize=9.5, color=INK_2)
    ax.set_ylim(22, 112)
    ax.set_yticks(range(30, 100, 10))
    ax.set_xlim(-1, 90)
    ax.set_xlabel("frame")
    ax.set_ylabel("signal value")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    title(ax, "interpolate_gaps: fill short interior gaps, leave the rest as NaN")
    save(fig, "interpolate_gaps.png")


def fig_smooth() -> dict[str, float]:
    fps = 30
    t = np.arange(12 * fps) / fps
    clean = 88 - 83 * np.exp(-((((t % 2) - 1) / 0.22) ** 2))  # push-up-like depth signal
    noisy = clean + np.random.default_rng(0).normal(0, 3, t.size)
    smoothed = signals.smooth(noisy, fps=fps, window_s=0.4, polyorder=2)
    window = int(fps * 0.4) | 1
    moving = np.convolve(noisy, np.ones(window) / window, mode="same")

    zoom = (t >= 2) & (t <= 6)
    dips = (t > 2.5) & (t < 3.5)
    lows = {
        "true": float(clean[dips].min()),
        "savgol": float(smoothed[dips].min()),
        "moving": float(moving[dips].min()),
    }

    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(11, 6.6), height_ratios=[3, 1.25])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.plot(t[zoom], noisy[zoom], color=AXIS, lw=1, label="raw signal (noisy)")
    ax.plot(t[zoom], clean[zoom], color=INK, lw=1.2, label="true shape")
    ax.plot(
        t[zoom],
        moving[zoom],
        color=ORANGE,
        lw=2,
        label=f"moving average, same {window}-frame window",
    )
    ax.plot(t[zoom], smoothed[zoom], color=BLUE, lw=2, label="Savitzky–Golay (smooth)")
    ax.axhline(5, color=MUTED, lw=1.3, ls=(0, (5, 4)))
    ax.text(5.97, 7, "shallow threshold: 5°", color=INK_2, fontsize=9.5, ha="right")
    ax.annotate(
        f"moving average bottoms out at {lows['moving']:.0f}°",
        xy=(3.0, lows["moving"]),
        xytext=(3.25, 30),
        fontsize=9.5,
        color=INK_2,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    ax.annotate(
        f"Savitzky–Golay: {lows['savgol']:.0f}°   (true: {lows['true']:.0f}°)",
        xy=(3.0, lows["savgol"]),
        xytext=(3.25, 18),
        fontsize=9.5,
        color=INK_2,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=1),
    )
    ax.set_xlim(2, 6)
    ax.set_ylim(-4, 124)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("upper-arm angle (°)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=4)
    title(ax, "Why Savitzky–Golay: it keeps the depth of each dip")

    frames = np.arange(40)
    wave = 50 + 20 * np.sin(frames / 5)
    wave[20] = np.nan
    whole = savgol_filter(wave, window, 2)
    per_run = signals.smooth(wave, fps=fps, window_s=0.4, polyorder=2)
    ax_n.set_xlim(-1, 50)
    ax_n.set_ylim(-0.9, 1.7)
    ax_n.axis("off")
    for row, (series, label) in enumerate(
        [(per_run, "smooth(), one run at a time"), (whole, "savgol on the whole array")]
    ):
        lost = int(np.isnan(series).sum())
        for f in frames:
            gone = np.isnan(series[f])
            ax_n.add_patch(
                Rectangle((f - 0.42, row - 0.32), 0.84, 0.64, lw=0, fc=INK_2 if gone else BLUE_TINT)
            )
        ax_n.text(
            40.5,
            row,
            f"{label}\n{lost} frame{'s' if lost != 1 else ''} lost",
            va="center",
            fontsize=9.5,
            color=INK,
        )
    ax_n.text(
        -1,
        1.62,
        "One missing frame (dark = NaN): smoothed per valid run it stays one "
        "frame; smoothed across it, it blanks the whole window.",
        fontsize=9.5,
        color=INK_2,
        va="top",
    )
    save(fig, "smoothing.png")
    return lows


def main() -> None:
    fig_coordinates()
    fig_angle()
    fig_upper_arm()
    fig_elbow_vs_upper_arm()
    fig_torso_tilt()
    fig_hip_magnitude()
    fig_hip_sign()
    fig_hip_mirror()
    fig_runs()
    fig_in_position()
    fig_interpolate()
    lows = fig_smooth()
    print("dip minima:", {k: round(v, 1) for k, v in lows.items()})


if __name__ == "__main__":
    main()
