"""
src/visualisation/xt_heatmap.py

Plotting functions for xT grids: raw heatmap, absolute difference between
two versions, and relative (%) change between two versions. All three use
the shared theme (src/visualisation/theme.py) for consistent colors/fonts.
"""

import numpy as np
from mplsoccer import Pitch

from src.visualisation.theme import (
    BG_COLOR, LINE_COLOR, TEXT_COLOR, HEATMAP_CMAP, DIVERGING_CMAP, apply_theme,
)

apply_theme()


def _style_colorbar(cbar):
    """Shared helper: make colorbar text visible against the dark theme background."""
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)


def plot_xt_heatmap(xt_grid: np.ndarray, title: str = "Expected Threat (xT)", vmin=None, vmax=None):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG_COLOR, line_color=LINE_COLOR)
    fig, ax = pitch.draw(figsize=(10, 7))

    pcm = ax.imshow(
        xt_grid.T, extent=(0, 120, 80, 0), cmap=HEATMAP_CMAP, alpha=0.9, aspect="auto",
        vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(pcm, ax=ax, label="xT value", shrink=0.7)
    _style_colorbar(cbar)

    ax.set_title(title, color=TEXT_COLOR, fontsize=13)
    fig.patch.set_facecolor(BG_COLOR)
    return fig


def plot_xt_diff(xt_a: np.ndarray, xt_b: np.ndarray, label_a: str, label_b: str):
    """Diverging heatmap of (xt_b - xt_a) — highlights where the two versions disagree most."""
    diff = xt_b - xt_a
    vmax = np.abs(diff).max()

    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG_COLOR, line_color=LINE_COLOR)
    fig, ax = pitch.draw(figsize=(10, 7))

    pcm = ax.imshow(
        diff.T, extent=(0, 120, 80, 0), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
        alpha=0.9, aspect="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, label=f"xT difference ({label_b} − {label_a})", shrink=0.7)
    _style_colorbar(cbar)

    ax.set_title(f"Where {label_b} values zones higher or lower than {label_a}",
                 color=TEXT_COLOR, fontsize=11)
    fig.patch.set_facecolor(BG_COLOR)
    return fig


def plot_xt_pct_change(xt_a: np.ndarray, xt_b: np.ndarray, label_a: str, label_b: str):
    """Percent change from xt_a to xt_b — safe against divide-by-zero for near-empty zones."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = np.where(xt_a > 1e-6, (xt_b - xt_a) / xt_a * 100, np.nan)

    vmax = np.nanmax(np.abs(pct_change))

    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG_COLOR, line_color=LINE_COLOR)
    fig, ax = pitch.draw(figsize=(10, 7))

    pcm = ax.imshow(
        pct_change.T, extent=(0, 120, 80, 0), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
        alpha=0.9, aspect="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, label=f"% change ({label_b} vs {label_a})", shrink=0.7)
    _style_colorbar(cbar)

    ax.set_title("Relative change in zone value", color=TEXT_COLOR, fontsize=11)
    fig.patch.set_facecolor(BG_COLOR)
    return fig