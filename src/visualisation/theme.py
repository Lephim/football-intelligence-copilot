"""
src/visualisation/theme.py

Shared visual theme for all pitch visualisations — one place to control
colors, fonts, and styling so every chart in the project looks consistent.
"""

import matplotlib.pyplot as plt

# --- palette ---
BG_COLOR = "#0d1117"          # dark charcoal-navy, softer than pure black
LINE_COLOR = "#e8e8e8"        # pitch lines, slightly off-white
TEXT_COLOR = "#f0f0f0"

ACCENT_GOLD = "#f5b942"       # primary accent — nodes, goals, highlights
ACCENT_TEAL = "#4ecdc4"       # secondary accent — edges, non-goal shots
ACCENT_CORAL = "#ff6b6b"      # optional tertiary accent, if a third category is ever needed

# a warm, no-purple sequential colormap for heatmaps (replaces "inferno")
HEATMAP_CMAP = "YlOrRd"       # yellow -> orange -> red; intuitive "low to high" without purple/black murk

# diverging colormap for difference plots — kept as RdBu-like but can swap if desired
DIVERGING_CMAP = "RdYlBu_r"


def apply_theme():
    """Call once, e.g. at the top of a notebook or in each plotting module's import."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Verdana", "DejaVu Sans", "Arial"],  # falls back gracefully if Verdana unavailable
        "font.weight": "medium",
        "axes.titleweight": "bold",
        "text.color": TEXT_COLOR,
    })