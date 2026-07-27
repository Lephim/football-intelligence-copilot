import joblib
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from src.analytics.xg_model import prepare_shot_features, predict_xg
from pathlib import Path
from src.visualisation.theme import BG_COLOR, LINE_COLOR, ACCENT_GOLD, ACCENT_TEAL, TEXT_COLOR, apply_theme

apply_theme()

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # shot_map.py -> visualisation -> src -> project root
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xg_model.pkl"

def build_shot_map_data(events: pd.DataFrame, team: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """Prepare shots for one team with predicted xG attached."""
    shots = prepare_shot_features(events)
    shots = shots[shots["team"] == team].copy()

    model = joblib.load(model_path)
    shots["xg"] = predict_xg(model, shots)

    return shots


def plot_shot_map(shots, team, title=""):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG_COLOR, line_color=LINE_COLOR, half=True)
    fig, ax = pitch.draw(figsize=(8, 8))

    goals = shots[shots["is_goal"] == 1]
    misses = shots[shots["is_goal"] == 0]

    pitch.scatter(
        misses["x"], misses["y"], s=misses["xg"] * 1500 + 100,
        facecolors="none", edgecolors=ACCENT_TEAL, linewidth=1.5, alpha=0.75, ax=ax, zorder=1,
    )
    pitch.scatter(
        goals["x"], goals["y"], s=goals["xg"] * 1500 + 100,
        facecolors=ACCENT_GOLD, edgecolors=BG_COLOR, linewidth=1.2, alpha=0.95, ax=ax, zorder=2,
    )

    total_xg = shots["xg"].sum()
    actual_goals = shots["is_goal"].sum()
    ax.set_title(f"{title}\nTotal xG: {total_xg:.2f}  |  Actual goals: {actual_goals}",
                 color=TEXT_COLOR, fontsize=12)
    fig.patch.set_facecolor(BG_COLOR)
    return fig


def team_xg_summary(all_shots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total xG and actual goals per team — useful for a match-level summary."""
    return (
        all_shots.groupby("team")
        .agg(total_xg=("xg", "sum"), actual_goals=("is_goal", "sum"), shots=("is_goal", "count"))
        .reset_index()
        .sort_values("total_xg", ascending=False)
    )