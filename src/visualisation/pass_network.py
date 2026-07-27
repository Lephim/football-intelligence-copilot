import pandas as pd
from mplsoccer import Pitch
from src.visualisation.theme import BG_COLOR, LINE_COLOR, ACCENT_GOLD, ACCENT_TEAL, TEXT_COLOR, apply_theme

apply_theme()


def build_pass_network(events: pd.DataFrame, team: str, only_completed: bool = True):
    """
    Compute node positions (avg location per player) and edge weights
    (pass counts between player pairs) for a team's passing network.
    """
    passes = events[(events["event_type"] == "Pass") & (events["team"] == team)].copy()

    if only_completed:
        # StatsBomb marks failed passes with a non-null outcome
        # (e.g. "Incomplete", "Out", "Pass Offside"); completed passes have outcome == None
        passes = passes[passes["outcome"].isna()]

    passes = passes.dropna(subset=["x", "y", "recipient"])

    # node positions: average location of each player's passes
    node_positions = passes.groupby("player")[["x", "y"]].mean()

    # edge weights: count of pass combinations, direction-aware (A->B distinct from B->A)
    edge_counts = (
        passes.groupby(["player", "recipient"])
        .size()
        .reset_index(name="pass_count")
    )

    return node_positions, edge_counts


def plot_pass_network(node_positions, edge_counts, title=""):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG_COLOR, line_color=LINE_COLOR)
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in edge_counts.iterrows():
        passer, recipient, count = row["player"], row["recipient"], row["pass_count"]
        if passer not in node_positions.index or recipient not in node_positions.index:
            continue
        x1, y1 = node_positions.loc[passer, ["x", "y"]]
        x2, y2 = node_positions.loc[recipient, ["x", "y"]]
        pitch.lines(x1, y1, x2, y2, lw=count * 0.5, color=ACCENT_TEAL, alpha=0.6, ax=ax, zorder=1)

    pitch.scatter(
        node_positions["x"], node_positions["y"],
        s=600, color=ACCENT_GOLD, edgecolors=BG_COLOR, linewidth=1.5, ax=ax, zorder=2
    )

    for player, row in node_positions.iterrows():
        ax.annotate(
            player.split()[-1], (row["x"], row["y"]),
            ha="center", va="center", fontsize=8, color=BG_COLOR, fontweight="bold", zorder=3
        )

    ax.set_title(title, color=TEXT_COLOR, fontsize=14)
    fig.patch.set_facecolor(BG_COLOR)
    return fig