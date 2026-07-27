"""
src/analytics/xt_model.py

Expected Threat (xT) model — a zone-based possession-value model,
following the approach popularized by Karun Singh (2018), building
on Sarah Rudd's original 2011 concept.

Library code only: pure functions for building the zone-level
shot/move statistics and solving the iterative Markov-chain fixed
point. The training/data-pulling pipeline lives in
scripts/build_xt_grid.py, not here.
"""

import numpy as np
import pandas as pd

from src.analytics.xg_model import prepare_shot_features, predict_xg

PITCH_LENGTH, PITCH_WIDTH = 120, 80
N_COLS, N_ROWS = 16, 12

FULL_ZONE_INDEX = pd.MultiIndex.from_product(
    [range(N_COLS), range(N_ROWS)], names=["col", "row"]
)


def _zone_index(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map pitch coordinates to (col, row) grid cell indices."""
    col = np.clip((np.asarray(x) / PITCH_LENGTH * N_COLS).astype(int), 0, N_COLS - 1)
    row = np.clip((np.asarray(y) / PITCH_WIDTH * N_ROWS).astype(int), 0, N_ROWS - 1)
    return col, row


def build_move_and_shot_data(all_events: pd.DataFrame, xg_model, include_turnovers: bool = False) -> dict:
    """
    From raw events across many matches, compute the zone-level statistics
    needed to solve for xT.

    If include_turnovers=False (default): only shots and completed moves
    are counted, so shot_prob + move_prob = 1 per zone. Turnovers are
    implicitly excluded from the denominator entirely.

    If include_turnovers=True: failed passes/dispossessions are also
    counted in the denominator, so shot_prob + move_prob + turnover_prob = 1
    per zone, with turnover_value fixed at 0 (losing the ball ends this
    possession's attacking value — the simplest defensible assumption;
    see README for the documented next step of a negative, opponent-xT-based
    turnover value).

    Shot features/values are computed via the shared xG feature-extraction
    function (src/analytics/xg_model.py), not duplicated here, so xG and xT
    always agree on what a shot's features are. `xg_model` must be the
    artifact dict saved by scripts/train_xg_model.py
    ({"model":..., "feature_names":..., "penalty_xg":...}).
    """
    all_actions = (
        all_events[all_events["event_type"].isin(["Pass", "Carry", "Shot"])]
        .dropna(subset=["x", "y"])
        .copy()
    )

    is_shot = all_actions["event_type"] == "Shot"
    is_completed_move = (~is_shot) & (all_actions["outcome"].isna())
    is_turnover = (~is_shot) & (~is_completed_move)  # failed pass, dispossession, etc.

    all_actions["col"], all_actions["row"] = _zone_index(
        all_actions["x"].to_numpy(), all_actions["y"].to_numpy()
    )

    denom_mask = pd.Series(True, index=all_actions.index) if include_turnovers else (is_shot | is_completed_move)
    actions = all_actions[denom_mask].copy()

    total_per_zone = (
        actions.groupby(["col", "row"]).size().reindex(FULL_ZONE_INDEX, fill_value=0)
    )

    # --- shots: via the shared xG feature-extraction function ---
    shots = prepare_shot_features(all_events, include_penalties=False)
    shots["col"], shots["row"] = _zone_index(shots["x"].to_numpy(), shots["y"].to_numpy())
    shots["xg"] = predict_xg(xg_model, shots)

    shots_per_zone = shots.groupby(["col", "row"]).size().reindex(FULL_ZONE_INDEX, fill_value=0)
    shot_value = shots.groupby(["col", "row"])["xg"].mean().reindex(FULL_ZONE_INDEX, fill_value=0)

    # --- moves: destination zone + transition counts ---
    moves = all_actions[is_completed_move].copy()
    moves["end_col"], moves["end_row"] = _zone_index(
        moves["end_x"].fillna(moves["x"]).to_numpy(),
        moves["end_y"].fillna(moves["y"]).to_numpy(),
    )
    move_counts = moves.groupby(["col", "row"]).size().reindex(FULL_ZONE_INDEX, fill_value=0)
    transition_counts = moves.groupby(["col", "row", "end_col", "end_row"]).size()

    shot_prob = (shots_per_zone / total_per_zone).fillna(0)
    move_prob = (move_counts / total_per_zone).fillna(0)

    result = {
        "shot_prob": shot_prob,
        "shot_value": shot_value,
        "move_prob": move_prob,
        "move_counts": move_counts,
        "transition_counts": transition_counts,
    }

    if include_turnovers:
        turnovers = all_actions[is_turnover].copy()
        turnover_counts = turnovers.groupby(["col", "row"]).size().reindex(FULL_ZONE_INDEX, fill_value=0)
        turnover_prob = (turnover_counts / total_per_zone).fillna(0)
        result["turnover_prob"] = turnover_prob
        result["turnover_value"] = 0.0  # simplest version — see README future work

    return result


def compute_xt_grid(
    data: dict,
    max_iterations: int = 500,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """
    Solve for the xT value of every zone via the iterative Markov-chain
    fixed point (Singh, 2018): start all zones at 0, then repeatedly
    recompute each zone's value using the previous iteration's values
    for reachable destination zones, stopping once the grid stabilizes
    (max change between iterations drops below `tolerance`).

    xT[zone] = shot_prob[zone] * shot_value[zone]
             + move_prob[zone] * sum_over_destinations( P(dest | zone) * xT[dest] )
             + turnover_prob[zone] * turnover_value   [only if include_turnovers was used]
    """
    shot_prob = data["shot_prob"]
    shot_value = data["shot_value"]
    move_prob = data["move_prob"]
    move_counts = data["move_counts"]
    transition_counts = data["transition_counts"]
    turnover_prob = data.get("turnover_prob")  # None if include_turnovers=False
    turnover_value = data.get("turnover_value", 0.0)

    xt = np.zeros((N_COLS, N_ROWS))
    max_change = float("inf")

    for iteration in range(max_iterations):
        xt_new = np.zeros((N_COLS, N_ROWS))

        for col in range(N_COLS):
            for row in range(N_ROWS):
                sp = shot_prob.get((col, row), 0.0)
                sv = shot_value.get((col, row), 0.0)
                mp = move_prob.get((col, row), 0.0)
                tp = turnover_prob.get((col, row), 0.0) if turnover_prob is not None else 0.0
                total_moves = move_counts.get((col, row), 0)

                move_value = 0.0
                if total_moves > 0:
                    try:
                        dests = transition_counts.loc[(col, row)]
                    except KeyError:
                        dests = None
                    if dests is not None:
                        for (end_col, end_row), count in dests.items():
                            prob = count / total_moves
                            move_value += prob * xt[end_col, end_row]

                xt_new[col, row] = sp * sv + mp * move_value + tp * turnover_value

        max_change = np.abs(xt_new - xt).max()
        xt = xt_new

        if (iteration + 1) % 10 == 0 or max_change < tolerance:
            print(f"  iteration {iteration + 1} — max change: {max_change:.6f}")

        if max_change < tolerance:
            print(f"Converged after {iteration + 1} iterations (tolerance={tolerance}).")
            return xt

    print(f"Did not reach tolerance after {max_iterations} iterations "
          f"(final max change: {max_change:.6f}). Consider raising max_iterations "
          f"or loosening tolerance.")
    return xt