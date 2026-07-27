"""
src/analytics/xg_model.py

xG model: feature extraction + inference.
Training/evaluation pipeline lives in scripts/train_xg_model.py, not here.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

GOAL_X = 120
GOAL_Y_TOP = 44
GOAL_Y_BOTTOM = 36

FEATURE_COLUMNS = ["distance", "angle", "is_header", "is_free_kick", "under_pressure", "first_time"]


def _shot_distance(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (40 - y) ** 2)


def _shot_angle(x, y):
    top_dx, top_dy = GOAL_X - x, GOAL_Y_TOP - y
    bottom_dx, bottom_dy = GOAL_X - x, GOAL_Y_BOTTOM - y
    angle_top = np.arctan2(top_dy, top_dx)
    angle_bottom = np.arctan2(bottom_dy, bottom_dx)
    return np.abs(angle_top - angle_bottom)


def _extract_shot_field(row: dict, flat_key: str, default=None):
    """StatsBomb events (via statsbombpy) are flattened to top-level keys
    like 'shot_type', 'shot_technique' — not nested sub-dicts."""
    if not isinstance(row, dict):
        return default
    return row.get(flat_key, default)


def prepare_shot_features(events: pd.DataFrame, include_penalties: bool = False) -> pd.DataFrame:
    """
    Extract shots with geometric + contextual features and a goal/no-goal label.

    Penalties are excluded by default: their distance/angle are fixed and
    uninformative (they're not really a geometry problem), so mixing them
    into a distance/angle regression distorts the fit. Their conversion
    rate is better modeled as a separate empirical constant — see
    PENALTY_XG handling in train_xg_model / predict_xg.
    """
    shots = events[events["event_type"] == "Shot"].dropna(subset=["x", "y"]).copy()

    shots["distance"] = _shot_distance(shots["x"], shots["y"])
    shots["angle"] = _shot_angle(shots["x"], shots["y"])
    shots["is_goal"] = (shots["outcome"] == "Goal").astype(int)

    shot_type = shots["raw"].apply(lambda r: _extract_shot_field(r, "shot_type", default="Open Play"))
    technique = shots["raw"].apply(lambda r: _extract_shot_field(r, "shot_technique", default="Normal"))
    body_part = shots["raw"].apply(lambda r: _extract_shot_field(r, "shot_body_part", default="Right Foot"))
    first_time = shots["raw"].apply(lambda r: bool(_extract_shot_field(r, "shot_first_time", default=False)))
    under_pressure = shots["raw"].apply(lambda r: bool(_extract_shot_field(r, "under_pressure", default=False)))

    shots["shot_type"] = shot_type
    shots["is_penalty"] = (shot_type == "Penalty").astype(int)
    shots["is_free_kick"] = (shot_type == "Free Kick").astype(int)
    shots["is_header"] = (body_part == "Head").astype(int)
    shots["first_time"] = first_time.astype(int)
    shots["under_pressure"] = under_pressure.astype(int)
    shots["technique"] = technique

    if not include_penalties:
        shots = shots[shots["is_penalty"] == 0].copy()

    return shots


def train_xg_model(train_shots: pd.DataFrame) -> dict:
    """
    Fit the logistic regression on train_shots (already penalty-excluded,
    since penalties shouldn't be in the geometric model — see
    prepare_shot_features). Returns a bundled artifact: the fitted model
    PLUS its feature names, so inference can never silently mismatch
    columns if FEATURE_COLUMNS changes later.
    """
    X = train_shots[FEATURE_COLUMNS]
    y = train_shots["is_goal"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return {"model": model, "feature_names": FEATURE_COLUMNS}


def compute_penalty_xg(all_shots_with_penalties: pd.DataFrame) -> float:
    """
    Penalties aren't modeled geometrically — their scoring rate is just an
    empirical constant. Pass a shots DataFrame built with
    include_penalties=True to compute it from real data.
    """
    penalties = all_shots_with_penalties[all_shots_with_penalties["is_penalty"] == 1]
    if len(penalties) == 0:
        return 0.76  # reasonable published-literature fallback if no penalties in sample
    return penalties["is_goal"].mean()


def predict_xg(artifact: dict, shots: pd.DataFrame) -> np.ndarray:
    """
    Predict xG for each shot. Penalty rows (is_penalty == 1, if present)
    get the artifact's stored empirical penalty rate instead of a model
    prediction, since the geometric model was never trained on penalties.
    """
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    penalty_xg = artifact.get("penalty_xg", 0.76)  # fallback for older artifacts without this key

    X = shots[feature_names]
    preds = model.predict_proba(X)[:, 1]

    if "is_penalty" in shots.columns:
        preds = np.where(shots["is_penalty"].to_numpy() == 1, penalty_xg, preds)

    return preds