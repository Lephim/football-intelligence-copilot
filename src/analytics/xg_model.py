import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

GOAL_X = 120
GOAL_Y_TOP = 44  # StatsBomb goal is 8 yards wide, centered on y=40 -> posts at 36/44
GOAL_Y_BOTTOM = 36


def _shot_angle(x, y):
    """Angle (radians) subtended by the goal mouth, as seen from (x, y)."""
    # vectors from shot location to each goalpost
    top_dx, top_dy = GOAL_X - x, GOAL_Y_TOP - y
    bottom_dx, bottom_dy = GOAL_X - x, GOAL_Y_BOTTOM - y

    angle_top = np.arctan2(top_dy, top_dx)
    angle_bottom = np.arctan2(bottom_dy, bottom_dx)

    return np.abs(angle_top - angle_bottom)


def _shot_distance(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (40 - y) ** 2)


def prepare_shot_features(events: pd.DataFrame) -> pd.DataFrame:
    """Extract shots with distance, angle, and goal/no-goal label."""
    shots = events[events["event_type"] == "Shot"].dropna(subset=["x", "y"]).copy()

    shots["distance"] = _shot_distance(shots["x"], shots["y"])
    shots["angle"] = _shot_angle(shots["x"], shots["y"])
    shots["is_goal"] = (shots["outcome"] == "Goal").astype(int)

    return shots


def train_xg_model(shots: pd.DataFrame) -> LogisticRegression:
    """Fit a simple 2-feature logistic regression xG model."""
    X = shots[["distance", "angle"]]
    y = shots["is_goal"]

    model = LogisticRegression()
    model.fit(X, y)
    return model


def predict_xg(model: LogisticRegression, shots: pd.DataFrame) -> pd.Series:
    """Return predicted goal probability (xG) for each shot."""
    X = shots[["distance", "angle"]]
    return model.predict_proba(X)[:, 1]