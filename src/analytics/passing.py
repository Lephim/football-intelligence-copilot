import numpy as np
import pandas as pd

def calculate_progressive_passes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Given normalized events (from load_events), return only the Pass events
    classified as progressive, with a new `progression` column.
    """
    df = events[events["event_type"] == "Pass"].copy()
    df = df.dropna(subset=["x", "y", "end_x", "end_y"])

    def distance_to_goal(x, y):
        return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)

    df["progression"] = distance_to_goal(df["x"], df["y"]) - distance_to_goal(df["end_x"], df["end_y"])

    # zone-dependent threshold, computed as a column via np.select
    conditions = [
        df["end_x"] < 60,
        (df["x"] < 60) & (df["end_x"] >= 60),
        (df["x"] >= 60) & (df["end_x"] >= 60),
    ]
    thresholds = [30, 15, 10]
    df["threshold"] = np.select(conditions, thresholds)

    return df[df["progression"] >= df["threshold"]].drop(columns="threshold")