"""
scripts/build_xt_grid.py

One-off pipeline: pulls all cached WSL events, builds zone-level
shot/move statistics, runs the iterative xT solver, and saves the
resulting 16x12 grid to disk for use by the API/visualisation layer.

Run from the project root:
    python -m scripts.build_xt_grid
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.ingestion.loader import load_events_cached, get_all_match_ids
from src.analytics.xt_model import build_move_and_shot_data, compute_xt_grid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPETITION_ID = 37
SEASON_IDS = [281, 90, 42, 4]
XG_MODEL_PATH = PROJECT_ROOT / "models" / "xg_model.pkl"
XT_OUTPUT_PATH = PROJECT_ROOT / "models" / "xt_grid.npy"


def load_all_events(competition_id: int, season_ids: list[int]) -> pd.DataFrame:
    match_ids = get_all_match_ids(competition_id, season_ids)
    print(f"Found {len(match_ids)} matches.")

    all_events = []
    for i, match_id in enumerate(match_ids, start=1):
        try:
            events = load_events_cached(match_id)
            all_events.append(events)
        except Exception as e:
            print(f"  [skip] {match_id}: {e}")
        if i % 50 == 0:
            print(f"  loaded {i}/{len(match_ids)}")

    combined = pd.concat(all_events, ignore_index=True)
    print(f"Total events loaded: {len(combined)}")
    return combined


def main():
    events = load_all_events(COMPETITION_ID, SEASON_IDS)
    xg_model = joblib.load(XG_MODEL_PATH)

    print("Building zone-level shot/move statistics...")
    data = build_move_and_shot_data(events, xg_model)

    print("Running iterative xT solver...")
    xt_grid = compute_xt_grid(data)

    np.save(XT_OUTPUT_PATH, xt_grid)
    print(f"xT grid saved to {XT_OUTPUT_PATH}")
    print(f"Grid min/max: {xt_grid.min():.4f} / {xt_grid.max():.4f}")


if __name__ == "__main__":
    main()