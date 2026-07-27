"""
src/api/main.py

FastAPI service exposing the project's analytics (xG, xT, progressive
passes, passing networks) as JSON. Run with `src/` on the path, e.g.
from the project root:

    PYTHONPATH=src uvicorn api.main:app --reload
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from analytics.passing import calculate_progressive_passes
from analytics.xg_model import prepare_shot_features, predict_xg
from ingestion.loader import load_events_cached
from visualisation.pass_network import build_pass_network
from visualisation.shot_map import team_xg_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # main.py -> api -> src -> project root
DATA_DIR = PROJECT_ROOT / "data" / "processed"
XG_MODEL = joblib.load(PROJECT_ROOT / "models" / "xg_model.pkl")
XT_GRID = np.load(PROJECT_ROOT / "models" / "xt_grid.npy")

app = FastAPI(title="Football Intelligence API")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (handles NaN/numpy dtypes correctly)."""
    return json.loads(df.to_json(orient="records"))


def _load_events(match_id: int) -> pd.DataFrame:
    try:
        return load_events_cached(match_id, cache_dir=str(DATA_DIR))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not load match {match_id}: {e}")


def _shots_with_xg(events: pd.DataFrame) -> pd.DataFrame:
    shots = prepare_shot_features(events)
    shots["xg"] = predict_xg(XG_MODEL, shots)
    return shots


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/matches")
def list_matches():
    match_ids = sorted(
        int(p.stem.removeprefix("events_")) for p in DATA_DIR.glob("events_*.parquet")
    )
    return {"count": len(match_ids), "match_ids": match_ids}


@app.get("/matches/{match_id}/shots")
def match_shots(match_id: int, team: str | None = Query(default=None)):
    events = _load_events(match_id)
    shots = _shots_with_xg(events)
    if team is not None:
        shots = shots[shots["team"] == team]
    return _records(shots[["team", "player", "x", "y", "distance", "angle", "is_goal", "xg"]])


@app.get("/matches/{match_id}/xg-summary")
def match_xg_summary(match_id: int):
    events = _load_events(match_id)
    shots = _shots_with_xg(events)
    return _records(team_xg_summary(shots))


@app.get("/matches/{match_id}/progressive-passes")
def match_progressive_passes(match_id: int, team: str | None = Query(default=None)):
    events = _load_events(match_id)
    if team is not None:
        events = events[events["team"] == team]
    passes = calculate_progressive_passes(events)
    return _records(passes[["team", "player", "recipient", "x", "y", "end_x", "end_y", "progression"]])


@app.get("/matches/{match_id}/passing-network")
def match_passing_network(match_id: int, team: str, only_completed: bool = True):
    events = _load_events(match_id)
    node_positions, edge_counts = build_pass_network(events, team, only_completed=only_completed)
    nodes = _records(node_positions.reset_index())
    edges = _records(edge_counts)
    return {"nodes": nodes, "edges": edges}


@app.get("/xt-grid")
def xt_grid():
    grid = [[None if np.isnan(v) else v for v in row] for row in XT_GRID]
    return {"n_cols": XT_GRID.shape[0], "n_rows": XT_GRID.shape[1], "grid": grid}
