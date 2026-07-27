"""
scripts/train_xg_model.py

One-off (re-runnable) training pipeline for the xG model.
Pulls WSL shot data across multiple seasons, splits by match,
trains a logistic regression, evaluates it, and saves the fitted
model to disk for the API/serving layer to load at inference time.

Run from the project root:
    python -m scripts.train_xg_model
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from src.ingestion.loader import load_events_cached, get_all_match_ids
from src.analytics.xg_model import prepare_shot_features

# --- config -----------------------------------------------------------
COMPETITION_ID = 37  # FA Women's Super League
SEASON_IDS = [281, 90, 42, 4]  # 2023/24, 2022/23, 2021/22, 2020/21
MODEL_OUTPUT_PATH = "models/xg_model.pkl"
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
# ------------------------------------------------------------------------


def build_shot_dataset(competition_id: int, season_ids: list[int]) -> pd.DataFrame:
    match_ids = get_all_match_ids(competition_id, season_ids)
    print(f"Found {len(match_ids)} matches across {len(season_ids)} season(s).")

    all_shots = []
    for i, match_id in enumerate(match_ids, start=1):
        try:
            events = load_events_cached(match_id)
        except Exception as e:
            print(f"  [skip] match {match_id} failed to load: {e}")
            continue

        shots = prepare_shot_features(events)
        all_shots.append(shots)

        if i % 20 == 0:
            print(f"  processed {i}/{len(match_ids)} matches...")

    dataset = pd.concat(all_shots, ignore_index=True)
    print(f"Total shots collected: {len(dataset)}  (goals: {dataset['is_goal'].sum()})")
    return dataset


def match_level_split(shots: pd.DataFrame, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    match_ids = shots["match_id"].unique()

    train_val_ids, test_ids = train_test_split(
        match_ids, test_size=test_size, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size / (1 - test_size), random_state=random_state
    )

    train = shots[shots["match_id"].isin(train_ids)]
    val = shots[shots["match_id"].isin(val_ids)]
    test = shots[shots["match_id"].isin(test_ids)]

    print(f"Split -> train: {len(train)} shots ({train['match_id'].nunique()} matches), "
          f"val: {len(val)} shots ({val['match_id'].nunique()} matches), "
          f"test: {len(test)} shots ({test['match_id'].nunique()} matches)")

    return train, val, test


def train_xg_model(train: pd.DataFrame) -> LogisticRegression:
    X = train[["distance", "angle"]]
    y = train["is_goal"]

    model = LogisticRegression()
    model.fit(X, y)
    return model


def evaluate(model: LogisticRegression, split: pd.DataFrame, split_name: str) -> dict:
    X = split[["distance", "angle"]]
    y = split["is_goal"]
    preds = model.predict_proba(X)[:, 1]

    metrics = {
        "log_loss": log_loss(y, preds),
        "brier_score": brier_score_loss(y, preds),
        "roc_auc": roc_auc_score(y, preds),
    }
    print(f"[{split_name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main():
    shots = build_shot_dataset(COMPETITION_ID, SEASON_IDS)
    train, val, test = match_level_split(shots)

    model = train_xg_model(train)

    print("\nEvaluation:")
    evaluate(model, val, "validation")
    evaluate(model, test, "test")

    print(f"\nModel coefficients: distance={model.coef_[0][0]:.4f}, angle={model.coef_[0][1]:.4f}")
    print(f"Intercept: {model.intercept_[0]:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()