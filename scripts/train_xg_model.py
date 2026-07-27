"""
scripts/train_xg_model.py — updated for the richer feature set.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from src.ingestion.loader import load_events_cached, get_all_match_ids
from src.analytics.xg_model import prepare_shot_features, train_xg_model, predict_xg, compute_penalty_xg, FEATURE_COLUMNS

COMPETITION_ID = 37
SEASON_IDS = [281, 90, 42, 4]
MODEL_OUTPUT_PATH = "models/xg_model.pkl"


def build_shot_dataset(competition_id, season_ids):
    match_ids = get_all_match_ids(competition_id, season_ids)
    all_shots = []
    for i, match_id in enumerate(match_ids, start=1):
        try:
            events = load_events_cached(match_id)
        except Exception as e:
            print(f"  [skip] {match_id}: {e}")
            continue
        # include_penalties=True here — we need them to compute the empirical
        # penalty rate, even though they're excluded from the geometric model's training set
        shots = prepare_shot_features(events, include_penalties=True)
        shots["match_id"] = match_id
        all_shots.append(shots)
        if i % 50 == 0:
            print(f"  processed {i}/{len(match_ids)}")
    return pd.concat(all_shots, ignore_index=True)


def match_level_split(shots, test_size=0.2, val_size=0.1, random_state=42):
    match_ids = shots["match_id"].unique()
    train_val_ids, test_ids = train_test_split(match_ids, test_size=test_size, random_state=random_state)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size / (1 - test_size), random_state=random_state)
    return (shots[shots["match_id"].isin(train_ids)],
            shots[shots["match_id"].isin(val_ids)],
            shots[shots["match_id"].isin(test_ids)])


def evaluate(artifact, split, split_name):
    y = split["is_goal"]
    preds = predict_xg(artifact, split)
    metrics = {
        "log_loss": log_loss(y, preds),
        "brier_score": brier_score_loss(y, preds),
        "roc_auc": roc_auc_score(y, preds),
    }
    print(f"[{split_name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main():
    all_shots = build_shot_dataset(COMPETITION_ID, SEASON_IDS)
    print(f"Total shots: {len(all_shots)} (goals: {all_shots['is_goal'].sum()}, "
          f"penalties: {all_shots['is_penalty'].sum()})")

    penalty_xg = compute_penalty_xg(all_shots)
    print(f"Empirical penalty conversion rate: {penalty_xg:.3f}")

    non_penalty_shots = all_shots[all_shots["is_penalty"] == 0].copy()
    train, val, test = match_level_split(non_penalty_shots)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    artifact = train_xg_model(train)
    artifact["penalty_xg"] = penalty_xg  # bundle it into the saved artifact

    print("\nEvaluation (non-penalty shots only):")
    evaluate(artifact, val, "validation")
    evaluate(artifact, test, "test")

    model = artifact["model"]
    print("\nCoefficients:")
    for name, coef in zip(FEATURE_COLUMNS, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"\nSaved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()