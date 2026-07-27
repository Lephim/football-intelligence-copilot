"""
src/analytics/sequence_model.py

EXPERIMENTAL — see README's "Experimental: Possession Predictability" section.

Treats each possession as a sequence of (event_type, zone) tokens and
fits a smoothed order-1 Markov chain over them. The atomic output is
per-action "surprise" (xSurprise): -log2 P(token | previous token).

Uses its OWN, coarser zone grid (SEQ_N_COLS x SEQ_N_ROWS), independently
tunable from xT's 16x12 grid — this model's context-sparsity needs are
different (a (event_type, zone) token space is already much larger than
xT's plain zones, so it needs fewer, larger cells to keep enough
observations per context).
"""

from collections import defaultdict

import numpy as np
import pandas as pd

PITCH_LENGTH, PITCH_WIDTH = 120, 80
DEFAULT_N_COLS, DEFAULT_N_ROWS = 10, 8  # coarser than xT's 16x12 — see module docstring

TOKEN_EVENT_TYPES = ["Pass", "Carry", "Shot", "Dribble"]


def _seq_zone_index(x: np.ndarray, y: np.ndarray, n_cols: int, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Map pitch coordinates to a (col, row) grid at the given resolution —
    resolution is now a parameter, not a fixed constant, so different
    experiments can be compared without editing this file."""
    col = np.clip((np.asarray(x) / PITCH_LENGTH * n_cols).astype(int), 0, n_cols - 1)
    row = np.clip((np.asarray(y) / PITCH_WIDTH * n_rows).astype(int), 0, n_rows - 1)
    return col, row


def _extract_event_order(row: dict):
    """StatsBomb's true chronological order within the match, pulled from
    the preserved raw row — avoids needing a loader.py schema change."""
    if not isinstance(row, dict):
        return None
    return row.get("index")


def tokenize_events(events: pd.DataFrame, n_cols: int = DEFAULT_N_COLS, n_rows: int = DEFAULT_N_ROWS) -> pd.DataFrame:
    """
    Build one row per on-ball action, ordered correctly within each
    possession, with a `token` column: (event_type, col, row).
    """
    actions = (
        events[events["event_type"].isin(TOKEN_EVENT_TYPES)]
        .dropna(subset=["x", "y", "possession"])
        .copy()
    )

    actions["event_order"] = actions["raw"].apply(_extract_event_order)
    actions = actions.dropna(subset=["event_order"]).copy()
    actions["event_order"] = actions["event_order"].astype(int)

    actions["col"], actions["row"] = _seq_zone_index(actions["x"].to_numpy(), actions["y"].to_numpy(), n_cols, n_rows)
    actions["token"] = list(zip(actions["event_type"], actions["col"], actions["row"]))

    return actions.sort_values(["match_id", "possession", "event_order"]).reset_index(drop=True)


def build_possession_sequences(tokenized_actions: pd.DataFrame) -> dict:
    """Group tokenized actions into ordered per-possession sequences:
    {(match_id, possession): [token, token, ...]}."""
    sequences = {}
    for (match_id, possession), group in tokenized_actions.groupby(["match_id", "possession"]):
        sequences[(match_id, possession)] = group["token"].tolist()
    return sequences


def build_transition_model(sequences: dict, alpha: float = 1.0) -> dict:
    """
    Fit a smoothed order-1 Markov model: P(next_token | current_token).
    `sequences` should be TRAINING possessions only.
    """
    counts = defaultdict(lambda: defaultdict(int))
    context_totals = defaultdict(int)
    vocab = set()

    for seq in sequences.values():
        vocab.update(seq)
        for i in range(len(seq) - 1):
            context, nxt = seq[i], seq[i + 1]
            counts[context][nxt] += 1
            context_totals[context] += 1

    return {
        "counts": counts,
        "context_totals": context_totals,
        "vocab_size": len(vocab),
        "alpha": alpha,
    }


def token_probability(model: dict, context, token) -> float:
    """P(token | context) under the smoothed model."""
    c = model["counts"].get(context, {}).get(token, 0)
    total = model["context_totals"].get(context, 0)
    alpha, vocab_size = model["alpha"], model["vocab_size"]
    return (c + alpha) / (total + alpha * vocab_size)


def compute_action_surprise(model: dict, sequences: dict) -> pd.DataFrame:
    """
    Per-action xSurprise = -log2 P(token | previous token), with the
    observed raw count attached so downstream filtering can distinguish
    genuinely rare-but-real transitions from unseen/smoothing-floor cases.
    """
    rows = []
    for (match_id, possession), seq in sequences.items():
        for i in range(1, len(seq)):
            context, token = seq[i - 1], seq[i]
            c = model["counts"].get(context, {}).get(token, 0)
            context_total = model["context_totals"].get(context, 0)
            p = token_probability(model, context, token)
            rows.append({
                "match_id": match_id,
                "possession": possession,
                "position_in_possession": i,
                "context_token": context,
                "token": token,
                "surprise": -np.log2(p),
                "observed_count": c,
                "context_total": context_total,
            })
    return pd.DataFrame(rows)


def compute_possession_perplexity(model: dict, sequences: dict) -> pd.DataFrame:
    """Aggregate per-action surprise to one perplexity value per possession."""
    action_surprise = compute_action_surprise(model, sequences)
    if action_surprise.empty:
        return pd.DataFrame(columns=["match_id", "possession", "perplexity", "n_actions"])

    grouped = action_surprise.groupby(["match_id", "possession"])["surprise"].agg(["mean", "count"])
    grouped["perplexity"] = 2 ** grouped["mean"]
    return grouped.reset_index().rename(columns={"count": "n_actions"})[
        ["match_id", "possession", "perplexity", "n_actions"]
    ]


def shuffled_baseline_sequences(sequences: dict, random_state: int = 42) -> dict:
    """Randomly permute token order WITHIN each possession — sanity-check
    baseline: a real model should score real sequences lower (more
    predictable) than their shuffled versions."""
    rng = np.random.default_rng(random_state)
    shuffled = {}
    for key, seq in sequences.items():
        seq_copy = seq.copy()
        rng.shuffle(seq_copy)
        shuffled[key] = seq_copy
    return shuffled