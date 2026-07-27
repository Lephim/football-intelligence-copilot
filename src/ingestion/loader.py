import pandas as pd
from statsbombpy import sb
import os


def _extract_end_location(row: dict) -> tuple[float | None, float | None]:
    """Pull end (x, y) from whichever type-specific field applies to this event."""
    event_type = row.get("type")

    field_map = {
        "Pass": "pass_end_location",
        "Carry": "carry_end_location",
        "Shot": "shot_end_location",
    }

    field = field_map.get(event_type)
    if field is None:
        return None, None

    end_loc = row.get(field)
    if isinstance(end_loc, list) and len(end_loc) >= 2:
        return end_loc[0], end_loc[1]
    return None, None


def _extract_outcome(row: dict) -> str | None:
    """Pull outcome from whichever type-specific field applies to this event."""
    event_type = row.get("type")

    outcome_field_map = {
        "Pass": "pass_outcome",
        "Shot": "shot_outcome",
        "Duel": "duel_outcome",
        "Dribble": "dribble_outcome",
    }

    field = outcome_field_map.get(event_type)
    if field is None:
        return None
    return row.get(field)

def _extract_recipient(row: dict) -> str | None:
    """Pass events carry a recipient; other event types don't."""
    if row.get("type") != "Pass":
        return None
    return row.get("pass_recipient")


def get_available_seasons(competition_id: int) -> pd.DataFrame:
    comps = sb.competitions()
    return comps[comps["competition_id"] == competition_id]


def get_all_match_ids(competition_id: int, season_ids: list[int]) -> list[int]:
    match_ids = []
    for season_id in season_ids:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        match_ids.extend(matches["match_id"].tolist())
    return match_ids

def load_events_cached(match_id: int, cache_dir: str = "data/processed") -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/events_{match_id}.parquet"

    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    events = load_events(match_id)
    events.to_parquet(cache_path)
    return events

def load_events(match_id: int) -> pd.DataFrame:
    """
    Load and normalize StatsBomb event data for a single match.

    Returns one row per event with: event_id, match_id, minute, second,
    period, team, player, event_type, x, y, end_x, end_y, outcome,
    possession, raw.
    """
    raw_df = sb.events(match_id=match_id)
    records = raw_df.to_dict(orient="records")

    rows = []
    for r in records:
        loc = r.get("location")
        x, y = (loc[0], loc[1]) if isinstance(loc, list) and len(loc) >= 2 else (None, None)
        end_x, end_y = _extract_end_location(r)

        rows.append({
            "event_id": r.get("id"),
            "match_id": match_id,
            "minute": r.get("minute"),
            "second": r.get("second"),
            "period": r.get("period"),
            "team": r.get("team"),
            "player": r.get("player"),
            "recipient": _extract_recipient(r),
            "event_type": r.get("type"),
            "x": x,
            "y": y,
            "end_x": end_x,
            "end_y": end_y,
            "outcome": _extract_outcome(r),
            "possession": r.get("possession"),
            "raw": r,
        })

    return pd.DataFrame(rows)