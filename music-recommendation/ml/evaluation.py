"""Offline ranking evaluation with a chronological holdout."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ml.recommender import EVENT_WEIGHTS, HybridMusicRecommender


def chronological_split(
    interactions: pd.DataFrame, *, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest positive event(s) per user; keep at least one train event."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5.")
    if interactions.empty:
        return interactions.copy(), interactions.copy()
    frame = interactions.copy()
    if "timestamp" in frame.columns:
        frame["_time"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame = frame.sort_values(["user_id", "_time"])
    else:
        frame = frame.sort_values(["user_id"])
    train_parts = []
    test_parts = []
    for _, group in frame.groupby("user_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 3:
            train_parts.append(group)
            continue
        holdout_count = max(1, int(round(len(group) * test_fraction)))
        holdout_count = min(holdout_count, len(group) - 1)
        train_parts.append(group.iloc[:-holdout_count])
        test_parts.append(group.iloc[-holdout_count:])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else frame.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else frame.iloc[0:0]
    return (
        train.drop(columns=["_time"], errors="ignore"),
        test.drop(columns=["_time"], errors="ignore"),
    )


def _dcg(relevance: list[int]) -> float:
    return sum(rel / __import__("math").log2(i + 2) for i, rel in enumerate(relevance))


def evaluate_ranker(
    tracks: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    *,
    strategy: str,
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate top-k ranking. Positive play/complete/like/repeat/save events count."""
    recommender = HybridMusicRecommender(tracks, train_interactions)
    relevant_by_user: dict[str, set[str]] = {}
    for row in test_interactions.itertuples(index=False):
        event_type = str(getattr(row, "event_type", "")).lower()
        if event_type in EVENT_WEIGHTS:
            relevant_by_user.setdefault(str(row.user_id), set()).add(str(row.track_id))

    precisions = []
    recalls = []
    ndcgs = []
    aps = []
    recommended_catalogue: set[str] = set()
    evaluated_users = 0

    for user_id, relevant in relevant_by_user.items():
        if not relevant:
            continue
        ranked = recommender.recommend(
            user_id, limit=k, strategy=strategy, exclude_seen=True
        )
        recommended = [str(item["track_id"]) for item in ranked]
        recommended_catalogue.update(recommended)
        hits = [1 if item in relevant else 0 for item in recommended]
        hit_count = sum(hits)
        precisions.append(hit_count / max(k, 1))
        recalls.append(hit_count / len(relevant))
        ideal = [1] * min(len(relevant), k)
        ndcgs.append(_dcg(hits) / _dcg(ideal) if ideal else 0.0)
        running_hits = 0
        precision_sum = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                running_hits += 1
                precision_sum += running_hits / rank
        aps.append(precision_sum / min(len(relevant), k))
        evaluated_users += 1

    return {
        "strategy": strategy,
        "k": k,
        "users_evaluated": evaluated_users,
        "precision_at_k": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "recall_at_k": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        "map_at_k": float(sum(aps) / len(aps)) if aps else 0.0,
        "catalogue_coverage": (
            len(recommended_catalogue) / max(len(tracks), 1)
        ),
        "note": "Metrics reflect only the supplied dataset and split; synthetic demo data is not real-world evidence.",
    }


def compare_strategies(
    tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    k: int = 10,
) -> list[dict[str, Any]]:
    train, test = chronological_split(interactions)
    return [
        evaluate_ranker(tracks, train, test, strategy=strategy, k=k)
        for strategy in ("popularity", "content", "svd", "knn", "hybrid")
    ]
