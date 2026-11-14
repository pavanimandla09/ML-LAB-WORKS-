"""Offline ranking evaluation with a chronological holdout."""
from __future__ import annotations

from typing import Any

import numpy as np
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


def _strategy_scores(recommender: HybridMusicRecommender, user_id: str, strategy: str) -> dict[str, float]:
    """Return candidate track scores using the same strategy definitions as recommend()."""
    vectors = recommender._all_score_vectors(str(user_id))
    if strategy == "hybrid":
        if str(user_id) not in recommender.user_index or recommender.svd_model is None:
            scores = 0.55 * vectors["content"] + 0.45 * vectors["popularity"]
        else:
            scores = (
                0.45 * vectors["svd"]
                + 0.40 * vectors["content"]
                + 0.15 * vectors["popularity"]
            )
    elif strategy == "svd" and (
        recommender.svd_model is None or str(user_id) not in recommender.user_index
    ):
        scores = vectors["popularity"]
    elif strategy == "knn" and not np.any(vectors["knn"]):
        scores = vectors["popularity"]
    elif strategy == "content" and not np.any(vectors["content"]):
        scores = vectors["popularity"]
    else:
        scores = vectors[strategy]
    return {
        track_id: float(scores[index])
        for index, track_id in enumerate(recommender.track_ids)
    }


def _sampled_pair_accuracy(
    recommender: HybridMusicRecommender,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    *,
    strategy: str,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Balanced pair-classification accuracy using held-out positives and sampled unseen negatives.

    For each evaluated user, sample as many catalogue tracks with no recorded
    positive interaction as there are held-out positives. Predict the top
    n_positive tracks in this balanced candidate set as relevant. This fixes
    the decision threshold by the known candidate-set class balance; it is not
    a calibrated probability or general-purpose classifier accuracy.
    """
    rng = np.random.default_rng(random_state)
    positive_events = test_interactions[
        test_interactions["event_type"].astype(str).str.lower().isin(EVENT_WEIGHTS)
    ].copy()
    if positive_events.empty:
        return {"accuracy": 0.0, "accuracy_pairs": 0, "accuracy_definition": "balanced held-out user-track pairs"}

    train_positive = train_interactions[
        train_interactions["event_type"].astype(str).str.lower().isin(EVENT_WEIGHTS)
    ] if not train_interactions.empty else train_interactions
    total_correct = 0
    total_pairs = 0

    for user_id, group in positive_events.groupby("user_id"):
        positives = set(group["track_id"].astype(str)) & set(recommender.track_ids)
        if not positives:
            continue
        seen_positive = set()
        if not train_positive.empty:
            seen_positive = set(
                train_positive.loc[
                    train_positive["user_id"].astype(str) == str(user_id), "track_id"
                ].astype(str)
            )
        # Exclude all known positives for this user from the negative pool.
        all_user_positive = set(
            positive_events.loc[
                positive_events["user_id"].astype(str) == str(user_id), "track_id"
            ].astype(str)
        ) | seen_positive
        negative_pool = [
            track_id for track_id in recommender.track_ids
            if track_id not in all_user_positive
        ]
        if not negative_pool:
            continue
        negative_count = min(len(positives), len(negative_pool))
        negatives = set(rng.choice(negative_pool, size=negative_count, replace=False).tolist())
        candidates = positives | negatives
        scores = _strategy_scores(recommender, str(user_id), strategy)
        ranked = sorted(candidates, key=lambda track_id: scores.get(track_id, 0.0), reverse=True)
        predicted_positive = set(ranked[:len(positives)])
        true_positive = len(predicted_positive & positives)
        true_negative = len((candidates - predicted_positive) & negatives)
        total_correct += true_positive + true_negative
        total_pairs += len(candidates)

    return {
        "accuracy": total_correct / total_pairs if total_pairs else 0.0,
        "accuracy_pairs": total_pairs,
        "accuracy_definition": "balanced held-out user-track pairs; top-N classification on equal positive/negative candidates",
    }


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

    pair_accuracy = _sampled_pair_accuracy(
        recommender, train_interactions, test_interactions, strategy=strategy
    )

    return {
        "strategy": strategy,
        **pair_accuracy,
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
