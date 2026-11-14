"""Offline ranking evaluation with a chronological holdout."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.recommender import (
    COLD_START_WEIGHTS,
    EVENT_WEIGHTS,
    HYBRID_WEIGHTS,
    HybridMusicRecommender,
)


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
            weights = COLD_START_WEIGHTS
        else:
            weights = recommender.hybrid_weights
        active_weights = {
            key: weight for key, weight in weights.items()
            if np.any(vectors[key])
        }
        weight_total = sum(active_weights.values())
        if weight_total:
            scores = sum(
                vectors[key] * (weight / weight_total)
                for key, weight in active_weights.items()
            )
        else:
            scores = vectors["popularity"]
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
        seen_positive = set()
        if not train_positive.empty:
            seen_positive = set(
                train_positive.loc[
                    train_positive["user_id"].astype(str) == str(user_id), "track_id"
                ].astype(str)
            )
        # The recommender excludes tracks already present in a user's training
        # history, so repeated tracks cannot be valid ranking positives.
        positives = (
            set(group["track_id"].astype(str)) & set(recommender.track_ids)
        ) - seen_positive
        if not positives:
            continue
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
        "accuracy_definition": "balanced novel held-out user-track pairs; repeated training positives excluded; top-N classification on equal positive/negative candidates",
    }


def evaluate_ranker(
    tracks: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    *,
    strategy: str,
    k: int = 10,
    hybrid_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate top-k ranking. Positive play/complete/like/repeat/save events count."""
    recommender = HybridMusicRecommender(
        tracks, train_interactions, hybrid_weights=hybrid_weights
    )
    relevant_by_user: dict[str, set[str]] = {}
    train_seen_by_user: dict[str, set[str]] = {}
    if not train_interactions.empty:
        train_positive = train_interactions[
            train_interactions["event_type"].astype(str).str.lower().isin(EVENT_WEIGHTS)
        ]
        for user_id, group in train_positive.groupby("user_id"):
            train_seen_by_user[str(user_id)] = set(group["track_id"].astype(str))

    for row in test_interactions.itertuples(index=False):
        event_type = str(getattr(row, "event_type", "")).lower()
        user_id = str(row.user_id)
        track_id = str(row.track_id)
        # Evaluate only novel held-out tracks: recommend() intentionally
        # excludes tracks already consumed in the training history.
        if (
            event_type in EVENT_WEIGHTS
            and track_id not in train_seen_by_user.get(user_id, set())
        ):
            relevant_by_user.setdefault(user_id, set()).add(track_id)

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
        "note": "Metrics reflect only the supplied dataset and split; verify dataset provenance and licensing before making real-world claims.",
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


def chronological_train_validation_test_split(
    interactions: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train/validation/test partitions independently per user."""
    if not 0.05 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.05 and 0.4.")
    if not 0.05 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.05 and 0.4.")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation_fraction + test_fraction must be below 0.8.")
    if interactions.empty:
        empty = interactions.copy()
        return empty, empty.copy(), empty.copy()

    frame = interactions.copy()
    if "timestamp" in frame.columns:
        frame["_time"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame = frame.sort_values(["user_id", "_time"], kind="stable")
    else:
        frame = frame.sort_values(["user_id"], kind="stable")

    train_parts, validation_parts, test_parts = [], [], []
    for _, group in frame.groupby("user_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 5:
            train_parts.append(group)
            continue
        test_count = max(1, int(round(len(group) * test_fraction)))
        validation_count = max(1, int(round(len(group) * validation_fraction)))
        while test_count + validation_count >= len(group) - 1:
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                break
        train_end = len(group) - validation_count - test_count
        validation_end = len(group) - test_count
        train_parts.append(group.iloc[:train_end])
        validation_parts.append(group.iloc[train_end:validation_end])
        test_parts.append(group.iloc[validation_end:])

    def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return frame.iloc[0:0].drop(columns=["_time"], errors="ignore")
        return pd.concat(parts, ignore_index=True).drop(columns=["_time"], errors="ignore")

    return combine(train_parts), combine(validation_parts), combine(test_parts)


def tune_hybrid_weights(
    tracks: pd.DataFrame,
    train_interactions: pd.DataFrame,
    validation_interactions: pd.DataFrame,
    *,
    k: int = 10,
) -> dict[str, Any]:
    """Select hybrid weights on validation NDCG@K; never inspect the final test set."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    recommender = HybridMusicRecommender(tracks, train_interactions)
    positive_validation = validation_interactions[
        validation_interactions["event_type"].astype(str).str.lower().isin(EVENT_WEIGHTS)
    ].copy()
    if positive_validation.empty:
        raise ValueError("Validation partition contains no positive listening events.")

    user_cases = []
    for user_id, group in positive_validation.groupby("user_id"):
        user_id = str(user_id)
        seen = recommender._seen_track_ids(user_id)
        relevant = set(group["track_id"].astype(str)) - seen
        if not relevant:
            continue
        vectors = recommender._all_score_vectors(user_id)
        user_cases.append((user_id, relevant, seen, vectors))
    if not user_cases:
        raise ValueError("No novel positive validation tracks remain after excluding training history.")

    best_score = -1.0
    best_weights: dict[str, float] | None = None
    tested = 0
    signal_names = ("content", "knn", "svd", "popularity")
    # Exhaustive simplex grid at 0.1 increments: 286 candidates, all summing to 1.
    for content_units in range(11):
        for knn_units in range(11 - content_units):
            for svd_units in range(11 - content_units - knn_units):
                popularity_units = 10 - content_units - knn_units - svd_units
                units = (content_units, knn_units, svd_units, popularity_units)
                weights = {
                    name: units[index] / 10.0
                    for index, name in enumerate(signal_names)
                }
                ndcgs = []
                for _, relevant, seen, vectors in user_cases:
                    active = {
                        name: weight for name, weight in weights.items()
                        if weight > 0 and np.any(vectors[name])
                    }
                    total = sum(active.values())
                    if total:
                        scores = sum(
                            vectors[name] * (weight / total)
                            for name, weight in active.items()
                        )
                    else:
                        scores = vectors["popularity"]
                    ranked_indices = np.argsort(-scores, kind="stable")
                    recommended = [
                        recommender.track_ids[int(index)]
                        for index in ranked_indices
                        if recommender.track_ids[int(index)] not in seen
                    ][:k]
                    hits = [1 if track_id in relevant else 0 for track_id in recommended]
                    ideal = [1] * min(len(relevant), k)
                    ndcgs.append(_dcg(hits) / _dcg(ideal) if ideal else 0.0)
                score = float(np.mean(ndcgs)) if ndcgs else 0.0
                tested += 1
                if score > best_score:
                    best_score = score
                    best_weights = weights

    if best_weights is None:
        raise RuntimeError("Hybrid weight search did not produce a candidate.")
    return {
        "weights": best_weights,
        "validation_ndcg_at_k": best_score,
        "validation_users": len(user_cases),
        "candidates_tested": tested,
        "selection_metric": f"mean NDCG@{k} on chronological validation partition",
    }
