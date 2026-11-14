"""Fine-tune hybrid weights locally around the coarse validation optimum.

The coarse search and all refinement use only the chronological validation
partition. The final chronological test partition is evaluated only after the
weights have been selected.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.evaluation import (
    _dcg,
    chronological_train_validation_test_split,
    evaluate_ranker,
    tune_hybrid_weights,
)
from ml.recommender import EVENT_WEIGHTS, HybridMusicRecommender


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def _fine_tune_weights(
    tracks: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    coarse_weights: dict[str, float],
    *,
    k: int,
    radius: float = 0.10,
    step: float = 0.05,
) -> dict[str, Any]:
    """Search a local 0.05-step simplex around the coarse validation winner."""
    recommender = HybridMusicRecommender(tracks, train)
    positives = validation[
        validation["event_type"].astype(str).str.lower().isin(EVENT_WEIGHTS)
    ]
    user_cases = []
    for user_id, group in positives.groupby("user_id"):
        user_id = str(user_id)
        seen = recommender._seen_track_ids(user_id)
        relevant = set(group["track_id"].astype(str)) - seen
        if not relevant:
            continue
        vectors = recommender._all_score_vectors(user_id)
        seen_indices = np.fromiter(
            (i for i, track_id in enumerate(recommender.track_ids) if track_id in seen),
            dtype=int,
        )
        user_cases.append((relevant, seen_indices, vectors))

    if not user_cases:
        raise ValueError("No novel positive validation tracks remain after excluding training history.")

    names = ("content", "artist", "knn", "svd", "popularity")
    axes = []
    for name in names:
        center = float(coarse_weights[name])
        values = {
            round(max(0.0, min(1.0, center + offset)), 2)
            for offset in np.arange(-radius, radius + step / 2, step)
        }
        axes.append(sorted(values))

    candidates = []
    for values in itertools.product(*axes):
        if abs(sum(values) - 1.0) > 1e-8:
            continue
        candidates.append(dict(zip(names, values)))
    # Always retain the coarse optimum, even if floating-point axis construction changes.
    if not any(all(abs(candidate[n] - coarse_weights[n]) < 1e-8 for n in names) for candidate in candidates):
        candidates.append(dict(coarse_weights))

    best_score = -1.0
    best_weights = None
    for weights in candidates:
        per_user_ndcg = []
        for relevant, seen_indices, vectors in user_cases:
            active = [
                name for name, weight in weights.items()
                if weight > 0 and np.any(vectors[name])
            ]
            if active:
                total = sum(weights[name] for name in active)
                scores = sum(
                    vectors[name] * (weights[name] / total) for name in active
                )
            else:
                scores = vectors["popularity"]

            take = min(k, max(len(scores) - len(seen_indices), 0))
            if take:
                score_copy = scores.copy()
                if len(seen_indices):
                    score_copy[seen_indices] = -np.inf
                indices = np.argpartition(score_copy, -take)[-take:]
                indices = indices[np.argsort(-score_copy[indices], kind="stable")]
                ranked = [recommender.track_ids[int(i)] for i in indices]
            else:
                ranked = []
            hits = [int(track_id in relevant) for track_id in ranked]
            ideal = [1] * min(len(relevant), k)
            per_user_ndcg.append(_dcg(hits) / _dcg(ideal) if ideal else 0.0)

        score = float(np.mean(per_user_ndcg)) if per_user_ndcg else 0.0
        if score > best_score:
            best_score = score
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("Fine-tuning did not produce a candidate.")
    return {
        "weights": best_weights,
        "validation_ndcg_at_k": best_score,
        "validation_users": len(user_cases),
        "candidates_tested": len(candidates),
        "selection_metric": f"mean NDCG@{k} on chronological validation partition",
        "local_radius": radius,
        "weight_step": step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine coarse hybrid weights on validation data; evaluate the test partition once."
    )
    parser.add_argument("--tracks", type=Path, default=Path("data/tracks.csv"))
    parser.add_argument("--interactions", type=Path, default=Path("data/interactions.csv"))
    parser.add_argument("--dataset-name", default="user-supplied interaction dataset")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.k <= 50:
        parser.error("--k must be between 1 and 50.")

    tracks_path, interactions_path = map(
        _project_path, (args.tracks, args.interactions)
    )
    for path in (tracks_path, interactions_path):
        if not path.is_file():
            parser.error("CSV file not found: " + str(path))

    tracks = pd.read_csv(tracks_path)
    interactions = pd.read_csv(interactions_path)
    train, validation, test = chronological_train_validation_test_split(interactions)
    if validation.empty or test.empty:
        parser.error("Not enough events for chronological train/validation/test evaluation.")

    coarse = tune_hybrid_weights(tracks, train, validation, k=args.k)
    fine = _fine_tune_weights(
        tracks, train, validation, coarse["weights"], k=args.k
    )
    refit_train = pd.concat([train, validation], ignore_index=True)
    baseline = evaluate_ranker(tracks, refit_train, test, strategy="hybrid", k=args.k)
    coarse_test = evaluate_ranker(
        tracks, refit_train, test, strategy="hybrid", k=args.k,
        hybrid_weights=coarse["weights"],
    )
    fine_test = evaluate_ranker(
        tracks, refit_train, test, strategy="hybrid", k=args.k,
        hybrid_weights=fine["weights"],
    )
    report = {
        "dataset": args.dataset_name,
        "tracks": int(len(tracks)),
        "interaction_events": int(len(interactions)),
        "users": int(interactions["user_id"].nunique()),
        "partition_sizes": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "coarse_validation_selection": coarse,
        "fine_validation_selection": fine,
        "test_results": {
            "baseline_hybrid": baseline,
            "coarse_tuned_hybrid": coarse_test,
            "fine_tuned_hybrid": fine_test,
        },
        "evaluation_notes": [
            "Both coarse and fine weights are selected using validation NDCG@K only.",
            "The chronological test partition is not read during weight selection.",
            "Test results are reported for baseline, coarse tuning and local fine tuning on the same untouched partition.",
            "Balanced sampled-pair accuracy is a ranking proxy, not conventional classification accuracy or user satisfaction.",
            "Last.fm listening logs contain positive implicit feedback; sampled unseen tracks are not verified dislikes.",
            "No target score is guaranteed; report every metric and the data split.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        destination = _project_path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print("Wrote fine-tuning report to", destination)
    print(rendered)


if __name__ == "__main__":
    main()
