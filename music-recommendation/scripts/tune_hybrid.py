"""Tune SoundScout hybrid weights on validation data and report one untouched test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.evaluation import (
    evaluate_ranker,
    chronological_train_validation_test_split,
    tune_hybrid_weights,
)
from ml.recommender import NEGATIVE_EVENTS


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def _negative_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "event_type" not in frame.columns:
        return {event: 0 for event in sorted(NEGATIVE_EVENTS)}
    values = frame["event_type"].astype(str).str.lower()
    return {event: int((values == event).sum()) for event in sorted(NEGATIVE_EVENTS)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hybrid weights on chronological validation data and evaluate once on a later test partition."
    )
    parser.add_argument("--tracks", type=Path, default=Path("data/tracks.csv"))
    parser.add_argument("--interactions", type=Path, default=Path("data/interactions.csv"))
    parser.add_argument("--dataset-name", default="user-supplied interaction dataset")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    if not 1 <= args.k <= 50:
        parser.error("--k must be between 1 and 50.")

    tracks_path = _project_path(args.tracks)
    interactions_path = _project_path(args.interactions)
    for path in (tracks_path, interactions_path):
        if not path.is_file():
            parser.error("CSV file not found: " + str(path))

    tracks = pd.read_csv(tracks_path)
    interactions = pd.read_csv(interactions_path)
    required_tracks = {"track_id"}
    required_events = {"user_id", "track_id", "event_type"}
    if missing := required_tracks - set(tracks.columns):
        parser.error("Tracks CSV is missing required columns: " + ", ".join(sorted(missing)))
    if missing := required_events - set(interactions.columns):
        parser.error("Interactions CSV is missing required columns: " + ", ".join(sorted(missing)))
    if tracks.empty or interactions.empty:
        parser.error("Both CSV files must contain at least one data row.")

    train, validation, test = chronological_train_validation_test_split(interactions)
    if validation.empty or test.empty:
        parser.error(
            "Not enough events per user for chronological train/validation/test evaluation. "
            "Use a larger interaction dataset; do not tune and test on the same events."
        )

    selection = tune_hybrid_weights(tracks, train, validation, k=args.k)
    refit_train = pd.concat([train, validation], ignore_index=True)
    baseline = evaluate_ranker(
        tracks, refit_train, test, strategy="hybrid", k=args.k
    )
    tuned = evaluate_ranker(
        tracks, refit_train, test, strategy="hybrid", k=args.k,
        hybrid_weights=selection["weights"],
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
        "negative_feedback_counts": {
            "all": _negative_counts(interactions),
            "train": _negative_counts(train),
            "validation": _negative_counts(validation),
            "test": _negative_counts(test),
        },
        "weight_selection": selection,
        "test_results": {
            "baseline_hybrid": baseline,
            "validation_tuned_hybrid": tuned,
        },
        "evaluation_notes": [
            "Weights are selected using validation NDCG@K only; the later test partition is not used during selection.",
            "The final test comparison refits on train plus validation and evaluates on the untouched chronological test partition.",
            "Balanced sampled-pair accuracy is a ranking proxy, not conventional classification accuracy or user satisfaction.",
            "Last.fm listening logs do not contain explicit skip/unlike labels; zero negative-event counts mean the negative-feedback feature cannot be learned from that source.",
            "A high score is not guaranteed. Report all metrics and the dataset/split, including if tuning does not improve the test result.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        destination = _project_path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print("Wrote tuning report to " + str(destination))
    print(rendered)


if __name__ == "__main__":
    main()
