"""Evaluate SoundScout recommenders on demo or imported interaction CSVs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ml.evaluation import compare_strategies


def _project_path(path: Path) -> Path:
    """Resolve relative paths from the music-recommendation project directory."""
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all recommenders on the bundled demo data or supplied CSV files."
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Cutoff for ranking metrics (default: 10).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional JSON output file.",
    )
    parser.add_argument(
        "--tracks", type=Path, default=Path("data/tracks.csv"),
        help="Track catalogue CSV (default: data/tracks.csv).",
    )
    parser.add_argument(
        "--interactions", type=Path, default=Path("data/interactions.csv"),
        help="Interaction events CSV (default: data/interactions.csv).",
    )
    parser.add_argument(
        "--dataset-name", type=str, default=None,
        help="Optional label for the report, e.g. 'Last.fm 1K sampled'.",
    )
    args = parser.parse_args()

    if args.k < 1 or args.k > 50:
        parser.error("--k must be between 1 and 50.")

    tracks_path = _project_path(args.tracks)
    interactions_path = _project_path(args.interactions)
    for path in (tracks_path, interactions_path):
        if not path.is_file():
            parser.error("CSV file not found: " + str(path))

    tracks = pd.read_csv(tracks_path)
    interactions = pd.read_csv(interactions_path)

    required_tracks = {"track_id"}
    required_interactions = {"user_id", "track_id", "event_type"}
    missing_tracks = required_tracks - set(tracks.columns)
    missing_interactions = required_interactions - set(interactions.columns)
    if missing_tracks:
        parser.error("Tracks CSV is missing required columns: " + ", ".join(sorted(missing_tracks)))
    if missing_interactions:
        parser.error(
            "Interactions CSV is missing required columns: "
            + ", ".join(sorted(missing_interactions))
        )
    if tracks.empty or interactions.empty:
        parser.error("Both CSV files must contain at least one data row.")

    results = compare_strategies(tracks, interactions, k=args.k)
    is_default_demo = (
        tracks_path.resolve() == (PROJECT_DIR / "data" / "tracks.csv").resolve()
        and interactions_path.resolve() == (PROJECT_DIR / "data" / "interactions.csv").resolve()
    )
    dataset_name = args.dataset_name or (
        "bundled synthetic demonstration data"
        if is_default_demo else "user-supplied interaction dataset"
    )
    output = {
        "dataset": dataset_name,
        "tracks_file": str(tracks_path),
        "interactions_file": str(interactions_path),
        "tracks": int(len(tracks)),
        "interaction_events": int(len(interactions)),
        "users": int(interactions["user_id"].nunique()),
        "results": results,
        "evaluation_notes": [
            "The chronological holdout trains on earlier events and evaluates on later events per user.",
            "Last.fm listening events are positive implicit feedback; they do not contain explicit dislikes or verified negative labels.",
            "The reported balanced sampled-pair accuracy treats sampled unseen tracks as candidate negatives. It is a ranking proxy, not conventional classification accuracy.",
            "Use Precision@K, Recall@K, MAP@K and NDCG@K as the primary recommendation-ranking metrics.",
            "Do not describe any score as guaranteed or as real-world user satisfaction.",
        ],
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        destination = _project_path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print("Wrote evaluation report to " + str(destination))
    print(rendered)


if __name__ == "__main__":
    main()
