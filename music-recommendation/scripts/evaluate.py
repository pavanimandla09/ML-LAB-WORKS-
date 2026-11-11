"""Run all recommenders against the bundled synthetic interaction sample."""
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



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="Cutoff for ranking metrics (default: 10).")
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    args = parser.parse_args()
    if args.k < 1 or args.k > 50:
        parser.error("--k must be between 1 and 50.")

    tracks = pd.read_csv(PROJECT_DIR / "data" / "tracks.csv")
    interactions = pd.read_csv(PROJECT_DIR / "data" / "interactions.csv")
    results = compare_strategies(tracks, interactions, k=args.k)
    output = {
        "dataset": "bundled synthetic demonstration data",
        "tracks": int(len(tracks)),
        "interaction_events": int(len(interactions)),
        "results": results,
        "warning": "These metrics demonstrate the evaluation pipeline only. They are not evidence of real-user recommendation quality.",
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        destination = args.output
        if not destination.is_absolute():
            destination = PROJECT_DIR / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print("Wrote evaluation report to " + str(destination))
    print(rendered)


if __name__ == "__main__":
    main()
