"""Tests for Last.fm TSV conversion and event-sampling behaviour."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.import_lastfm_1k import import_lastfm


class LastFmImporterTests(unittest.TestCase):
    def _write_rows(self, path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerows(rows)

    def test_import_maps_one_source_user_to_demo_account(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.tsv"
            output = root / "out"
            rows = [
                ("u1", "2009-04-08T01:57:47Z", "artist-1", "Artist A", "track-1", "Song One"),
                ("u1", "2009-04-08T02:57:47Z", "artist-1", "Artist A", "track-2", "Song Two"),
                ("u2", "2009-04-08T03:57:47Z", r"\N", "Artist B", r"\N", "Song Three"),
            ]
            self._write_rows(source, rows)
            stats = import_lastfm(
                source, output, max_users=0, max_events=0,
                chunksize=2, min_track_plays=1,
                demo_source_user_id="u1",
            )
            self.assertEqual(stats["users"], 2)
            self.assertEqual(stats["tracks"], 3)
            with (output / "interactions.csv").open(encoding="utf-8", newline="") as handle:
                events = list(csv.DictReader(handle))
            self.assertIn("demo", {event["user_id"] for event in events})
            self.assertIn("u2", {event["user_id"] for event in events})
            self.assertTrue(all(event["event_type"] == "play" for event in events))

    def test_event_budget_is_distributed_across_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.tsv"
            output = root / "out"
            rows = []
            for user in ("u1", "u2"):
                for index in range(6):
                    rows.append((
                        user,
                        f"2009-04-{index + 1:02d}T01:00:00Z",
                        f"artist-{user}",
                        f"Artist {user}",
                        f"track-{user}-{index}",
                        f"Song {user} {index}",
                    ))
            self._write_rows(source, rows)
            stats = import_lastfm(
                source, output, max_users=2, max_events=4,
                chunksize=3, min_track_plays=1,
            )
            self.assertEqual(stats["events"], 4)
            with (output / "interactions.csv").open(encoding="utf-8", newline="") as handle:
                events = list(csv.DictReader(handle))
            counts = {}
            for event in events:
                counts[event["user_id"]] = counts.get(event["user_id"], 0) + 1
            self.assertEqual(counts, {"u1": 2, "u2": 2})

    def test_invalid_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.tsv"
            output = root / "out"
            self._write_rows(source, [("u1", "not-a-date", "a", "Artist", "t", "Song")])
            with self.assertRaises(ValueError):
                import_lastfm(
                    source, output, max_users=0, max_events=0,
                    chunksize=2, min_track_plays=1,
                )


if __name__ == "__main__":
    unittest.main()
