"""Convert the public Last.fm 1K TSV into SoundScout's canonical CSV schema.

Two streaming passes sample listeners uniformly, and per-listener reservoir
sampling prevents a user-grouped TSV from spending the event budget on only
the first few listeners.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "lastfm_import"
SOURCE_COLUMNS = [
    "source_user_id", "timestamp", "artist_mbid", "artist_name",
    "source_track_id", "track_name",
]
TRACK_COLUMNS = [
    "track_id", "title", "artist", "album", "genre", "mood", "tags",
    "energy", "danceability", "valence", "tempo", "acousticness",
    "instrumentalness", "audio_file",
]


def _clean(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} or text == r"\N" else text


def _track_id(track_mbid: str, artist: str, title: str) -> str:
    identity = (
        "mbid:" + track_mbid.casefold()
        if track_mbid
        else "fallback:" + artist.casefold() + "\0" + title.casefold()
    )
    return "lfm_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]


def _reader(source: Path, chunksize: int):
    return pd.read_csv(
        source, sep="\t", header=None, names=SOURCE_COLUMNS, dtype=str,
        keep_default_na=False, chunksize=chunksize, quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    for column in SOURCE_COLUMNS:
        if column not in chunk.columns:
            chunk[column] = ""
        chunk[column] = chunk[column].map(_clean)
    chunk = chunk[
        (chunk["source_user_id"] != "")
        & (chunk["artist_name"] != "")
        & (chunk["track_name"] != "")
        & (chunk["timestamp"] != "")
    ].copy()
    if chunk.empty:
        return chunk
    parsed = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
    chunk = chunk[parsed.notna()].copy()
    if not chunk.empty:
        chunk["timestamp"] = parsed.loc[chunk.index].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return chunk


def import_lastfm(
    source: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    max_users: int = 150,
    max_events: int = 250_000,
    chunksize: int = 100_000,
    min_track_plays: int = 2,
    demo_source_user_id: str = "",
    random_state: int = 42,
) -> dict[str, int]:
    """Write canonical tracks.csv and interactions.csv in output_dir.

    A zero max_users or max_events value means no limit. With a finite event
    budget, reservoir sampling retains a time-spread sample per selected user.
    """
    if not source.is_file():
        raise FileNotFoundError("Input TSV file not found: " + str(source))
    if max_users < 0 or max_events < 0 or chunksize < 1 or min_track_plays < 1:
        raise ValueError("Limits must be non-negative and chunksize/min_track_plays must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_state)

    # Pass 1: discover unique IDs without retaining all listening rows.
    all_users: set[str] = set()
    for raw_chunk in _reader(source, chunksize):
        chunk = _clean_chunk(raw_chunk)
        all_users.update(chunk["source_user_id"].unique().tolist())
    if not all_users:
        raise ValueError("No valid user IDs were found in the input TSV.")

    ordered_users = sorted(all_users)
    if demo_source_user_id and demo_source_user_id not in all_users:
        raise ValueError(
            "The requested --demo-source-user-id was not found in the input TSV."
        )
    if max_users and len(ordered_users) > max_users:
        if demo_source_user_id:
            remaining_users = [user for user in ordered_users if user != demo_source_user_id]
            sample_count = max(0, max_users - 1)
            selected_users = {demo_source_user_id, *rng.sample(remaining_users, sample_count)}
        else:
            selected_users = set(rng.sample(ordered_users, max_users))
    else:
        selected_users = set(ordered_users)

    if max_events and max_events < len(selected_users):
        # Keep at least one event per selected user when the event budget is tiny.
        ordered_selected = sorted(selected_users)
        if demo_source_user_id:
            remaining_users = [user for user in ordered_selected if user != demo_source_user_id]
            sample_count = max(0, max_events - 1)
            selected_users = {demo_source_user_id, *rng.sample(remaining_users, sample_count)}
        else:
            selected_users = set(rng.sample(ordered_selected, max_events))

    user_limits: dict[str, int] = {}
    if max_events:
        ordered_selected = sorted(selected_users)
        base, remainder = divmod(max_events, len(ordered_selected))
        user_limits = {
            user_id: base + (1 if index < remainder else 0)
            for index, user_id in enumerate(ordered_selected)
        }

    # Pass 2: reservoir-sample event rows for each selected listener.
    event_buckets: dict[str, list[tuple[str, str, str, str]]] = {
        user_id: [] for user_id in selected_users
    }
    event_counts: Counter[str] = Counter()
    for raw_chunk in _reader(source, chunksize):
        chunk = _clean_chunk(raw_chunk)
        if chunk.empty:
            continue
        chunk = chunk[chunk["source_user_id"].isin(selected_users)]
        for row in chunk.itertuples(index=False):
            user_id = _clean(row.source_user_id)
            artist = _clean(row.artist_name)
            title = _clean(row.track_name)
            mbid = _clean(row.source_track_id)
            timestamp = _clean(row.timestamp)
            record = (timestamp, artist, title, mbid)
            event_counts[user_id] += 1
            bucket = event_buckets[user_id]
            if not max_events:
                bucket.append(record)
                continue
            quota = user_limits[user_id]
            if len(bucket) < quota:
                bucket.append(record)
            elif quota:
                replacement = rng.randrange(event_counts[user_id])
                if replacement < quota:
                    bucket[replacement] = record

    events: list[tuple[str, str, str, str]] = []
    track_counts: Counter[str] = Counter()
    tracks: dict[str, dict[str, str | float]] = {}
    for source_user_id in sorted(event_buckets):
        output_user_id = (
            "demo"
            if demo_source_user_id and source_user_id == demo_source_user_id
            else source_user_id
        )
        for timestamp, artist, title, mbid in event_buckets[source_user_id]:
            track_id = _track_id(mbid, artist, title)
            if track_id not in tracks:
                tracks[track_id] = {
                    "track_id": track_id, "title": title, "artist": artist,
                    "album": "", "genre": "Unknown", "mood": "", "tags": "",
                    "energy": 0.0, "danceability": 0.0, "valence": 0.0,
                    "tempo": 0.0, "acousticness": 0.0,
                    "instrumentalness": 0.0, "audio_file": "",
                }
            events.append((output_user_id, track_id, "play", timestamp))
            track_counts[track_id] += 1

    keep_tracks = {
        track_id for track_id, count in track_counts.items()
        if count >= min_track_plays
    }
    events = [event for event in events if event[1] in keep_tracks]
    tracks = {key: value for key, value in tracks.items() if key in keep_tracks}
    if not events or not tracks:
        raise ValueError(
            "No usable listening events remained. Check the TSV format and lower --min-track-plays."
        )

    events.sort(key=lambda event: (event[0], event[3], event[1]))
    with (output_dir / "tracks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_COLUMNS)
        writer.writeheader()
        writer.writerows(tracks[key] for key in sorted(tracks))
    with (output_dir / "interactions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "track_id", "event_type", "timestamp"])
        writer.writerows(events)

    return {
        "users": len({event[0] for event in events}),
        "tracks": len(tracks),
        "events": len(events),
        "track_csv_rows": len(tracks),
        "interaction_csv_rows": len(events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path, help="Extracted Last.fm 1K listening TSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-users", type=int, default=150, help="0 means no user limit (default: 150)")
    parser.add_argument("--max-events", type=int, default=250_000, help="0 means no event limit (default: 250000)")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--min-track-plays", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--demo-source-user-id", default="",
        help="Optional source user ID whose events should map to the local demo account ID.",
    )
    args = parser.parse_args()
    try:
        stats = import_lastfm(
            args.tsv, args.output_dir, max_users=args.max_users,
            max_events=args.max_events, chunksize=args.chunksize,
            min_track_plays=args.min_track_plays,
            demo_source_user_id=args.demo_source_user_id,
            random_state=args.random_state,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print("Created " + str(args.output_dir.resolve()))
    for key, value in stats.items():
        print(key + ": " + str(value))
    print("Imported records contain metadata only; add licensed audio separately for playback.")


if __name__ == "__main__":
    main()
