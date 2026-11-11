"""SQLite persistence and deterministic demo-data seeding."""
from __future__ import annotations

import csv
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import bcrypt
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")
DATA_DIR = PROJECT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRACKS_CSV = DATA_DIR / "tracks.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"

_raw_db_path = Path(os.getenv("DATABASE_PATH", "data/music.db"))
DATABASE_PATH = _raw_db_path if _raw_db_path.is_absolute() else PROJECT_DIR / _raw_db_path


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Yield a transactional connection and always close it."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                album TEXT NOT NULL DEFAULT '',
                genre TEXT NOT NULL DEFAULT 'Unknown',
                mood TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                energy REAL NOT NULL DEFAULT 0,
                danceability REAL NOT NULL DEFAULT 0,
                valence REAL NOT NULL DEFAULT 0,
                tempo REAL NOT NULL DEFAULT 0,
                acousticness REAL NOT NULL DEFAULT 0,
                instrumentalness REAL NOT NULL DEFAULT 0,
                audio_file TEXT NOT NULL DEFAULT '',
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(track_id) REFERENCES tracks(track_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS playlists (
                playlist_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS playlist_tracks (
                playlist_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                position INTEGER NOT NULL DEFAULT 0,
                added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(playlist_id, track_id),
                FOREIGN KEY(playlist_id) REFERENCES playlists(playlist_id) ON DELETE CASCADE,
                FOREIGN KEY(track_id) REFERENCES tracks(track_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_user_time
                ON interactions(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_interactions_track
                ON interactions(track_id);
            CREATE INDEX IF NOT EXISTS idx_playlist_tracks_position
                ON playlist_tracks(playlist_id, position);
            """
        )

        track_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        if track_count == 0 and TRACKS_CSV.exists():
            with TRACKS_CSV.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = []
                columns = (
                    "track_id", "title", "artist", "album", "genre", "mood", "tags",
                    "energy", "danceability", "valence", "tempo", "acousticness",
                    "instrumentalness", "audio_file",
                )
                for raw in reader:
                    row = dict(raw)
                    for column in columns:
                        if column not in row:
                            row[column] = ""
                    for number in (
                        "energy", "danceability", "valence", "tempo",
                        "acousticness", "instrumentalness",
                    ):
                        row[number] = float(row[number] or 0)
                    rows.append(tuple(row[column] for column in columns))
            conn.executemany(
                """
                INSERT OR IGNORE INTO tracks (
                    track_id, title, artist, album, genre, mood, tags,
                    energy, danceability, valence, tempo, acousticness,
                    instrumentalness, audio_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        interaction_count = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]
        if interaction_count == 0 and INTERACTIONS_CSV.exists():
            with INTERACTIONS_CSV.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                seed_events = [
                    (
                        row["user_id"],
                        row["track_id"],
                        row["event_type"],
                        0,
                        row["timestamp"],
                    )
                    for row in reader
                    if row.get("user_id") and row.get("track_id") and row.get("event_type")
                ]
            conn.executemany(
                """
                INSERT INTO interactions (
                    user_id, track_id, event_type, duration_seconds, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                seed_events,
            )

        demo_user = conn.execute(
            "SELECT user_id FROM users WHERE username = ?", ("demo",)
        ).fetchone()
        if demo_user is None:
            password_hash = bcrypt.hashpw(
                b"music12345", bcrypt.gensalt(rounds=12)
            )
            conn.execute(
                """
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
                """,
                ("demo", "demo", "demo@example.local", password_hash),
            )
