"""FastAPI service and same-origin web application for the music recommender."""
from __future__ import annotations

import os
import re
import secrets
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.audio import ensure_demo_audio
from app.db import AUDIO_DIR, PROJECT_DIR, connection, init_db
from app.security import (
    JWT_SECRET,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from ml.recommender import HybridMusicRecommender


STATIC_DIR = PROJECT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

security = HTTPBearer(auto_error=False)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a"}


@asynccontextmanager
async def lifespan(application: FastAPI):
    if os.getenv("APP_ENV", "development").lower() == "production":
        secret = JWT_SECRET.strip()
        if (
            len(secret.encode("utf-8")) < 32
            or secret == "local-development-secret-change-before-deploy"
            or "replace-this" in secret.lower()
        ):
            raise RuntimeError("Set a strong random JWT_SECRET of at least 32 bytes before running in production.")
    init_db()
    ensure_demo_audio()
    application.state.recommender = None
    yield


app = FastAPI(
    title="SoundScout Music Recommendation API",
    description="Hybrid music discovery using content-based filtering, SVD, item-KNN and popularity baselines.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=AUDIO_DIR), name="media")


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=1, max_length=72)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=24, pattern=r"^[A-Za-z0-9_.-]+$")
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=72)


class InteractionRequest(BaseModel):
    track_id: str = Field(min_length=1, max_length=80)
    event_type: Literal["play", "complete", "repeat", "like", "unlike", "skip"]
    duration_seconds: int = Field(default=0, ge=0, le=86400)


class PlaylistRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)


class PlaylistTrackRequest(BaseModel):
    track_id: str = Field(min_length=1, max_length=80)


def _public_user(row: sqlite3.Row | dict) -> dict:
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "email": row["email"],
    }


def _current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    try:
        payload = decode_access_token(credentials.credentials)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.") from exc
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid access token.")
    with connection() as conn:
        row = conn.execute(
            "SELECT user_id, username, email FROM users WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Account no longer exists.")
    return _public_user(row)


def _load_catalogue() -> pd.DataFrame:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM tracks ORDER BY title").fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def _load_interactions() -> pd.DataFrame:
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT user_id, track_id, event_type, duration_seconds, timestamp
            FROM interactions
            ORDER BY timestamp, interaction_id
            """
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def _get_recommender() -> HybridMusicRecommender:
    current = getattr(app.state, "recommender", None)
    if current is None:
        catalogue = _load_catalogue()
        events = _load_interactions()
        if catalogue.empty:
            raise HTTPException(status_code=503, detail="No tracks have been added to the catalogue.")
        current = HybridMusicRecommender(catalogue, events)
        app.state.recommender = current
    return current


def _track_payload(row: sqlite3.Row | dict) -> dict:
    result = dict(row)
    audio_file = Path(str(result.get("audio_file", ""))).name
    result["audio_url"] = f"/media/{quote(audio_file)}" if audio_file else ""
    return result


def _recommendation_payload(user_id: str, strategy: str, limit: int) -> list[dict]:
    model = _get_recommender()
    ranked = model.recommend(user_id, limit=limit, strategy=strategy)
    ids = [item["track_id"] for item in ranked]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM tracks WHERE track_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {row["track_id"]: _track_payload(row) for row in rows}
    results = []
    for item in ranked:
        track = by_id.get(item["track_id"])
        if track:
            results.append({**track, **item})
    return results


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        with connection() as conn:
            track_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            event_count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        model = _get_recommender()
        return {
            "status": "ok",
            "tracks": track_count,
            "interactions": event_count,
            "recommendation_engine": model.diagnostics(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The service is not ready.") from exc


@app.post("/api/auth/register")
def register(payload: RegisterRequest):
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", payload.email.strip()):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(payload.password.encode("utf-8")) > 72:
        raise HTTPException(status_code=422, detail="Password must be at most 72 UTF-8 bytes.")
    user_id = str(uuid.uuid4())
    username = payload.username.strip().lower()
    email = payload.email.strip().lower()
    try:
        password_hash = hash_password(payload.password)
        with connection() as conn:
            duplicate = conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
                (username, email),
            ).fetchone()
            if duplicate is not None:
                raise HTTPException(status_code=409, detail="That username or email is already registered.")
            conn.execute(
                """
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, username, email, password_hash),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409, detail="That username or email is already registered."
        ) from exc
    user = {"user_id": user_id, "username": username, "email": email}
    return {"access_token": create_access_token(user_id, user["username"]), "token_type": "bearer", "user": user}


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    with connection() as conn:
        row = conn.execute(
            """
            SELECT user_id, username, email, password_hash
            FROM users WHERE username = ? COLLATE NOCASE
            """,
            (payload.username.strip(),),
        ).fetchone()
    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    user = _public_user(row)
    return {
        "access_token": create_access_token(user["user_id"], user["username"]),
        "token_type": "bearer",
        "user": user,
    }


@app.get("/api/auth/me")
def me(current_user: dict = Depends(_current_user)):
    return current_user


@app.get("/api/tracks")
def list_tracks(
    q: str = Query(default="", max_length=100),
    genre: str = Query(default="", max_length=40),
    limit: int = Query(default=100, ge=1, le=250),
):
    sql = "SELECT * FROM tracks"
    clauses = []
    params: list = []
    if q.strip():
        pattern = f"%{q.strip()}%"
        clauses.append("(title LIKE ? OR artist LIKE ? OR album LIKE ? OR tags LIKE ?)")
        params.extend([pattern, pattern, pattern, pattern])
    if genre.strip() and genre.lower() != "all":
        clauses.append("genre = ? COLLATE NOCASE")
        params.append(genre.strip())
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY title LIMIT ?"
    params.append(limit)
    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_track_payload(row) for row in rows]


@app.get("/api/genres")
def list_genres():
    with connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT genre FROM tracks WHERE genre <> '' ORDER BY genre"
        ).fetchall()
    return [row["genre"] for row in rows]


@app.get("/api/recommendations/similar/{track_id}")
def similar_tracks(
    track_id: str,
    limit: int = Query(default=8, ge=1, le=30),
):
    model = _get_recommender()
    try:
        ranked = model.similar_tracks(track_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Track not found.") from exc
    ids = [item["track_id"] for item in ranked]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM tracks WHERE track_id IN ({placeholders})", ids
        ).fetchall()
    by_id = {row["track_id"]: _track_payload(row) for row in rows}
    return [{**by_id[item["track_id"]], **item} for item in ranked if item["track_id"] in by_id]


@app.get("/api/recommendations/{strategy}")
def recommendations(
    strategy: Literal["hybrid", "content", "svd", "knn", "popularity"],
    limit: int = Query(default=12, ge=1, le=50),
    current_user: dict = Depends(_current_user),
):
    return _recommendation_payload(current_user["user_id"], strategy, limit)


@app.post("/api/interactions")
def record_interaction(
    payload: InteractionRequest,
    current_user: dict = Depends(_current_user),
):
    with connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM tracks WHERE track_id = ?", (payload.track_id,)
        ).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="Track not found.")
        if payload.event_type == "unlike":
            conn.execute(
                """
                DELETE FROM interactions
                WHERE user_id = ? AND track_id = ? AND event_type = 'like'
                """,
                (current_user["user_id"], payload.track_id),
            )
        conn.execute(
            """
            INSERT INTO interactions (
                user_id, track_id, event_type, duration_seconds
            ) VALUES (?, ?, ?, ?)
            """,
            (
                current_user["user_id"],
                payload.track_id,
                payload.event_type,
                payload.duration_seconds,
            ),
        )
    app.state.recommender = None
    return {"recorded": True, "event_type": payload.event_type}


@app.get("/api/library/liked")
def liked_tracks(current_user: dict = Depends(_current_user)):
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT t.*
            FROM tracks t
            WHERE EXISTS (
                SELECT 1 FROM interactions i
                WHERE i.user_id = ? AND i.track_id = t.track_id
                  AND i.event_type = 'like'
                  AND i.interaction_id > COALESCE((
                      SELECT MAX(u.interaction_id) FROM interactions u
                      WHERE u.user_id = ? AND u.track_id = t.track_id
                        AND u.event_type = 'unlike'
                  ), 0)
            )
            ORDER BY t.title
            """,
            (current_user["user_id"], current_user["user_id"]),
        ).fetchall()
    return [_track_payload(row) for row in rows]


@app.get("/api/history")
def listening_history(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(_current_user),
):
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT i.event_type, i.duration_seconds, i.timestamp, t.*
            FROM interactions i JOIN tracks t ON t.track_id = i.track_id
            WHERE i.user_id = ?
            ORDER BY i.interaction_id DESC LIMIT ?
            """,
            (current_user["user_id"], limit),
        ).fetchall()
    return [
        {
            **_track_payload(row),
            "event_type": row["event_type"],
            "duration_seconds": row["duration_seconds"],
            "played_at": row["timestamp"],
        }
        for row in rows
    ]


@app.get("/api/playlists")
def list_playlists(current_user: dict = Depends(_current_user)):
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT p.*, COUNT(pt.track_id) AS track_count
            FROM playlists p LEFT JOIN playlist_tracks pt
              ON pt.playlist_id = p.playlist_id
            WHERE p.user_id = ?
            GROUP BY p.playlist_id
            ORDER BY p.created_at DESC
            """,
            (current_user["user_id"],),
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/playlists", status_code=201)
def create_playlist(
    payload: PlaylistRequest,
    current_user: dict = Depends(_current_user),
):
    playlist_id = str(uuid.uuid4())
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO playlists (playlist_id, user_id, title, description)
            VALUES (?, ?, ?, ?)
            """,
            (playlist_id, current_user["user_id"], payload.title.strip(), payload.description.strip()),
        )
    return {"playlist_id": playlist_id, "title": payload.title.strip(), "track_count": 0}


@app.get("/api/playlists/{playlist_id}")
def playlist_detail(
    playlist_id: str,
    current_user: dict = Depends(_current_user),
):
    with connection() as conn:
        playlist = conn.execute(
            "SELECT * FROM playlists WHERE playlist_id = ? AND user_id = ?",
            (playlist_id, current_user["user_id"]),
        ).fetchone()
        if playlist is None:
            raise HTTPException(status_code=404, detail="Playlist not found.")
        rows = conn.execute(
            """
            SELECT t.* FROM playlist_tracks pt
            JOIN tracks t ON t.track_id = pt.track_id
            WHERE pt.playlist_id = ? ORDER BY pt.position, pt.added_at
            """,
            (playlist_id,),
        ).fetchall()
    return {"playlist": dict(playlist), "tracks": [_track_payload(row) for row in rows]}


@app.post("/api/playlists/{playlist_id}/tracks", status_code=201)
def add_playlist_track(
    playlist_id: str,
    payload: PlaylistTrackRequest,
    current_user: dict = Depends(_current_user),
):
    with connection() as conn:
        playlist = conn.execute(
            "SELECT playlist_id FROM playlists WHERE playlist_id = ? AND user_id = ?",
            (playlist_id, current_user["user_id"]),
        ).fetchone()
        track = conn.execute(
            "SELECT track_id FROM tracks WHERE track_id = ?", (payload.track_id,)
        ).fetchone()
        if playlist is None:
            raise HTTPException(status_code=404, detail="Playlist not found.")
        if track is None:
            raise HTTPException(status_code=404, detail="Track not found.")
        position = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM playlist_tracks WHERE playlist_id = ?",
            (playlist_id,),
        ).fetchone()[0]
        try:
            conn.execute(
                """
                INSERT INTO playlist_tracks (playlist_id, track_id, position)
                VALUES (?, ?, ?)
                """,
                (playlist_id, payload.track_id, position),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Track is already in this playlist.") from exc
        conn.execute(
            """
            INSERT INTO interactions (user_id, track_id, event_type)
            VALUES (?, ?, 'playlist_add')
            """,
            (current_user["user_id"], payload.track_id),
        )
    app.state.recommender = None
    return {"added": True}


@app.delete("/api/playlists/{playlist_id}/tracks/{track_id}")
def remove_playlist_track(
    playlist_id: str,
    track_id: str,
    current_user: dict = Depends(_current_user),
):
    with connection() as conn:
        playlist = conn.execute(
            "SELECT playlist_id FROM playlists WHERE playlist_id = ? AND user_id = ?",
            (playlist_id, current_user["user_id"]),
        ).fetchone()
        if playlist is None:
            raise HTTPException(status_code=404, detail="Playlist not found.")
        deleted = conn.execute(
            "DELETE FROM playlist_tracks WHERE playlist_id = ? AND track_id = ?",
            (playlist_id, track_id),
        ).rowcount
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Track is not in this playlist.")
    return {"removed": True}


@app.post("/api/admin/tracks", status_code=201)
async def upload_track(
    title: str = Form(..., min_length=1, max_length=120),
    artist: str = Form(..., min_length=1, max_length=120),
    genre: str = Form(..., min_length=1, max_length=40),
    album: str = Form(default="", max_length=120),
    mood: str = Form(default="", max_length=80),
    tags: str = Form(default="", max_length=500),
    energy: float = Form(default=0.5, ge=0, le=1),
    danceability: float = Form(default=0.5, ge=0, le=1),
    valence: float = Form(default=0.5, ge=0, le=1),
    tempo: float = Form(default=100, ge=30, le=250),
    acousticness: float = Form(default=0.5, ge=0, le=1),
    instrumentalness: float = Form(default=0.0, ge=0, le=1),
    audio: UploadFile = File(...),
    x_admin_key: str | None = Header(default=None),
):
    configured_key = os.getenv("ADMIN_API_KEY", "")
    if not configured_key:
        raise HTTPException(status_code=503, detail="Audio uploads are disabled until ADMIN_API_KEY is configured.")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, configured_key):
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    extension = Path(audio.filename or "").suffix.lower()
    if extension not in {".mp3", ".wav", ".ogg", ".m4a"}:
        raise HTTPException(status_code=415, detail="Upload an MP3, WAV, OGG or M4A audio file.")
    content = await audio.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file must be between 1 byte and 25 MB.")
    header = content[:16]
    signatures = {
        ".wav": len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE",
        ".ogg": header.startswith(b"OggS"),
        ".m4a": len(header) >= 8 and header[4:8] == b"ftyp",
        ".mp3": header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
        ),
    }
    if not signatures.get(extension, False):
        raise HTTPException(status_code=415, detail="The uploaded file does not match its audio extension.")
    track_id = f"upload-{uuid.uuid4().hex[:12]}"
    audio_filename = f"{track_id}{extension}"
    destination = AUDIO_DIR / audio_filename
    destination.write_bytes(content)
    try:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO tracks (
                    track_id, title, artist, album, genre, mood, tags,
                    energy, danceability, valence, tempo, acousticness,
                    instrumentalness, audio_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track_id, title.strip(), artist.strip(), album.strip(), genre.strip(),
                    mood.strip(), tags.strip(), energy, danceability, valence, tempo,
                    acousticness, instrumentalness, audio_filename,
                ),
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    app.state.recommender = None
    with connection() as conn:
        row = conn.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
    return _track_payload(row)
