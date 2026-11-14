"""Pretrained MiniLM semantic recommender for track metadata."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from ml.recommender import EVENT_WEIGHTS, _event_strength

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_FIELDS = ("title", "artist", "album", "genre", "mood", "tags")

def track_texts(tracks: pd.DataFrame) -> list[str]:
    fields = [name for name in TEXT_FIELDS if name in tracks.columns]
    if not fields:
        return [""] * len(tracks)
    frame = tracks[fields].fillna("").astype(str)
    return frame.apply(
        lambda row: " | ".join(
            f"{name}: {str(row[name]).strip()}"
            for name in fields if str(row[name]).strip()
        ), axis=1
    ).tolist()

class MiniLMSemanticRecommender:
    """Rank tracks by cosine similarity to a user's pretrained semantic profile."""
    def __init__(self, tracks: pd.DataFrame, interactions: pd.DataFrame | None = None, *, encoder: Any | None = None) -> None:
        if tracks is None or tracks.empty or "track_id" not in tracks.columns:
            raise ValueError("The track catalogue must contain track_id rows.")
        self.tracks = tracks.copy().fillna("")
        self.tracks["track_id"] = self.tracks["track_id"].astype(str)
        self.tracks = self.tracks.drop_duplicates("track_id").reset_index(drop=True)
        self.track_ids = self.tracks["track_id"].tolist()
        self.track_index = {track_id: index for index, track_id in enumerate(self.track_ids)}
        self.interactions = interactions.copy() if interactions is not None else pd.DataFrame()
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "MiniLM evaluation requires optional dependencies. Install with: "
                    "py -3.12 -m pip install -r requirements-ml.txt"
                ) from exc
            encoder = SentenceTransformer(MODEL_NAME)
        self.encoder = encoder
        self.track_embeddings = np.asarray(
            self.encoder.encode(
                track_texts(self.tracks), batch_size=64, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            ), dtype=np.float32
        )
        if self.track_embeddings.ndim != 2 or self.track_embeddings.shape[0] != len(self.track_ids):
            raise ValueError("The MiniLM encoder returned an unexpected embedding shape.")

    def _events_for_user(self, user_id: str) -> pd.DataFrame:
        required = {"user_id", "track_id", "event_type"}
        if self.interactions.empty or not required.issubset(self.interactions.columns):
            return pd.DataFrame()
        events = self.interactions.copy()
        events["user_id"] = events["user_id"].astype(str)
        events["track_id"] = events["track_id"].astype(str)
        events["event_type"] = events["event_type"].astype(str).str.lower()
        return events[(events["user_id"] == str(user_id)) & events["track_id"].isin(self.track_index)].copy()

    def _seen_track_ids(self, user_id: str) -> set[str]:
        events = self._events_for_user(user_id)
        return set(events["track_id"].astype(str)) if not events.empty else set()

    def score_vector(self, user_id: str) -> np.ndarray:
        events = self._events_for_user(user_id)
        if events.empty:
            return np.zeros(len(self.track_ids), dtype=float)
        positive = events[events["event_type"].isin(EVENT_WEIGHTS)].copy()
        if positive.empty:
            return np.zeros(len(self.track_ids), dtype=float)
        positive["weight"] = _event_strength(positive)
        indices = positive["track_id"].map(self.track_index).to_numpy(dtype=int)
        weights = positive["weight"].to_numpy(dtype=np.float32)
        profile = np.average(self.track_embeddings[indices], axis=0, weights=weights)
        profile = normalize(profile.reshape(1, -1), norm="l2")[0]
        return np.asarray(self.track_embeddings @ profile, dtype=float)

    def recommend(self, user_id: str, *, limit: int = 12, strategy: str = "minilm", exclude_seen: bool = True) -> list[dict[str, Any]]:
        scores = self.score_vector(str(user_id))
        if exclude_seen:
            seen = self._seen_track_ids(str(user_id))
            if seen:
                scores = scores.copy()
                scores[[self.track_index[item] for item in seen]] = -np.inf
        ranked = np.argsort(-scores, kind="stable")
        output = []
        for index in ranked:
            if not np.isfinite(scores[index]):
                continue
            output.append({
                "track_id": self.track_ids[int(index)], "score": float(scores[index]),
                "reason": "Semantic match to themes and artists in your listening history",
                "method": "minilm", "components": {"minilm": float(scores[index])},
            })
            if len(output) >= max(1, min(int(limit), 50)):
                break
        return output
