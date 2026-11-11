"""Explainable music recommendation algorithms and hybrid ranking.

Collaborative filtering is trained on positive implicit interactions. SVD is
implemented with scikit-learn TruncatedSVD over a log-weighted sparse
user-track matrix; it is not a rating-prediction model.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


EVENT_WEIGHTS = {
    "play": 1.0,
    "complete": 1.8,
    "repeat": 1.5,
    "like": 3.0,
    "playlist_add": 2.5,
}
NEGATIVE_EVENTS = {"skip", "unlike"}
NUMERIC_FEATURES = (
    "energy",
    "danceability",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
)


def _minmax(values: np.ndarray) -> np.ndarray:
    """Return a finite [0, 1] scaled vector; constants map to zero."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high - low < 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


class HybridMusicRecommender:
    """Content, SVD, item-KNN and popularity recommenders over a catalogue."""

    def __init__(
        self,
        tracks: pd.DataFrame,
        interactions: pd.DataFrame | None = None,
        *,
        random_state: int = 42,
        max_svd_components: int = 20,
        knn_neighbors: int = 20,
    ) -> None:
        if tracks is None or tracks.empty:
            raise ValueError("The track catalogue must contain at least one track.")
        if "track_id" not in tracks.columns:
            raise ValueError("The track catalogue must contain a track_id column.")

        self.tracks = tracks.copy().fillna("")
        self.tracks["track_id"] = self.tracks["track_id"].astype(str)
        self.tracks = self.tracks.drop_duplicates("track_id", keep="first").reset_index(drop=True)
        self.track_ids = self.tracks["track_id"].tolist()
        self.track_index = {track_id: i for i, track_id in enumerate(self.track_ids)}
        self.interactions = (
            interactions.copy() if interactions is not None else pd.DataFrame()
        )
        self.random_state = random_state
        self.max_svd_components = max_svd_components
        self.knn_neighbors = knn_neighbors
        self.svd_model: TruncatedSVD | None = None
        self.svd_user_factors: np.ndarray | None = None
        self.user_index: dict[str, int] = {}
        self.user_ids: list[str] = []
        self.user_item: csr_matrix = csr_matrix((0, len(self.track_ids)), dtype=float)
        self.item_user: csr_matrix = csr_matrix((len(self.track_ids), 0), dtype=float)
        self.knn_model: NearestNeighbors | None = None

        self._fit_content()
        self._fit_interactions()

    def _fit_content(self) -> None:
        text_columns = [
            col for col in ("title", "artist", "album", "genre", "mood", "tags")
            if col in self.tracks.columns
        ]
        if text_columns:
            text = self.tracks[text_columns].astype(str).agg(" ".join, axis=1)
        else:
            text = pd.Series([""] * len(self.tracks))
        text = text.str.lower().str.replace(r"[_|,/]", " ", regex=True)

        self.text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=12000,
            stop_words="english",
            sublinear_tf=True,
        )
        try:
            text_matrix = self.text_vectorizer.fit_transform(text)
        except ValueError:
            # All metadata fields may be blank in a partially imported catalogue.
            text_matrix = csr_matrix((len(self.tracks), 0), dtype=float)

        numeric_columns = [c for c in NUMERIC_FEATURES if c in self.tracks.columns]
        numeric_matrix = csr_matrix((len(self.tracks), 0), dtype=float)
        if numeric_columns:
            numeric = self.tracks[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0.0)
            scaler = StandardScaler()
            numeric_values = scaler.fit_transform(numeric)
            numeric_values = np.clip(numeric_values, -3.0, 3.0)
            numeric_matrix = csr_matrix(numeric_values)

        if text_matrix.shape[1] and numeric_matrix.shape[1]:
            self.content_matrix = hstack(
                [text_matrix * 0.85, numeric_matrix * 0.15], format="csr"
            )
        elif text_matrix.shape[1]:
            self.content_matrix = text_matrix.tocsr()
        elif numeric_matrix.shape[1]:
            self.content_matrix = numeric_matrix
        else:
            self.content_matrix = csr_matrix((len(self.tracks), 1), dtype=float)
        self.content_matrix = normalize(self.content_matrix, norm="l2", axis=1)

    def _fit_interactions(self) -> None:
        if self.interactions.empty or not {"user_id", "track_id", "event_type"}.issubset(
            self.interactions.columns
        ):
            self._fit_empty_interactions()
            return

        events = self.interactions.copy()
        events["user_id"] = events["user_id"].astype(str)
        events["track_id"] = events["track_id"].astype(str)
        events["event_type"] = events["event_type"].astype(str).str.lower()
        events = events[events["track_id"].isin(self.track_index)]
        positive = events[events["event_type"].isin(EVENT_WEIGHTS)].copy()

        if positive.empty:
            self._fit_empty_interactions()
            return

        positive["weight"] = positive["event_type"].map(EVENT_WEIGHTS).astype(float)
        grouped = positive.groupby(["user_id", "track_id"], as_index=False)["weight"].sum()
        self.user_ids = sorted(grouped["user_id"].unique().tolist())
        self.user_index = {user_id: i for i, user_id in enumerate(self.user_ids)}

        rows = grouped["user_id"].map(self.user_index).to_numpy(dtype=int)
        cols = grouped["track_id"].map(self.track_index).to_numpy(dtype=int)
        vals = np.log1p(grouped["weight"].to_numpy(dtype=float))
        self.user_item = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_ids), len(self.track_ids)),
            dtype=float,
        )
        self.item_user = normalize(self.user_item.T.tocsr(), norm="l2", axis=1)

        min_dimension = min(self.user_item.shape)
        component_count = min(self.max_svd_components, min_dimension - 1)
        if component_count >= 1 and self.user_item.nnz >= 2:
            self.svd_model = TruncatedSVD(
                n_components=component_count,
                n_iter=7,
                random_state=self.random_state,
            )
            self.svd_user_factors = self.svd_model.fit_transform(self.user_item)

        if len(self.track_ids) >= 2 and self.item_user.shape[1] >= 1:
            neighbor_count = min(self.knn_neighbors + 1, len(self.track_ids))
            self.knn_model = NearestNeighbors(
                n_neighbors=neighbor_count,
                metric="cosine",
                algorithm="brute",
            )
            self.knn_model.fit(self.item_user)

    def _fit_empty_interactions(self) -> None:
        self.user_ids = []
        self.user_index = {}
        self.user_item = csr_matrix((0, len(self.track_ids)), dtype=float)
        self.item_user = csr_matrix((len(self.track_ids), 0), dtype=float)
        self.svd_model = None
        self.svd_user_factors = None
        self.knn_model = None

    def _events_for_user(self, user_id: str) -> pd.DataFrame:
        if self.interactions.empty:
            return pd.DataFrame()
        if not {"user_id", "track_id", "event_type"}.issubset(self.interactions.columns):
            return pd.DataFrame()
        events = self.interactions.copy()
        events["user_id"] = events["user_id"].astype(str)
        events["track_id"] = events["track_id"].astype(str)
        events["event_type"] = events["event_type"].astype(str).str.lower()
        return events[
            (events["user_id"] == str(user_id))
            & events["track_id"].isin(self.track_index)
        ].copy()

    def _seen_track_ids(self, user_id: str) -> set[str]:
        events = self._events_for_user(user_id)
        if events.empty:
            return set()
        # A skip is still a consumed track for the current session; do not
        # immediately recommend it again. An unlike does not erase listening.
        return set(events["track_id"].astype(str).tolist())

    def _content_scores(
        self, user_id: str, seed_track_id: str | None = None
    ) -> np.ndarray:
        if seed_track_id is not None:
            seed_index = self.track_index.get(str(seed_track_id))
            if seed_index is None:
                return np.zeros(len(self.track_ids), dtype=float)
            profile = self.content_matrix[seed_index]
        else:
            events = self._events_for_user(user_id)
            if events.empty:
                return np.zeros(len(self.track_ids), dtype=float)
            positive = events[events["event_type"].isin(EVENT_WEIGHTS)].copy()
            if positive.empty:
                return np.zeros(len(self.track_ids), dtype=float)
            positive["weight"] = positive["event_type"].map(EVENT_WEIGHTS).astype(float)
            if "timestamp" in positive.columns:
                positive = positive.sort_values("timestamp")
            positive = positive.tail(50)
            indices = positive["track_id"].map(self.track_index).to_numpy(dtype=int)
            weights = positive["weight"].to_numpy(dtype=float)
            profile = csr_matrix(
                weights.reshape(1, -1)
            ) @ self.content_matrix[indices]
            profile = normalize(profile, norm="l2", axis=1)
        return (profile @ self.content_matrix.T).toarray().ravel()

    def _svd_scores(self, user_id: str) -> np.ndarray | None:
        if (
            self.svd_model is None
            or self.svd_user_factors is None
            or str(user_id) not in self.user_index
        ):
            return None
        user_row = self.user_index[str(user_id)]
        latent_user = self.svd_user_factors[user_row]
        return np.asarray(latent_user @ self.svd_model.components_, dtype=float)

    def _knn_scores(self, user_id: str) -> np.ndarray:
        scores = np.zeros(len(self.track_ids), dtype=float)
        if self.knn_model is None:
            return scores
        events = self._events_for_user(user_id)
        if events.empty:
            return scores
        positive = events[events["event_type"].isin(EVENT_WEIGHTS)].copy()
        if positive.empty:
            return scores
        positive["weight"] = positive["event_type"].map(EVENT_WEIGHTS).astype(float)
        grouped = positive.groupby("track_id")["weight"].sum()
        for track_id, weight in grouped.items():
            source_index = self.track_index.get(str(track_id))
            if source_index is None:
                continue
            distances, neighbors = self.knn_model.kneighbors(
                self.item_user[source_index], return_distance=True
            )
            for distance, neighbor_index in zip(distances[0], neighbors[0]):
                if int(neighbor_index) == source_index:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                scores[int(neighbor_index)] += similarity * float(weight)
        return scores

    def _popularity_scores(self) -> np.ndarray:
        scores = np.zeros(len(self.track_ids), dtype=float)
        if self.interactions.empty:
            return scores
        if not {"track_id", "event_type"}.issubset(self.interactions.columns):
            return scores
        events = self.interactions.copy()
        events["track_id"] = events["track_id"].astype(str)
        events["event_type"] = events["event_type"].astype(str).str.lower()
        events = events[
            events["track_id"].isin(self.track_index)
            & events["event_type"].isin(EVENT_WEIGHTS)
        ]
        if events.empty:
            return scores
        grouped = events.groupby("track_id").agg(
            total=("event_type", "size"),
            listeners=("user_id", "nunique") if "user_id" in events.columns else ("event_type", "size"),
            weighted=("event_type", lambda values: sum(EVENT_WEIGHTS[x] for x in values)),
        )
        for track_id, row in grouped.iterrows():
            idx = self.track_index.get(str(track_id))
            if idx is not None:
                scores[idx] = (
                    np.log1p(float(row["weighted"]))
                    + 0.35 * np.log1p(float(row["listeners"]))
                )
        return scores

    def _all_score_vectors(
        self, user_id: str, seed_track_id: str | None = None
    ) -> dict[str, np.ndarray]:
        content = _minmax(self._content_scores(user_id, seed_track_id))
        popularity = _minmax(self._popularity_scores())
        collaborative_raw = self._svd_scores(user_id)
        collaborative = (
            _minmax(collaborative_raw)
            if collaborative_raw is not None
            else np.zeros(len(self.track_ids), dtype=float)
        )
        knn = _minmax(self._knn_scores(user_id))
        return {
            "content": content,
            "svd": collaborative,
            "knn": knn,
            "popularity": popularity,
        }

    def recommend(
        self,
        user_id: str,
        *,
        limit: int = 12,
        strategy: str = "hybrid",
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Return ranked tracks and the score/reason for each recommendation."""
        allowed = {"hybrid", "content", "svd", "knn", "popularity"}
        if strategy not in allowed:
            raise ValueError(f"Unknown strategy '{strategy}'. Use one of {sorted(allowed)}.")
        limit = max(1, min(int(limit), 50))
        vectors = self._all_score_vectors(str(user_id))
        user_events = self._events_for_user(str(user_id))
        has_positive_history = (
            not user_events.empty
            and user_events["event_type"].isin(EVENT_WEIGHTS).any()
        )
        fallback_to_popularity = False
        if strategy == "hybrid":
            # SVD is unavailable for a cold-start user; the remaining weights
            # are re-normalised rather than treating unavailable CF as zero.
            weights = {"svd": 0.45, "content": 0.40, "popularity": 0.15}
            if str(user_id) not in self.user_index or self.svd_model is None:
                weights = {"content": 0.55, "popularity": 0.45}
            score = sum(vectors[key] * weight for key, weight in weights.items())
        elif strategy == "svd" and (
            self.svd_model is None or str(user_id) not in self.user_index
        ):
            score = vectors["popularity"]
            fallback_to_popularity = True
        elif strategy == "knn" and not has_positive_history:
            score = vectors["popularity"]
            fallback_to_popularity = True
        elif strategy == "content" and not np.any(vectors["content"]):
            score = vectors["popularity"]
            fallback_to_popularity = True
        else:
            score = vectors[strategy]

        seen = self._seen_track_ids(str(user_id)) if exclude_seen else set()
        ranked = np.argsort(-score, kind="stable")
        results: list[dict[str, Any]] = []
        has_history = has_positive_history
        for idx in ranked:
            track_id = self.track_ids[int(idx)]
            if track_id in seen:
                continue
            content_score = float(vectors["content"][idx])
            cf_score = float(vectors["svd"][idx])
            pop_score = float(vectors["popularity"][idx])
            if not has_history or fallback_to_popularity:
                reason = "Popular starting points while this model learns your listening profile"
            elif strategy == "content":
                reason = "Similar genre, mood or sound to tracks in your listening history"
            elif strategy in {"svd", "knn"}:
                reason = "Listeners with similar music taste also enjoyed this track"
            elif strategy == "popularity":
                reason = "Popular with listeners in this music catalogue"
            elif content_score >= cf_score and content_score > 0.2:
                reason = "Similar genre, mood or sound to tracks in your listening history"
            elif cf_score > 0.1:
                reason = "Listeners with similar music taste also enjoyed this track"
            else:
                reason = "Popular with listeners in this music catalogue"
            results.append(
                {
                    "track_id": track_id,
                    "score": float(score[idx]),
                    "reason": reason,
                    "method": "popularity-fallback" if fallback_to_popularity else strategy,
                    "components": {
                        "content": content_score,
                        "svd": cf_score,
                        "knn": float(vectors["knn"][idx]),
                        "popularity": pop_score,
                    },
                }
            )
            if len(results) >= limit:
                break
        return results

    def similar_tracks(
        self, track_id: str, *, limit: int = 8
    ) -> list[dict[str, Any]]:
        """Find tracks similar to a seed using content vectors."""
        seed_index = self.track_index.get(str(track_id))
        if seed_index is None:
            raise KeyError(f"Unknown track_id: {track_id}")
        similarities = (
            self.content_matrix[seed_index] @ self.content_matrix.T
        ).toarray().ravel()
        ranked = np.argsort(-similarities, kind="stable")
        results = []
        for idx in ranked:
            if int(idx) == seed_index:
                continue
            results.append(
                {
                    "track_id": self.track_ids[int(idx)],
                    "score": float(similarities[idx]),
                    "reason": "Shares musical metadata and audio characteristics with the selected track",
                    "method": "content",
                    "components": {"content": float(similarities[idx])},
                }
            )
            if len(results) >= max(1, min(int(limit), 50)):
                break
        return results

    def diagnostics(self) -> dict[str, Any]:
        return {
            "tracks": len(self.track_ids),
            "interaction_users": len(self.user_ids),
            "interaction_cells": int(self.user_item.nnz),
            "svd_available": self.svd_model is not None,
            "svd_components": (
                int(self.svd_model.n_components) if self.svd_model is not None else 0
            ),
            "knn_available": self.knn_model is not None,
            "content_features": int(self.content_matrix.shape[1]),
        }
