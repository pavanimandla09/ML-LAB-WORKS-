import unittest

import pandas as pd

from ml.evaluation import chronological_split, evaluate_ranker
from ml.recommender import HybridMusicRecommender


def sample_catalogue():
    return pd.DataFrame(
        [
            {"track_id": "t1", "title": "Bright Morning", "artist": "A", "genre": "Pop", "mood": "happy", "tags": "synth upbeat", "energy": 0.8, "danceability": 0.8, "valence": 0.9, "tempo": 120, "acousticness": 0.1, "instrumentalness": 0.0},
            {"track_id": "t2", "title": "Sunlit Road", "artist": "B", "genre": "Pop", "mood": "happy", "tags": "synth pop bright", "energy": 0.75, "danceability": 0.75, "valence": 0.85, "tempo": 115, "acousticness": 0.15, "instrumentalness": 0.0},
            {"track_id": "t3", "title": "Quiet Piano", "artist": "C", "genre": "Jazz", "mood": "calm", "tags": "piano mellow acoustic", "energy": 0.2, "danceability": 0.3, "valence": 0.4, "tempo": 75, "acousticness": 0.9, "instrumentalness": 0.1},
            {"track_id": "t4", "title": "Blue Note", "artist": "D", "genre": "Jazz", "mood": "calm", "tags": "jazz piano mellow", "energy": 0.3, "danceability": 0.4, "valence": 0.45, "tempo": 82, "acousticness": 0.8, "instrumentalness": 0.0},
            {"track_id": "t5", "title": "Night Circuit", "artist": "E", "genre": "Electronic", "mood": "intense", "tags": "techno electronic synth", "energy": 0.95, "danceability": 0.85, "valence": 0.4, "tempo": 135, "acousticness": 0.05, "instrumentalness": 0.4},
            {"track_id": "t6", "title": "Slow Orbit", "artist": "F", "genre": "Ambient", "mood": "dreamy", "tags": "ambient atmospheric pads", "energy": 0.15, "danceability": 0.2, "valence": 0.5, "tempo": 68, "acousticness": 0.8, "instrumentalness": 0.9},
        ]
    )


def sample_events():
    rows = [
        ("u1", "t1", "like", "2026-01-01T10:00:00Z"),
        ("u1", "t2", "play", "2026-01-02T10:00:00Z"),
        ("u1", "t3", "skip", "2026-01-03T10:00:00Z"),
        ("u2", "t1", "like", "2026-01-01T10:00:00Z"),
        ("u2", "t2", "complete", "2026-01-02T10:00:00Z"),
        ("u2", "t5", "play", "2026-01-03T10:00:00Z"),
        ("u3", "t3", "like", "2026-01-01T10:00:00Z"),
        ("u3", "t4", "complete", "2026-01-02T10:00:00Z"),
        ("u3", "t6", "play", "2026-01-03T10:00:00Z"),
        ("u4", "t5", "like", "2026-01-01T10:00:00Z"),
        ("u4", "t1", "play", "2026-01-02T10:00:00Z"),
        ("u4", "t2", "complete", "2026-01-03T10:00:00Z"),
    ]
    return pd.DataFrame(rows, columns=["user_id", "track_id", "event_type", "timestamp"])


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.tracks = sample_catalogue()
        self.events = sample_events()
        self.model = HybridMusicRecommender(self.tracks, self.events)

    def test_svd_is_fitted_for_multi_user_history(self):
        self.assertTrue(self.model.diagnostics()["svd_available"])
        self.assertGreaterEqual(self.model.diagnostics()["svd_components"], 1)

    def test_recommendations_exclude_previously_consumed_tracks(self):
        results = self.model.recommend("u1", limit=10, strategy="hybrid")
        ids = {item["track_id"] for item in results}
        self.assertNotIn("t1", ids)
        self.assertNotIn("t2", ids)
        self.assertNotIn("t3", ids)

    def test_content_similarity_returns_other_tracks(self):
        results = self.model.similar_tracks("t1", limit=5)
        ids = [item["track_id"] for item in results]
        self.assertNotIn("t1", ids)
        self.assertEqual(len(ids), 5)

    def test_cold_start_user_gets_a_fallback(self):
        results = self.model.recommend("brand-new-user", limit=4, strategy="hybrid")
        self.assertGreater(len(results), 0)
        self.assertTrue(all(item["reason"] for item in results))

    def test_supported_strategies_return_ranked_results(self):
        for strategy in ("content", "svd", "knn", "popularity", "hybrid"):
            results = self.model.recommend("u1", limit=4, strategy=strategy)
            self.assertLessEqual(len(results), 4)
            scores = [item["score"] for item in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_svd_and_knn_cold_start_use_popularity_fallback(self):
        for strategy in ("svd", "knn"):
            results = self.model.recommend("new-account", limit=3, strategy=strategy)
            self.assertGreater(len(results), 0)
            self.assertTrue(all(item["method"] == "popularity-fallback" for item in results))

    def test_empty_metadata_catalogue_does_not_crash(self):
        catalogue = pd.DataFrame([
            {"track_id": "x", "title": "", "artist": ""},
            {"track_id": "y", "title": "", "artist": ""},
        ])
        model = HybridMusicRecommender(catalogue, pd.DataFrame())
        self.assertEqual(model.diagnostics()["tracks"], 2)
        self.assertEqual(len(model.similar_tracks("x")), 1)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.recommend("u1", strategy="not-a-model")

    def test_chronological_split_keeps_training_history(self):
        train, test = chronological_split(self.events, test_fraction=0.2)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertTrue(set(train["user_id"]).issuperset(set(test["user_id"])))

    def test_ranking_evaluation_returns_expected_metrics(self):
        train, test = chronological_split(self.events, test_fraction=0.2)
        result = evaluate_ranker(
            self.tracks, train, test, strategy="hybrid", k=3
        )
        self.assertEqual(result["strategy"], "hybrid")
        for metric in ("precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", "catalogue_coverage"):
            self.assertGreaterEqual(result[metric], 0.0)
            self.assertLessEqual(result[metric], 1.0)


if __name__ == "__main__":
    unittest.main()
