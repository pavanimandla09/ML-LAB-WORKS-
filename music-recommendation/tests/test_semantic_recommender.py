import unittest
import numpy as np
import pandas as pd

from ml.semantic_recommender import MiniLMSemanticRecommender, track_texts


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "bright morning" in lowered or "sunlit road" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "quiet piano" in lowered or "blue note" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class MiniLMSemanticTests(unittest.TestCase):
    def setUp(self):
        self.tracks = pd.DataFrame([
            {"track_id": "a", "title": "Bright Morning", "artist": "A", "genre": "Pop"},
            {"track_id": "b", "title": "Sunlit Road", "artist": "B", "genre": "Pop"},
            {"track_id": "c", "title": "Quiet Piano", "artist": "C", "genre": "Jazz"},
            {"track_id": "d", "title": "Blue Note", "artist": "D", "genre": "Jazz"},
            {"track_id": "e", "title": "Night Circuit", "artist": "E", "genre": "Electronic"},
        ])
        self.events = pd.DataFrame([
            {"user_id": "u1", "track_id": "a", "event_type": "like"},
        ])

    def test_track_text_uses_available_metadata(self):
        texts = track_texts(self.tracks)
        self.assertIn("title: Bright Morning", texts[0])
        self.assertIn("genre: Pop", texts[0])

    def test_semantic_recommender_ranks_related_metadata_and_excludes_seen(self):
        model = MiniLMSemanticRecommender(self.tracks, self.events, encoder=FakeEncoder())
        recommendations = model.recommend("u1", limit=3)
        ids = [item["track_id"] for item in recommendations]
        self.assertNotIn("a", ids)
        self.assertEqual(ids[0], "b")
        self.assertEqual(recommendations[0]["method"], "minilm")


if __name__ == "__main__":
    unittest.main()
