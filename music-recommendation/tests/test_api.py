"""HTTP-level smoke tests for authentication, catalogue and playlist flows."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import db
from app.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_database_path = db.DATABASE_PATH
        db.DATABASE_PATH = Path(cls.temp_dir.name) / "api-tests.sqlite3"
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        response = cls.client.post(
            "/api/auth/login",
            json={"username": "demo", "password": "music12345"},
        )
        if response.status_code != 200:
            raise RuntimeError("The seeded demo account could not sign in: " + response.text)
        cls.token = response.json()["access_token"]
        cls.headers = {"Authorization": "Bearer " + cls.token}

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        db.DATABASE_PATH = cls.original_database_path
        cls.temp_dir.cleanup()

    def test_health_endpoint_reports_seeded_catalogue(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertGreaterEqual(payload["tracks"], 20)
        self.assertTrue(payload["recommendation_engine"]["svd_available"])

    def test_recommendations_require_authentication(self):
        response = self.client.get("/api/recommendations/hybrid")
        self.assertEqual(response.status_code, 401)

    def test_all_recommendation_strategies_return_tracks(self):
        for strategy in ("hybrid", "content", "svd", "knn", "popularity"):
            response = self.client.get(
                "/api/recommendations/" + strategy,
                headers=self.headers,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsInstance(response.json(), list)
            self.assertGreater(len(response.json()), 0)
            self.assertTrue(all("audio_url" in item for item in response.json()))

    def test_demo_account_uses_its_seeded_svd_profile(self):
        response = self.client.get(
            "/api/recommendations/svd", headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(item["components"]["svd"] > 0 for item in response.json()))

    def test_search_and_similar_tracks(self):
        search = self.client.get("/api/tracks?q=Neon")
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.json()[0]["track_id"], "track-01")

        similar = self.client.get("/api/recommendations/similar/track-01")
        self.assertEqual(similar.status_code, 200)
        self.assertTrue(all(item["track_id"] != "track-01" for item in similar.json()))

    def test_playlist_create_add_and_read(self):
        created = self.client.post(
            "/api/playlists",
            headers=self.headers,
            json={"title": "Test playlist", "description": "Created by API tests"},
        )
        self.assertEqual(created.status_code, 201, created.text)
        playlist_id = created.json()["playlist_id"]

        added = self.client.post(
            "/api/playlists/" + playlist_id + "/tracks",
            headers=self.headers,
            json={"track_id": "track-10"},
        )
        self.assertEqual(added.status_code, 201, added.text)

        detail = self.client.get(
            "/api/playlists/" + playlist_id,
            headers=self.headers,
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["tracks"][0]["track_id"], "track-10")

    def test_like_and_unlike_update_library(self):
        liked = self.client.post(
            "/api/interactions",
            headers=self.headers,
            json={"track_id": "track-12", "event_type": "like"},
        )
        self.assertEqual(liked.status_code, 200)
        library = self.client.get("/api/library/liked", headers=self.headers)
        self.assertIn("track-12", [item["track_id"] for item in library.json()])

        unliked = self.client.post(
            "/api/interactions",
            headers=self.headers,
            json={"track_id": "track-12", "event_type": "unlike"},
        )
        self.assertEqual(unliked.status_code, 200)
        library = self.client.get("/api/library/liked", headers=self.headers)
        self.assertNotIn("track-12", [item["track_id"] for item in library.json()])

    def test_registration_and_duplicate_conflict(self):
        payload = {
            "username": "api_test_user",
            "email": "api-test@example.local",
            "password": "long-test-password",
        }
        created = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(created.status_code, 200, created.text)
        duplicate = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(duplicate.status_code, 409)

    def test_demo_audio_is_served(self):
        response = self.client.get("/media/track-01.wav")
        self.assertEqual(response.status_code, 200)
        self.assertIn("audio", response.headers.get("content-type", ""))


if __name__ == "__main__":
    unittest.main()
