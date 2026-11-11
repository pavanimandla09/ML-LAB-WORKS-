# SoundScout — Hybrid Music Recommendation System

SoundScout is a self-contained machine-learning music discovery application. It combines a responsive, Spotify-inspired player interface with a Python recommendation service. Its academic contribution is the recommendation engine: content-based filtering, implicit-feedback collaborative filtering using SVD, item-item KNN, a popularity baseline, and a hybrid ranker.

The existing machine-learning lab notebooks in the parent repository are kept separate and unchanged.

## What is included

- A browser-based music discovery application with sign-in, registration, search, genre browsing, playback, volume, seek, previous/next, shuffle, repeat, liked songs, playlists, and listening history.
- A Python FastAPI backend with SQLite persistence.
- Content-based recommendations using TF-IDF metadata vectors and cosine similarity.
- Collaborative recommendations using TruncatedSVD on a weighted, sparse user-track interaction matrix.
- Item-item KNN based on cosine distance between track listening profiles.
- A popularity baseline for trending tracks and cold-start users.
- Hybrid scoring that combines collaborative, content and popularity scores.
- Similar-track discovery and a short explanation with each recommendation.
- Deterministic synthetic catalogue and interaction fixtures, plus original generated audio previews.
- Unit tests and a chronological offline ranking evaluation script.

## Requirements

- Python 3.11 or newer.
- pip.
- A modern desktop or mobile browser.

No Spotify account, MongoDB instance, Clerk account, Cloudinary account or external API key is required for the local demonstration.

## Run locally

From a terminal:

    cd music-recommendation
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000 in a browser.

The application creates a local SQLite database at data/music.db and generates short, original synthesised WAV previews in data/audio the first time it starts. Both are ignored by Git.

### Demo account

    Username: demo
    Password: music12345

The demo account is only for a local classroom walkthrough. Do not deploy the public demo credentials. Create a new account for your own usage.

## Recommendation algorithms

### 1. Content-based filtering

Track text is formed from title, artist, album, genre, mood and tags. TF-IDF with unigrams and bigrams transforms that text into sparse vectors. The available audio descriptors (energy, danceability, valence, tempo, acousticness and instrumentalness) are standardised and combined with the text representation. L2-normalised vectors are compared with cosine similarity.

For a user profile, the model forms a weighted average of tracks the user played, completed, repeated, liked or added to a playlist. Similarity to this profile supplies the content score. This method can recommend a newly added track as soon as its metadata exists.

### 2. SVD collaborative filtering

Interactions are aggregated into a sparse user-track matrix. Positive implicit event strengths are:

| Event | Strength |
|---|---:|
| Play | 1.0 |
| Repeat | 1.5 |
| Completed listen | 1.8 |
| Playlist add | 2.5 |
| Like | 3.0 |

The aggregate is transformed with log(1 + strength), then factorised with scikit-learn TruncatedSVD. Predicted preference scores are obtained from the user's latent vector and the learned item components.

This is an **implicit-feedback ranking experiment**, not a star-rating predictor. Skips are recorded and treated as consumed tracks so they are not immediately recommended again; they are not inserted as negative numbers into the non-negative interaction matrix. TruncatedSVD is used here as a course-friendly matrix factorisation method. It is not the same as an implicit-ALS objective.

SVD requires multiple users and sufficient interaction history. If it cannot be fitted or a user is new, the hybrid ranker falls back to content and popularity signals.

### 3. Item-item KNN

For every track, the model builds a vector of listener interaction strengths. NearestNeighbors with cosine distance finds tracks with similar interaction profiles. The tracks adjacent to a user's positive listening history become KNN candidates.

This is collaborative item similarity. It is distinct from content-based cosine similarity, which compares song metadata and audio descriptors.

### 4. Popularity baseline

Popularity combines the log of weighted positive interactions with the log of distinct listeners. It gives new users something sensible to explore before the system has learned their taste. It is a baseline, not the primary personalisation method.

### 5. Hybrid ranker

For users with an available SVD model, the initial blend is:

    hybrid = 0.45 * SVD + 0.40 * content + 0.15 * popularity

Each component is min-max normalised before blending. When SVD is not available for a user, the current implementation renormalises the available weights:

    cold_start = 0.55 * content + 0.45 * popularity

These are starting weights, not claimed optimal weights. Tune them using a validation set before reporting them as final.

## Data

The CSV files in data/ are a **small, synthetic demonstration fixture**. The tracks, artists and interaction profiles are invented for this project, and the bundled WAV previews are generated by the application. They are useful for checking that the full pipeline runs; they are not real listening records and must not be used to claim production recommendation accuracy.

For a research-quality experiment, replace or supplement the fixture with a real dataset whose terms allow your intended use. Suitable starting points to investigate are:

- Million Song Dataset: metadata and audio-derived features for a large catalogue. The core dataset does not contain commercial audio recordings. The Taste Profile subset provides user-track play counts.
- Last.fm 1K: timestamped listening events that can be transformed into implicit-feedback interactions. Check the dataset record and its non-commercial-use terms before using or redistributing it.

Keep large raw datasets and generated model files out of Git. Document the data source, version, licence/terms, filtering and train/test split. Never upload copyrighted audio or album artwork without permission. The demo's generated preview sounds are not the commercial recordings represented by any external dataset.

## API

The interactive UI uses the same-origin API. Useful endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /api/auth/register | Create an account |
| POST | /api/auth/login | Sign in and receive a bearer token |
| GET | /api/auth/me | Current account |
| GET | /api/tracks?q=...&genre=... | Search and filter catalogue |
| GET | /api/genres | Available genres |
| GET | /api/recommendations/hybrid | Personalised hybrid ranking |
| GET | /api/recommendations/content | Content-based ranking |
| GET | /api/recommendations/svd | SVD collaborative ranking |
| GET | /api/recommendations/knn | Item-item KNN ranking |
| GET | /api/recommendations/popularity | Popularity baseline |
| GET | /api/recommendations/similar/{track_id} | Similar tracks for a selected track |
| POST | /api/interactions | Record a play, completion, repeat, like, unlike or skip |
| GET | /api/library/liked | Liked songs for the current account |
| GET | /api/history | Recent listening events |
| GET | /api/playlists | Current user's playlists |
| POST | /api/playlists | Create a playlist |
| POST | /api/playlists/{id}/tracks | Add a track to a playlist |
| GET | /api/playlists/{id} | Playlist details and tracks |
| DELETE | /api/playlists/{id}/tracks/{track_id} | Remove a track from a playlist |
| GET | /api/health | Service and model diagnostics |

The automatically generated interactive API reference is available at http://127.0.0.1:8000/docs.

### Optional catalogue upload

Uploads are disabled unless an administrator explicitly configures an API key. Set ADMIN_API_KEY in the server environment, restart the service, then send a multipart POST to /api/admin/tracks with the key in the X-Admin-Key header. Allowed extensions are MP3, WAV, OGG and M4A; the upload limit is 25 MB. Supply title, artist and genre, and optionally album, mood, tags and the six numeric descriptors.

Do not expose an upload key in browser JavaScript or commit it to Git. The endpoint is intended for a trusted administrator / local demo workflow.

## Tests and evaluation

Run unit tests from this directory:

    python -m unittest discover -s tests -v

Run the model comparison:

    python scripts/evaluate.py --k 10

Optionally save the JSON report:

    python scripts/evaluate.py --k 10 --output reports/demo-evaluation.json

The evaluation uses a chronological per-user holdout: earlier events train the model and later positive events are held out. It reports Precision@K, Recall@K, NDCG@K, MAP@K and catalogue coverage for popularity, content, SVD, KNN and hybrid models.

The bundled fixture is synthetic. Any metrics obtained from it demonstrate that the evaluation code runs, not that the system performs well on real listeners. For the final academic report, evaluate on a documented real dataset, use the same chronological split for every model, and report the dataset size and limitations alongside the metrics.

## Project structure

    music-recommendation/
    ├── app/
    │   ├── audio.py
    │   ├── db.py
    │   ├── main.py
    │   └── security.py
    ├── data/
    │   ├── audio/                 generated locally; gitignored
    │   ├── interactions.csv      synthetic sample events
    │   └── tracks.csv             synthetic sample catalogue
    ├── ml/
    │   ├── evaluation.py
    │   └── recommender.py
    ├── scripts/
    │   └── evaluate.py
    ├── static/
    │   ├── app.js
    │   ├── index.html
    │   └── styles.css
    ├── tests/
    │   └── test_recommender.py
    ├── .env.example
    ├── .gitignore
    └── requirements.txt

## Security and deployment notes

- Use a strong, random JWT_SECRET outside local development. The server refuses to start in production mode with its built-in development secret.
- Do not expose ADMIN_API_KEY in the frontend, source control or public logs.
- The included demo account has a known password and is for local demonstration only.
- Use HTTPS and a managed secret store for a deployed instance.
- Configure rate limiting, account recovery, moderation and storage quotas before public deployment.
- SQLite is appropriate for this local academic prototype. A multi-instance public deployment should use a managed database and durable object storage.
- Serve only music files and artwork that the project has the right to use.

## Application reuse and licence note

A MERN Spotify-style application was reviewed as a feature and architecture reference. Its README requires external identity and media services (including Clerk and Cloudinary) in addition to MongoDB. To make this academic project reproducible without account credentials or cloud setup, SoundScout implements a small local-first player and API instead of copying that application's source files. The original repository's licence must be retained if its code is later adopted; the SoundScout implementation does not include copied clone components.

The original lab repository has no project-specific licence visible in the inspected root. Confirm with the repository owner/faculty before choosing a licence for the combined coursework submission.

## Importing real Last.fm 1K interactions

The repository includes a streaming importer for the six-column Last.fm 1K listening file. Obtain the dataset from its official [Zenodo record](https://zenodo.org/records/6090214), extract the archive locally, and review the non-commercial-use terms before use.

Example (the raw file is not committed):

    python scripts/import_lastfm_1k.py /path/to/userid-timestamp-artid-artname-traid-traname.tsv --max-users 150 --max-events 250000 --min-track-plays 2

The importer makes two streaming passes: it selects a reproducible sample of listeners, then reservoir-samples events per selected user so the event limit is not consumed by only the first user groups in the TSV. The output is written to data/lastfm_import/ by default; it does not overwrite the bundled fixtures.

To use the imported interactions in the running app:

1. Stop the server.
2. Back up data/tracks.csv and data/interactions.csv.
3. Copy data/lastfm_import/tracks.csv and data/lastfm_import/interactions.csv over those two canonical files.
4. Delete data/music.db so the database is rebuilt from the new CSVs on next start.
5. Restart the app and sign in with a new account or use the demo account if you mapped one source user to it.

To map a known source listener to the bundled demo user, pass:

    --demo-source-user-id user_000639

Replace the example ID with a source user ID actually present in the selected data. Otherwise, the demo account will not have the imported listener's collaborative profile.

Last.fm 1K provides listening events and basic artist/track metadata; it does not provide the audio recordings or the numeric audio descriptors used by this application's richer content model. Imported rows therefore use Unknown genre, blank mood/tags, zero numeric descriptors and no audio file. The content model can still use track and artist text; the audio player cannot play an imported recording until you add an audio file that you have the rights to use. Do not download commercial recordings from unofficial sources.

Do not commit the extracted dataset or imported CSVs. The default import output directory is Git-ignored.
