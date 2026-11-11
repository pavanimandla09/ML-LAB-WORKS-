# Music Recommendation System — Project Research & Implementation Plan

> **Status:** Planning / repository research  
> **Target repository:** [ML-LAB-WORKS-](https://github.com/pavanimandla09/ML-LAB-WORKS-)  
> **Working fork:** [knockknock10/ML-LAB-WORKS-](https://github.com/knockknock10/ML-LAB-WORKS-)  
> **Project type:** Machine-learning recommendation engine integrated with a Spotify-inspired music application

## 1. Project idea

Build a music discovery application that looks and behaves like a modern music streaming player, but whose main academic contribution is its own recommendation engine.

The Spotify-style application is the **product interface**: users can browse a catalogue, search, play tracks, like songs, create playlists, and view listening history. The machine-learning service is the **core project**: it learns from song metadata and user listening behaviour to recommend relevant tracks.

This should not be presented as merely another Spotify UI clone. The differentiator is the comparison and integration of multiple recommendation algorithms, followed by an explanation of why a track was recommended.

### Main objectives

1. Build a usable music-player interface and catalogue.
2. Implement and compare content-based and collaborative recommendation methods.
3. Combine the best-performing methods into a hybrid recommender.
4. Collect user interactions from the application and use them as recommendation feedback.
5. Evaluate the models with ranking metrics and demonstrate cold-start handling.

## 2. Repository inspection and integration decision

The current fork contains the original ML lab Jupyter notebooks at the repository root. It does not currently contain a music application, recommendation code, or a project-specific dependency setup.

**Integration rule:** add this as a separate `music-recommendation/` project directory. Do not modify or reorganize the existing lab notebooks. Keep datasets, model files, and credentials out of Git.

## 3. Spotify-style application repositories reviewed

These are candidate application foundations, not code that has already been copied into this repository.

| Repository | What it offers | License / trade-off | Assessment |
|---|---|---|---|
| [LazzouziYoussef/spotify-clone](https://github.com/LazzouziYoussef/spotify-clone) | React + Express + MongoDB; music streaming, authentication, admin dashboard, and Socket.IO live features | MIT. The repository uses TypeScript, so the team must be comfortable maintaining TS files. | **Best MERN-style candidate to inspect first** |
| [Leandro-Amorim/supafy](https://github.com/Leandro-Amorim/supafy) | Next.js + Supabase; authentication, song search, favourites, playlists, uploads, listening history, and persistent queue | MIT. Uses a different backend model (Supabase/PostgreSQL) rather than the MERN API. | Good alternative when hosted storage and database features are preferred |
| [SyedAzhar2004/Spotify-Clone](https://github.com/SyedAzhar2004/Spotify-Clone) | MERN application with login, search, playlists, playback controls, and admin song management | MIT. Its README warns that later V2 work may not match the documented setup; inspect the intended commit before adopting. | Simpler alternative; verify setup and maintenance first |
| [asudbury/react-spotify-clone](https://github.com/asudbury/react-spotify-clone) | React Spotify Web API client and player-style interface | Depends on Spotify authentication/API behaviour; not a self-hosted music catalogue/player. | Useful UI reference, not the preferred playback foundation |

### Recommended application approach

Start by inspecting **LazzouziYoussef/spotify-clone** as the MERN integration candidate. Its MIT licence permits modification and redistribution subject to retaining the licence notice. Before importing any files, inspect its current branch, setup steps, dependencies, and audio-storage implementation. If TypeScript adds unnecessary friction for the team, compare it with the simpler MIT-licensed MERN option above.

Do not copy an entire clone into the lab repository without reviewing its code and retaining the applicable licence and copyright notices. Reuse only the components and application structure that are actually needed.

### Important Spotify API limitation

Do not make the project dependent on Spotify providing the actual audio stream. Spotify's documentation says Web API access and playback control have Premium requirements, and the Web Playback SDK is a Spotify Connect player rather than a general-purpose source of downloadable music. For a reproducible classroom demo, use audio files the team is authorised to host (for example, properly licensed sample tracks) and store metadata/artwork with the application. Spotify links or metadata can be optional external references, not the only way the demo works.

References:
- [Spotify Web API documentation](https://developer.spotify.com/documentation/web-api)
- [Spotify Web Playback SDK documentation](https://developer.spotify.com/documentation/web-playback-sdk)

## 4. Recommendation-system algorithm plan

The teammate has specifically mentioned **SVD** and **content-based filtering**. Those two are treated as the confirmed algorithm requirements. The third algorithm was not specified, so the plan below proposes **item-item KNN** as a comparison model and **popularity-based recommendation** as a simple baseline / cold-start fallback. Confirm the unnamed algorithm with the teammate before describing it as a team-agreed requirement.

### Algorithm A — Popularity-based baseline

Recommend tracks using aggregate signals such as unique listeners, play counts, likes, and recent listening activity.

- Gives a simple baseline against which the ML models can be evaluated.
- Works for visitors and new accounts that do not have listening history.
- Can produce trending or popular-in-a-genre lists.
- Must be calculated without leaking future interactions from the evaluation period.

This is a baseline, not the main personalised algorithm.

### Algorithm B — Content-based filtering (confirmed)

Recommend songs that are similar to tracks a user already likes or listens to.

**Track features to consider:**
- Artist, album, genre, tags, and track title.
- Text descriptions or mood labels, where available.
- Numeric audio descriptors such as tempo, energy, danceability, valence, acousticness, and instrumentalness, only when the selected dataset supplies reliable values.

**Proposed method:**
1. Clean and combine the text metadata.
2. Convert text fields into vectors using TF-IDF.
3. Scale numeric audio features before combining them with text vectors.
4. Calculate item similarity with cosine similarity.
5. Build a user profile from the tracks the user likes or listens to.
6. Rank unseen tracks by similarity to that profile and exclude tracks already consumed where appropriate.

**Strength:** works for new or rarely-played songs when their metadata is available.  
**Limitation:** it can over-specialise and keep recommending songs that resemble what the user already knows.

### Algorithm C — Collaborative filtering with SVD (confirmed)

Collaborative filtering uses patterns across users. If users with similar listening behaviour enjoyed a track, the system can recommend that track to another user with a similar interaction pattern.

Create a sparse user–track interaction matrix:

`R[user, track] = interaction strength`

Possible signals include play count, completed listens, likes, saves, repeat plays, and skips. The exact weighting should be documented and tested.

**SVD / matrix-factorisation flow:**
1. Map users and tracks to integer IDs.
2. Aggregate interactions into a sparse user–track matrix.
3. Apply truncated SVD / latent-factor decomposition.
4. Represent users and tracks in a lower-dimensional latent space.
5. Estimate scores for unseen user–track pairs using latent-factor similarity or reconstructed matrix scores.
6. Rank candidate tracks and filter out tracks already consumed.

Conceptually, the factorisation approximates the interaction matrix as:

`R ≈ U Σ Vᵀ`

**Important implementation detail:** Last.fm listening counts are implicit feedback, not explicit 1–5 star ratings. A plain SVD model can be used as a course-level matrix-factorisation experiment, but its input transformation and limitations must be stated. The Surprise library's SVD implementation is designed for explicit rating data and its documentation says it does not support implicit ratings or content-based information. For play-count data, use a sparse-matrix approach such as `sklearn.decomposition.TruncatedSVD` on a documented transformed interaction matrix, or collect explicit in-app ratings if the team specifically wants a rating-prediction implementation.

**Strength:** captures taste patterns that cannot be inferred from song metadata alone.  
**Limitation:** suffers from cold start for new users and new tracks, and requires interaction history.

### Algorithm D — Item-item KNN (proposed comparison algorithm)

Use nearest-neighbour similarity to find tracks that are frequently co-consumed by similar users.

- Build item vectors from user interaction columns (or use a clearly labelled content-vector KNN variant).
- Use cosine similarity / nearest neighbours to find tracks similar to a selected track.
- Recommend tracks that are close to the user's recent or liked tracks.

This is a useful, explainable comparison against SVD. Do not confuse item-item KNN with content-based cosine similarity: one uses interaction patterns; the other uses track attributes.

### Final model — Hybrid recommender

The final recommendation list should combine collaborative, content, and popularity scores. An initial scoring design is:

`hybrid_score = w_cf × normalized(SVD_score) + w_cb × normalized(content_score) + w_pop × normalized(popularity_score)`

Initial weights can be set for experimentation (for example, 0.50 / 0.40 / 0.10), but they must be tuned on a validation split rather than presented as optimal in advance.

- Users with sufficient listening history: give more weight to collaborative filtering.
- New users: use selected favourite artists/genres and content similarity, with popularity as a fallback.
- New tracks: use content-based similarity until interaction data accumulates.
- Provide a short reason such as “similar genre and mood”, “listeners with similar taste enjoyed this”, or “popular this week”.

## 5. Proposed user-facing features

### Core music application

- Sign up, sign in, and user profile.
- Home page with featured tracks, popular tracks, and personalised sections.
- Search by track, artist, album, or genre.
- Browse tracks by genre and mood.
- Play, pause, previous, next, seek, and volume controls.
- Persistent player that remains available while navigating.
- Play queue, shuffle, and repeat.
- Like / unlike tracks.
- Create, edit, and delete personal playlists.
- Recently played and listening history.
- Responsive layout for desktop and mobile.
- Admin/content-manager flow for adding track metadata, artwork, and permitted audio files.

### Machine-learning features — the project differentiator

- **Made for You:** personalised hybrid recommendations.
- **Similar Tracks:** item-to-item recommendations for the currently selected song.
- **Because You Listened To:** recommendations based on a liked/recently played track.
- **Discover:** tracks outside the user's usual listening pattern.
- **Mood / Genre Mixes:** recommendations filtered by metadata or audio descriptors.
- **Feedback loop:** likes, skips, completed plays, repeats, and playlist saves update the user's interaction history.
- **Recommendation explanations:** show the main reason a track appeared in the list.

## 6. Data strategy

### Preferred starting point: Million Song Dataset ecosystem

The [Million Song Dataset](https://millionsongdataset.com/) provides metadata and audio-derived features for one million tracks. The core dataset does **not** contain the actual audio recordings. Its companion datasets include the Taste Profile user-data subset and Last.fm tags.

This is a strong research starting point because the interaction data can support collaborative filtering while track metadata/tags can support content-based filtering. Start with a manageable sample and document the exact files and preprocessing.

### Alternative: Last.fm 1K

The [Last.fm 1K dataset](https://zenodo.org/records/6090214) contains approximately 19.15 million listening events from 992 users, with timestamps, artist IDs/names, track IDs/names. It is useful for implicit-feedback experiments. The dataset is made available for non-commercial use, so follow its terms and cite Last.fm when using it.

### Data-handling rules

- Do not commit large raw datasets or trained model binaries to Git.
- Add download/setup instructions and a small, synthetic fixture for tests.
- Keep a stable mapping between the app's track IDs and the ML dataset's track IDs.
- Do not assume all tracks have complete metadata or audio features.
- Do not use copyrighted commercial songs or artwork unless the team has permission.
- Clearly label synthetic interactions; never report synthetic-data metrics as real-world performance.

## 7. System architecture

Recommended separation of responsibilities:

```text
React music application
        |
        | REST API / user actions
        v
Node.js + Express application backend
        |
        +---- MongoDB
        |       users, tracks, playlists, likes,
        |       listening events, play history
        |
        +---- Python recommendation service
                  FastAPI
                  |
                  +-- data preprocessing
                  +-- content-based model
                  +-- SVD collaborative model
                  +-- KNN comparison
                  +-- hybrid scoring
                  +-- offline evaluation
```

The application backend records interactions. The Python service trains models offline and serves ranked track IDs and scores. The UI fetches those track IDs and renders the matching catalogue records.

### Example API contract

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/recommendations/for-you` | Personalised hybrid recommendations for the current user |
| GET | `/api/recommendations/similar/:trackId` | Similar tracks for a selected track |
| POST | `/api/interactions` | Record play, completion, skip, like, repeat, or playlist-save event |
| GET | `/api/tracks/search?q=` | Search the application catalogue |
| GET | `/health` (ML service) | Check recommendation service availability |

The exact route prefixes can be adapted to the selected clone's conventions.

## 8. Suggested project structure

Keep the new work isolated from the existing ML lab notebooks:

```text
music-recommendation/
├── README.md
├── docs/
│   ├── PROJECT_PROPOSAL.md
│   ├── DATASET.md
│   └── MODEL_EVALUATION.md
├── ml/
│   ├── data/
│   │   ├── raw/                 # downloaded locally; gitignored
│   │   └── processed/           # generated locally; gitignored
│   ├── notebooks/
│   │   └── 01_recommendation_eda.ipynb
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── popularity.py
│   │   ├── content_based.py
│   │   ├── collaborative_svd.py
│   │   ├── item_knn.py
│   │   ├── hybrid.py
│   │   └── evaluate.py
│   ├── tests/
│   └── requirements.txt
├── recommendation_api/
│   ├── app/
│   │   └── main.py
│   └── requirements.txt
└── .gitignore
```

The selected Spotify-style application can be integrated in a dedicated `app/` or `frontend/` directory after its structure and licence have been reviewed. Avoid maintaining two competing backends for the same user and track data.

## 9. Evaluation plan

Compare the same test users and candidate tracks across:

1. Popularity baseline.
2. Content-based only.
3. SVD collaborative filtering only.
4. Item-item KNN.
5. Hybrid recommender.

Use a chronological split of user interactions (older events for training, later events for validation/test) to reduce future-data leakage. Report:

- **Precision@K:** fraction of recommended tracks that are relevant.
- **Recall@K:** fraction of relevant held-out tracks that were recommended.
- **NDCG@K:** rewards relevant tracks appearing higher in the list.
- **MAP@K:** ranking quality across users.
- **RMSE / MAE:** only if an explicit rating or rating-like prediction task is actually defined.

Also report coverage and catalogue diversity so a model is not rewarded for repeatedly recommending only a few popular tracks. Do not invent scores; every number in the final report must come from a reproducible run.

## 10. Implementation milestones

| Phase | Deliverable | Completion check |
|---|---|---|
| 1. Confirm scope and data | Final dataset choice, fields, licence notes, app base decision | Data schema and reproducible download/preparation steps documented |
| 2. Build baselines | Popularity + content-based recommender | Given a track/user profile, returns ranked tracks |
| 3. Build collaborative models | SVD + item-item KNN | Models train on interaction data and return unseen-track scores |
| 4. Hybrid and evaluation | Tuned score blending and metrics | Side-by-side evaluation table; no fabricated results |
| 5. Application integration | Player UI + event logging + recommendation API | UI displays recommendations for an identified user |
| 6. Testing and documentation | Unit tests, demo instructions, final report | Fresh clone can run the demo from documented steps |

## 11. Acceptance criteria for the completed project

- Existing ML lab notebooks remain unchanged.
- The recommendation code is modular and reproducible.
- Popularity, content-based, SVD, KNN, and hybrid methods can be run independently.
- Recommendations exclude already-played tracks where appropriate and handle users with no history.
- Every model is evaluated against the same temporal test set.
- The application records listening feedback and uses it in recommendations.
- The UI explains at least one reason for each personalised recommendation.
- No secrets, unlicensed music, or large raw datasets are committed.
- The README explains setup, data preparation, model training, API startup, application startup, and evaluation.

## 12. Research references and reuse notes

### Application foundations
- [LazzouziYoussef/spotify-clone](https://github.com/LazzouziYoussef/spotify-clone) — MIT; React / Express / MongoDB candidate.
- [Leandro-Amorim/supafy](https://github.com/Leandro-Amorim/supafy) — MIT; Next.js / Supabase candidate.
- [SyedAzhar2004/Spotify-Clone](https://github.com/SyedAzhar2004/Spotify-Clone) — MIT; alternate MERN candidate.

### Recommendation-system references
- [d-elicio/Music-Recommender-System-from-scratch](https://github.com/d-elicio/Music-Recommender-System-from-scratch) — useful comparison of popularity, user/item collaborative filtering, SVD, and content-based methods; no repository licence was identified during this review, so use it as a reading reference rather than copying code.
- [DataSorcerer/Music-Recommendation-System](https://github.com/DataSorcerer/Music-Recommendation-System) — SVD and popularity-oriented music recommender using Million Song Dataset / Taste Profile data; no repository licence was identified during this review.
- [gdstudio-org/Embeat](https://github.com/gdstudio-org/Embeat) — more advanced item-to-item music retrieval reference; its repository declares CC BY-NC 4.0, so do not copy its code into this project without checking the licence implications.
- [arvindxyogesh/spotify-music-recommender](https://github.com/arvindxyogesh/spotify-music-recommender) — hybrid CF + content-based ranking design reference.

### Dataset and algorithm documentation
- [Million Song Dataset](https://millionsongdataset.com/) — metadata/audio-derived features and companion Taste Profile / Last.fm datasets.
- [Last.fm 1K dataset record](https://zenodo.org/records/6090214) — listening event format, dataset size, and non-commercial terms.
- [Surprise documentation](https://surpriselib.com/) — useful for rating-based recommender experiments; note its SVD implementation is for explicit ratings, not raw implicit listening events.
- [Spotify Web API](https://developer.spotify.com/documentation/web-api) and [Web Playback SDK](https://developer.spotify.com/documentation/web-playback-sdk) — official constraints for any optional Spotify integration.

---

## 13. Decisions still to confirm with the team

1. What was the third algorithm the teammate mentioned? KNN is proposed here as the comparison model, not a confirmed requirement.
2. Which application base should be used: MERN (Lazzouzi clone) or Next.js/Supabase (Supafy)?
3. Is the assignment expected to include a working web application, or is the primary grading deliverable the ML model and notebook?
4. Which dataset does the faculty/team expect, if any?

**Recommended direction:** build and evaluate the ML engine first, keep the player application as a separate integration layer, and submit the working system only after the recommendation pipeline has reproducible evidence.


## 14. Implementation delivered in the working branch

The initial research proposal has now been turned into a runnable local-first application in `music-recommendation/`.

Implemented components:

- FastAPI application and SQLite persistence for accounts, tracks, listening events and playlists.
- Responsive browser-based music player with search, genre browsing, playback controls, likes, playlists, history and model selector.
- TF-IDF plus cosine content filtering.
- Implicit-feedback SVD using scikit-learn `TruncatedSVD`.
- Item-item KNN using cosine neighbours over user-interaction vectors.
- Popularity baseline and hybrid blending with a cold-start fallback.
- API-level and recommender unit tests.
- Chronological top-K evaluation script for Precision@K, Recall@K, NDCG@K, MAP@K and catalogue coverage.
- Synthetic demo catalogue, synthetic interaction profiles and locally generated short WAV previews.
- GitHub Actions workflow for tests and model-evaluation smoke checks.

### Deliberate integration choice

The MERN clone candidate was inspected as an application feature reference. Its documented setup requires MongoDB, Clerk identity credentials and Cloudinary media credentials. To keep the coursework reproducible without external service accounts, this branch implements a smaller original player/API rather than importing the candidate's source files. The external repository is not presented as code that was copied or merged.

### Important boundary

The bundled sample data is synthetic and is only for verifying that the full project starts and the recommendation paths execute. It is not a substitute for the academic experiment on a properly licensed real dataset. The team still needs to choose the real evaluation dataset and report metrics from that dataset.

### Run and test

See [music-recommendation/README.md](../music-recommendation/README.md) for environment setup, demo credentials, API endpoints, test commands and model comparison.
