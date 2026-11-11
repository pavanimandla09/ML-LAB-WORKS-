# SoundScout Model Card

## Model purpose

SoundScout ranks music catalogue items for discovery. It is an educational recommender-system prototype, not a commercial streaming service or a production-validated personalisation model.

## Methods implemented

| Method | Input | Technique | Role |
|---|---|---|---|
| Content-based | Track title, artist, album, genre, mood, tags and available audio descriptors | TF-IDF + standardised numeric descriptors + cosine similarity | Recommends tracks similar to a user's positive listening profile |
| Collaborative SVD | Weighted user-track implicit feedback matrix | scikit-learn TruncatedSVD | Captures latent listener-track preference patterns |
| Item-item KNN | Track vectors over user interaction strengths | Cosine nearest neighbours | Finds tracks co-consumed by similar listener groups |
| Popularity | Counts of positive events and distinct listeners | Log-scaled aggregate score | Baseline and cold-start recommendation |
| Hybrid | Normalised outputs from SVD, content and popularity | Weighted score blend | Default personalised ranking |

## Interaction semantics

Positive event weights in the initial prototype:

- Play: 1.0
- Repeat: 1.5
- Complete: 1.8
- Playlist add: 2.5
- Like: 3.0

Repeated events are summed per user-track pair and transformed using log(1 + strength). Skips are retained for history and treated as consumed tracks so they are not immediately recommended again. Skips and unlike events are not assigned negative values in the SVD matrix.

This is a lightweight implicit-feedback modelling choice. It does not estimate star ratings and should not be described as an implicit-ALS model.

## Hybrid scoring

For a user represented in the SVD training matrix:

    0.45 × SVD + 0.40 × content + 0.15 × popularity

Each component is min-max normalised for the current catalogue. If SVD is not available for that user, the hybrid system uses:

    0.55 × content + 0.45 × popularity

These weights are initial design parameters. They need to be tuned on a chronological validation set before any claim that they are optimal.

## Data used in the committed demo

- 24 synthetic track metadata rows.
- 8 synthetic listener profiles, including the demo account profile.
- Synthetic play, complete, like, repeat and skip events.
- Short, original synthesised WAV previews generated locally on first start.

The bundled fixture exists only to exercise the full stack and the model interfaces. It does not represent actual listening patterns, commercial tracks, artists or performance.

## Intended evaluation

Use one chronological holdout policy for all methods. Earlier interactions train the models; later positive events form the relevance set.

Report:

- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- Catalogue coverage

Use a real, properly licensed dataset for the academic results. Compare each recommender on the same users, candidate catalogue and holdout interactions. State the dataset, sample filters, split date, K, and the number of test users. Do not report the synthetic fixture scores as real-user performance.

## Known limitations

- The demo catalogue is small; model rankings on it are not representative of a large catalogue.
- Metadata completeness drives content-based quality.
- Collaborative methods need enough interactions from multiple users.
- TruncatedSVD on a log-weighted implicit matrix is a simple course-level approximation; modern implicit-feedback models such as weighted matrix factorisation/implicit ALS may be stronger.
- A skip excludes a track from immediate rediscovery but is not yet a calibrated negative preference in the model.
- The hybrid weights are hand-set starting values rather than validated optimised weights.
- Popularity can introduce exposure bias and should be combined with diversity-aware ranking for a production system.
- Recommendation explanations are concise descriptions of the signals used; they are not causal explanations.
- The service uses SQLite and local file storage for a single-machine academic demo. It is not configured for multi-instance production deployment.

## Intended users and use

Students, instructors and reviewers who want to inspect, run and compare music recommendation methods in a reproducible small application.

## Safety, privacy and rights

Do not import copyrighted audio or artwork without authorisation. Review dataset terms before using, distributing or publishing results. The app stores account details and listening events locally in SQLite; do not place real credentials, secrets or private user data in public Git repositories.
