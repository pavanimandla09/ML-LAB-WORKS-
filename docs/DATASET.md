# Dataset Notes

## Bundled demonstration fixture

SoundScout ships two small CSV files:

- music-recommendation/data/tracks.csv — synthetic catalogue metadata.
- music-recommendation/data/interactions.csv — synthetic listener interaction events.

The track catalogue contains 24 invented tracks distributed across Pop, Indie, Electronic, Jazz, Ambient and Hip-Hop. Track artists and albums are fictional. The audio previews are generated as original short synthesised sounds when the application starts; no commercial recording is bundled.

The interaction file contains 8 synthetic profiles with play, complete, repeat, like and skip events. The demo account username is demo, and its database user ID is demo so the account resolves to its seeded sample interaction profile.

This fixture is small enough for tests and demonstrations. Do not use its offline metrics as evidence of real recommendation performance.

## CSV schemas

### tracks.csv

| Column | Meaning |
|---|---|
| track_id | Stable application identifier |
| title, artist, album | Track metadata |
| genre, mood, tags | Text attributes for content filtering |
| energy, danceability, valence | Normalised numeric descriptors from 0 to 1 |
| tempo | Tempo in beats per minute |
| acousticness, instrumentalness | Normalised descriptors from 0 to 1 |
| audio_file | Relative filename generated in data/audio |

The six numeric descriptors are illustrative metadata values for this demo. They are not extracted from the generated preview files.

### interactions.csv

| Column | Meaning |
|---|---|
| user_id | Demo user/profile identifier |
| track_id | Track identifier present in tracks.csv |
| event_type | play, complete, repeat, like or skip |
| timestamp | UTC event timestamp in ISO-8601 format |

Real data adapters should map source event schemas into this canonical format. Ensure unknown track IDs and malformed timestamps are filtered or reported, not silently mapped to unrelated tracks.

## Real dataset options

### Million Song Dataset and Taste Profile

The Million Song Dataset provides metadata and derived audio features for a million tracks. The core dataset does not include the actual audio recordings. It also links to companion datasets, including the Taste Profile subset, which contains user-track listening count information.

Official source: https://millionsongdataset.com/

The Taste Profile subset is a reasonable first real collaborative-filtering source. Check the dataset's current terms and the terms of all companion data before use. Create a stable join between the dataset's track identifiers and the application track catalogue; unmatched records should not be guessed or joined by title alone.

### Last.fm Dataset — 1K users

The Last.fm 1K dataset contains timestamped user listening events. Its published record reports 19,150,868 event rows from 992 unique users and provides user, timestamp, artist and track fields. It is suitable for a time-ordered implicit-feedback experiment after data cleaning.

Dataset record: https://zenodo.org/records/6090214

The published dataset is made available for non-commercial use and asks users to reference Last.fm. Review the record and applicable terms before using or redistributing it; do not assume commercial reuse is permitted.

## Recommended real-data preprocessing

1. Download data directly from the official dataset source and record the version/date.
2. Preserve original identifiers, timestamps and source fields in a raw local directory; do not commit the raw dataset to Git.
3. Remove malformed rows, empty user/track identifiers and invalid timestamps.
4. Map each source record to the canonical user_id, track_id, event_type, timestamp schema.
5. Aggregate repeated user-track events with a documented rule.
6. For a Last.fm event stream, a repeated play can be represented as play or repeat; do not invent likes or completion events if the source does not contain them.
7. Use a time-ordered train/validation/test split to avoid training on events that happen after the test period.
8. Report the number of users, tracks, interactions, sparsity, time range and filters applied.
9. Keep copyrighted recordings and cover art out of the repository unless the team has documented permission to use them.

## Next research step

Run the committed Last.fm importer on a local copy of the official listening TSV, document the selected users/tracks/events and filters, and evaluate every algorithm on the same chronological holdout. Keep the synthetic CSVs as deterministic smoke-test data so the test suite does not require large downloads or external network access.


## Import tool

The repository includes music-recommendation/scripts/import_lastfm_1k.py. It reads the official six-column TSV in chunks, samples user IDs reproducibly, and reservoir-samples listening events per selected user when an event budget is configured. This avoids a serious sampling bias when the TSV is grouped by listener: selecting the first 250,000 rows would otherwise overrepresent only the first few users.

Default output is data/lastfm_import/tracks.csv and data/lastfm_import/interactions.csv; this directory is ignored by Git. To run:

    python scripts/import_lastfm_1k.py /path/to/userid-timestamp-artid-artname-traid-traname.tsv --max-users 150 --max-events 250000 --min-track-plays 2

For a local application experiment, stop the server, back up the synthetic CSV files, copy the imported CSVs to data/tracks.csv and data/interactions.csv, remove data/music.db, and restart. The app database only seeds catalogue/interactions on first initialization.

The importer maps each Last.fm listen to a play event; it does not invent likes or completed-listen labels. It can optionally map one source user ID to the local demo user ID using --demo-source-user-id. Use only a source ID present in the imported data.

Last.fm 1K contains track/artist identifiers and timestamps, but not audio recordings or the numeric acoustic descriptors used by the demo. Imported tracks therefore have Unknown genre, empty tags and no audio file. Recommendations can use available track/artist text and collaborative listening patterns; actual playback requires audio files the team has permission to use. The dataset is for non-commercial use; cite Last.fm and comply with the terms at the official record.
