# SoundScout demo audio

SoundScout generates its own original instrumental audio at startup. No copyrighted commercial recordings are included.

- The first six tracks in `tracks.csv` are rendered as approximately 90-second full demo arrangements.
- The remaining catalogue tracks are rendered as approximately 18-second previews.
- Each track is synthesized from its own deterministic seed, tempo, scale, melody pattern, chord voicing and rhythm settings, so tracks no longer all replay the same short motif.
- Audio is written to the configured `AUDIO_DIR`. On startup, older short files are replaced automatically when they do not meet the expected duration.
- The six full demo tracks are suitable for a classroom demonstration of playback, seeking, queue controls and recommendation interactions. They are original generated instrumentals, not commercial vocal songs.

The first six tracks are `Neon Sunrise`, `Paper Planes`, `City Lights`, `Golden Hour`, `Quiet Corners` and `Wildflower Radio`.
