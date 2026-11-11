"""Generate original, short synthesised demo audio without third-party songs."""
from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from app.db import AUDIO_DIR, TRACKS_CSV


def ensure_demo_audio(sample_rate: int = 22050, duration: float = 8.0) -> int:
    """Create a small, deterministic WAV preview for each bundled demo track."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not TRACKS_CSV.exists():
        return 0

    created = 0
    with TRACKS_CSV.open("r", encoding="utf-8", newline="") as handle:
        tracks = list(csv.DictReader(handle))

    time_axis = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    fade_in = np.clip(time_axis / 0.35, 0.0, 1.0)
    fade_out = np.clip((duration - time_axis) / 0.65, 0.0, 1.0)
    envelope = fade_in * fade_out

    for index, track in enumerate(tracks):
        filename = Path(str(track.get("audio_file", ""))).name
        if not filename or not filename.lower().endswith(".wav"):
            continue
        destination = AUDIO_DIR / filename
        if destination.exists():
            continue

        tempo = max(60.0, min(160.0, float(track.get("tempo") or 100.0)))
        beat_seconds = 60.0 / tempo
        step = np.floor(time_axis / (beat_seconds / 2)).astype(int)
        root_hz = 110.0 + (index % 12) * 7.0
        scale = np.array([0, 2, 4, 7, 9, 7, 4, 2, 0, 4, 7, 11], dtype=float)
        notes = root_hz * np.power(2.0, scale / 12.0)
        melody_hz = notes[step % len(notes)]
        chord_hz = np.array(
            [root_hz, root_hz * (2.0 ** (4 / 12)), root_hz * (2.0 ** (7 / 12))]
        )
        pad = (
            np.sin(2.0 * np.pi * chord_hz[0] * time_axis)
            + 0.65 * np.sin(2.0 * np.pi * chord_hz[1] * time_axis)
            + 0.45 * np.sin(2.0 * np.pi * chord_hz[2] * time_axis)
        ) / 3.0
        melody = np.sin(2.0 * np.pi * melody_hz * time_axis)
        pulse = np.sin(2.0 * np.pi * (55.0 + index * 1.5) * time_axis)
        signal = (0.33 * melody + 0.22 * pad + 0.08 * pulse) * envelope
        signal = np.clip(signal, -0.95, 0.95)
        pcm = (signal * 32767.0).astype("<i2")

        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        created += 1

    return created
