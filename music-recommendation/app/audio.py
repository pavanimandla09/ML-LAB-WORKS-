"""Create distinct, original instrumental demo tracks for SoundScout.

The first six catalogue tracks are full-length 90-second arrangements. The
remaining tracks receive distinct 18-second previews. All audio is generated
locally from deterministic synthesis; no commercial recordings are bundled.
"""
from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np

from app.db import AUDIO_DIR, TRACKS_CSV


FULL_TRACK_COUNT = 6
FULL_TRACK_SECONDS = 90.0
PREVIEW_SECONDS = 18.0
SAMPLE_RATE = 22050

# Different harmonic palettes make each track sound recognisably different.
SCALES = (
    (0, 2, 4, 7, 9),       # bright major
    (0, 2, 3, 7, 8),       # mellow minor
    (0, 3, 5, 7, 10),      # blues
    (0, 2, 5, 7, 9),       # open / ambient
    (0, 2, 4, 7, 11),      # uplifting
    (0, 3, 5, 8, 10),      # cinematic minor
)


def _tone(frequency: float, length: int, rate: int, phase: float = 0.0) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(length, dtype=np.float64) / rate
    return np.sin(2.0 * np.pi * frequency * t + phase)


def _render_track(track: dict[str, str], index: int, duration: float) -> np.ndarray:
    """Render one deterministic arrangement with melody, bass, chords and rhythm."""
    rate = SAMPLE_RATE
    total = int(rate * duration)
    output = np.zeros(total, dtype=np.float64)
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(track.get("track_id", str(index))))
    rng = np.random.default_rng(seed)

    tempo = float(track.get("tempo") or 100)
    tempo = min(160.0, max(60.0, tempo))
    beat = 60.0 / tempo
    root_midi = 45 + (index * 3) % 12
    root = 440.0 * (2.0 ** ((root_midi - 69) / 12.0))
    scale = SCALES[index % len(SCALES)]
    motif = rng.choice(scale, size=16, replace=True)
    # Repeat a motif with small variations, rather than looping one short sample.
    motif[::4] = np.array([scale[(index + n) % len(scale)] for n in range(4)])

    beat_count = int(math.ceil(duration / beat))
    genre = (track.get("genre") or "").lower()
    is_ambient = genre == "ambient"
    is_jazz = genre == "jazz"
    is_electronic = genre == "electronic"
    is_hiphop = genre in {"hip-hop", "hip hop"}

    for beat_no in range(beat_count):
        start = int(beat_no * beat * rate)
        if start >= total:
            break
        length = min(int(beat * rate * 0.94), total - start)
        if length <= 0:
            continue
        t = np.arange(length, dtype=np.float64) / rate
        bar_beat = beat_no % 4
        bar_no = beat_no // 4
        section = (bar_no // 8) % 4
        note_step = motif[(beat_no + (section * 3)) % len(motif)]
        octave = 1.0 if (beat_no % 8 in (0, 2, 4, 6)) else 2.0
        melody_hz = root * (2.0 ** (float(note_step) / 12.0)) * octave

        # Soft attack/release on every note.
        attack = np.minimum(1.0, t / 0.025)
        release = np.minimum(1.0, (length / rate - t) / 0.10)
        env = np.clip(attack, 0, 1) * np.clip(release, 0, 1)
        if section == 2:
            env *= 1.12  # chorus lift
        if section == 3:
            env *= 0.72  # bridge drop

        if is_ambient:
            melody = 0.45 * _tone(melody_hz * 0.5, length, rate)
            melody += 0.20 * _tone(melody_hz, length, rate, 0.3)
        elif is_jazz:
            melody = 0.42 * _tone(melody_hz, length, rate)
            melody += 0.16 * _tone(melody_hz * 2.0, length, rate, 0.2)
        else:
            melody = 0.34 * _tone(melody_hz, length, rate)
            melody += 0.12 * _tone(melody_hz * 2.0, length, rate, 0.15)
            melody += 0.05 * _tone(melody_hz * 3.0, length, rate, 0.3)

        # Bass changes with the bar and keeps each track's groove grounded.
        chord_degree = scale[(bar_no + index) % len(scale)]
        bass_hz = root * (2.0 ** (chord_degree / 12.0)) * 0.5
        bass = 0.22 * _tone(bass_hz, length, rate)
        if is_hiphop or is_electronic:
            bass += 0.13 * _tone(bass_hz * 0.5, length, rate)

        # Chord pad with genre-dependent voicing.
        third = 3 if (index % 2) else 4
        fifth = 7
        pad = (
            0.10 * _tone(root, length, rate)
            + 0.075 * _tone(root * 2.0 ** (third / 12.0), length, rate, 0.1)
            + 0.055 * _tone(root * 2.0 ** (fifth / 12.0), length, rate, 0.2)
        )
        if is_ambient:
            pad *= 1.8

        segment = (melody * env + bass + pad) * 0.58

        # Original synthesized percussion; ambient tracks intentionally stay sparse.
        if not is_ambient:
            if bar_beat == 0:
                kick = np.exp(-t * 18.0) * np.sin(2 * np.pi * (48 - 22 * t) * t)
                segment += 0.30 * kick
            if bar_beat == 2 and not is_electronic:
                snare = rng.normal(0, 1, length) * np.exp(-t * 28.0)
                segment += 0.055 * snare
            if bar_beat in (1, 3) or (is_electronic and bar_beat % 2 == 1):
                hat = rng.normal(0, 1, length) * np.exp(-t * 65.0)
                segment += 0.025 * hat

        output[start:start + length] += segment

    # Smooth fade at the beginning and end; keep the waveform comfortably below clipping.
    fade_in = min(int(1.2 * rate), total)
    fade_out = min(int(2.0 * rate), total)
    if fade_in:
        output[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    if fade_out:
        output[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    peak = float(np.max(np.abs(output))) if total else 0.0
    if peak > 0:
        output = output / max(1.0, peak / 0.88)
    return np.clip(output, -0.98, 0.98)


def _existing_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as existing:
            return existing.getnframes() / float(existing.getframerate())
    except (OSError, wave.Error, ZeroDivisionError):
        return 0.0


def ensure_demo_audio(sample_rate: int = SAMPLE_RATE, duration: float = FULL_TRACK_SECONDS) -> int:
    """Ensure six full demo tracks and distinct previews exist on disk.

    Existing 8-second preview files are automatically replaced when their
    duration is shorter than the new catalogue target.
    """
    global SAMPLE_RATE
    SAMPLE_RATE = int(sample_rate)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not TRACKS_CSV.exists():
        return 0

    with TRACKS_CSV.open("r", encoding="utf-8", newline="") as handle:
        tracks = list(csv.DictReader(handle))

    created = 0
    for index, track in enumerate(tracks):
        filename = Path(str(track.get("audio_file", ""))).name
        if not filename or Path(filename).suffix.lower() != ".wav":
            continue
        target_duration = min(float(duration), FULL_TRACK_SECONDS) if index < FULL_TRACK_COUNT else PREVIEW_SECONDS
        destination = AUDIO_DIR / filename
        if destination.exists() and _existing_duration(destination) >= target_duration - 0.5:
            continue

        signal = _render_track(track, index, target_duration)
        pcm = (signal * 32767.0).astype("<i2")
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())
        created += 1

    return created
