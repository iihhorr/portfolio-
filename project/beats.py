#!/usr/bin/env python3
"""Beat / drop analysis -> out/beats.json and the cut schedule.

    python beats.py            # analyse input/track.* and write out/beats.json
"""
import glob
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(ROOT, "input")
OUT = os.path.join(ROOT, "out")

FINAL_DUR = 1.5      # white end card
FAST_DUR = 2.0       # half-beat cuts after the drop
TARGET_LEN = (13.0, 16.0)


def find_track():
    for ext in ("mp3", "m4a", "wav", "aac", "flac", "mp4", "mov"):
        hits = sorted(glob.glob(os.path.join(INPUT, f"track.{ext}")))
        if hits:
            return hits[0]
    hits = [p for p in glob.glob(os.path.join(INPUT, "*")) if os.path.isfile(p)
            and p.lower().endswith((".mp3", ".m4a", ".wav", ".mp4", ".mov", ".aac", ".flac"))]
    if not hits:
        raise SystemExit("Нет трека: положи input/track.mp3 (или .m4a / .mp4)")
    return sorted(hits)[0]


def to_wav(path):
    """Extract audio (works for video files too) to a mono 22.05 kHz wav."""
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, "_track.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "22050", wav], check=True)
    return wav


def analyse(cfg):
    import librosa

    src = find_track()
    y, sr = librosa.load(to_wav(src), sr=22050, mono=True)
    dur = len(y) / sr
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, bf = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop)
    bpm = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(bf, sr=sr, hop_length=hop)

    # drop = beat with the biggest energy jump (next 1.5 s vs previous 1.5 s)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    on = oenv / (oenv.max() + 1e-9)
    energy = rms / (rms.max() + 1e-9) * 0.7 + on[: len(rms)] * 0.3
    tpf = hop / sr
    win = int(1.5 / tpf)

    def jump(t):
        i = int(t / tpf)
        a = energy[max(0, i - win): i]
        b = energy[i: i + win]
        return (b.mean() if len(b) else 0) - (a.mean() if len(a) else 0)

    pre_min = TARGET_LEN[0] - FAST_DUR - FINAL_DUR   # intro before drop: 9.5 .. 12.5 s
    pre_max = TARGET_LEN[1] - FAST_DUR - FINAL_DUR
    if cfg.get("drop_time") is not None:
        drop = float(beats[np.argmin(np.abs(beats - cfg["drop_time"]))])
    else:
        cands = [b for b in beats if b >= pre_min and b <= dur - FAST_DUR - FINAL_DUR]
        if not cands:
            raise SystemExit("Трек слишком короткий для 13–16 с ролика с дропом")
        drop = float(max(cands, key=jump))

    # offset = a beat so that the drop lands ~11 s in (≈75 % of the clip)
    ideal = 11.0
    okb = [b for b in beats if pre_min <= drop - b <= pre_max]
    offset = float(min(okb, key=lambda b: abs(drop - b - ideal))) if okb else max(0.0, drop - ideal)
    data = dict(source=os.path.relpath(src, ROOT), bpm=round(bpm, 2), beats=[round(float(b), 4) for b in beats],
                drop=round(drop, 4), track_offset=round(offset, 4), track_duration=round(dur, 3))
    with open(os.path.join(OUT, "beats.json"), "w") as f:
        json.dump(data, f, indent=2)
    return data


def schedule(data, cfg, fps):
    """Return clip length and a list of slots (times are clip-relative seconds).
    slot = dict(t, kind='photo'|'pair'|'flash'|'final', beat_no, src_beat)"""
    bpm = data["bpm"]
    step = 2 if bpm > 130 else 1
    off, drop = data["track_offset"], data["drop"]
    beats = [b - off for b in data["beats"] if b >= off - 1e-6]
    d = drop - off
    pairs = set(cfg.get("pairs", []))
    slots, n = [], 0
    pre = [b for b in beats if b < d - 1e-6][::step]
    for b in pre:
        n += 1
        slots.append(dict(t=b, kind="pair" if n in pairs else "photo", beat_no=n, src_beat=b))
    # drop: 2 white frames, then half-beat cuts for FAST_DUR
    slots.append(dict(t=d, kind="flash", beat_no=n + 1, src_beat=d))
    post = [b for b in beats if d - 1e-6 <= b <= d + FAST_DUR + 0.6]
    half = []
    for a, b in zip(post, post[1:]):
        half += [a, (a + b) / 2]
    half.append(post[-1] if post else d)
    fast_end_target = d + FAST_DUR
    end_fast = min(half, key=lambda x: abs(x - fast_end_target))
    flash_end = d + 2 / fps
    first = True
    for h in half:
        if h >= end_fast - 1e-6:
            break
        n += 1
        slots.append(dict(t=flash_end if first else h, kind="photo", beat_no=n, src_beat=h, fast=True))
        first = False
    slots.append(dict(t=end_fast, kind="final", beat_no=n + 1, src_beat=end_fast))
    length = end_fast + FINAL_DUR
    return length, slots, step


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(INPUT, "config.json"), encoding="utf-8"))
    d = analyse(cfg)
    print(f"BPM {d['bpm']}, drop {d['drop']} s, track_offset {d['track_offset']} s")
