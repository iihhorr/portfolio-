#!/usr/bin/env python3
"""SOVAKAV word-by-word visualizer, driven entirely by a scenes JSON.

    python render.py                          # scenes.json
    python render.py --scenes scenes_track.json
    python render.py --contact                # only out/contact.png

Timing: either a beat grid (video.bpm + scene.beats) or explicit scene.start/end and
scene.word_times (absolute seconds). Audio: video.track (real track) or a synthesized bed.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw

import cards
from cards import F, clamp, lerp, ease_out_cubic

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100
M = 70                       # side margin
SAFE_BOTTOM, SAFE_RIGHT = 320, 140
CFG = TL = WAVE = GRAIN = None


# ------------------------------------------------------------------ text markup / layout
def parse_markup(text):
    words, cur, seg, acc = [], [], "", False
    for ch in text + " ":
        if ch == "*":
            if seg:
                cur.append((seg, acc)); seg = ""
            acc = not acc
        elif ch == " ":
            if seg:
                cur.append((seg, acc)); seg = ""
            if cur:
                words.append(cur); cur = []
        else:
            seg += ch
    return words


TRACK_EM = 0.02   # headline tracking +2 %


def head_font(acc, size):
    return F("m800", size) if acc else F("m400", size)


def seg_w(seg, acc, size):
    return cards.tracked_width(seg, head_font(acc, size), TRACK_EM)


def word_w(w, size):
    return sum(seg_w(s, a, size) for s, a in w)


def layout_text(words, s, size, v):
    tc = v["text"]
    x0, y0, maxw = tc["x"], tc["y"], tc["width"]
    lh = size * tc.get("line_height", 1.08)
    space = F("m400", size).getlength(" ") * 1.1
    lines, cur, cw = [], [], 0.0
    for wi, w in enumerate(words):
        ww = word_w(w, size)
        if cur and cw + space + ww > maxw:
            lines.append((cur, cw)); cur, cw = [], 0.0
        cur.append((wi, ww)); cw += (space if len(cur) > 1 else 0) + ww
    lines.append((cur, cw))
    asc = F("m800", size).getmetrics()[0]
    top = v["height"] * 0.46 - len(lines) * lh / 2 if s.get("center") else y0
    pos = {}
    for li, (ln, lw) in enumerate(lines):
        x = (v["width"] - lw) / 2 if s.get("center") else x0
        base = top + asc * 0.9 + li * lh
        for wi, ww in ln:
            pos[wi] = (x, base); x += ww + space
    bottom = top + asc * 0.9 + (len(lines) - 1) * lh + size * 0.35
    return dict(pos=pos, bottom=bottom, lines=len(lines))


# ------------------------------------------------------------------ timeline
def build_timeline(cfg):
    v = cfg["video"]
    beat = 60.0 / v.get("bpm", 90)
    step = beat * v.get("word_step_beats", 0.5)
    scenes, t = [], 0.0
    for s in cfg["scenes"]:
        txt = s["text"].upper() if v.get("caps", True) else s["text"]
        words = parse_markup(txt)
        start = s.get("start", t)
        if "word_times" in s:
            wt = [x - start for x in s["word_times"]]
        else:
            wt = [i * step for i in range(len(words))]
        assert len(wt) == len(words), f"scene {s.get('id')}: {len(words)} слов, {len(wt)} word_times"
        end = s["end"] if "end" in s else start + max(s.get("beats", 0) * beat, wt[-1] + s.get("hold", 1.0))
        size = s.get("size", v["text"]["size"])
        scenes.append(dict(cfg=s, start=start, end=end, dur=end - start, words=words, word_times=wt,
                           layout=layout_text(words, s, size, v), size=size, beat=beat,
                           th=cards.theme(s.get("theme", "dark"))))
        t = end
    return scenes


def total_len():
    return CFG["video"].get("duration", TL[-1]["end"])


def scene_at(t):
    for i, s in enumerate(TL):
        if t < s["end"]:
            return i
    return len(TL) - 1


# ------------------------------------------------------------------ audio
def load_track(path, n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)
    off = int(CFG["video"].get("track_offset", 0) * SR)
    a = a[off: off + n]
    return np.pad(a, (0, max(0, n - len(a))))


def synth_audio(cfg, n):
    a = cfg["video"]["audio"]
    out = np.zeros(n, np.float32)

    def add(sig, t0, gain=1.0):
        i = int(round(t0 * SR))
        if i < n:
            j = min(n, i + len(sig)); out[i:j] += sig[: j - i] * gain

    tk = np.arange(int(0.42 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(42 + 85 * np.exp(-tk * 28)) / SR) * np.exp(-tk * 8.5)
    tt = np.arange(int(0.045 * SR)) / SR
    tick = (np.sin(2 * np.pi * 2600 * tt) * 0.6 + np.sin(2 * np.pi * 5200 * tt) * 0.25) * np.exp(-tt * 110)
    th = np.arange(int(1.6 * SR)) / SR
    noise = np.convolve(np.random.default_rng(7).normal(0, 1, len(th)), np.ones(18) / 18, mode="same")
    hit = np.sin(2 * np.pi * np.cumsum(30 + 110 * np.exp(-th * 14)) / SR) * np.exp(-th * 3.2) * 1.6 + noise * np.exp(-th * 7) * 1.2
    pads = a.get("pads", [[55, 82.4, 110]])
    for k, s in enumerate(TL):
        sc = s["cfg"]
        if sc.get("silent"):
            continue
        for b in range(int(math.floor(s["dur"] / s["beat"] + 1e-6))):
            if not (b == 0 and sc.get("hit")):
                add(kick, s["start"] + b * s["beat"], a.get("kick", 0.9))
        if sc.get("hit"):
            add(hit, s["start"])
        L = int(s["dur"] * SR); tp = np.arange(L) / SR
        sig = sum(np.sin(2 * np.pi * f * tp + i) for i, f in enumerate(pads[k % len(pads)])) / len(pads[k % len(pads)])
        add((sig * np.clip(np.minimum(tp / 0.25, (s["dur"] - tp) / 0.2), 0, 1)).astype(np.float32), s["start"], a.get("pad", 0.15))
    for s in TL:
        for wt in s["word_times"]:
            add(tick, s["start"] + wt, a.get("tick", 0.35))
    out = np.tanh(out * 1.2)
    return out * a.get("peak", 0.3) / (np.abs(out).max() + 1e-9)


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())


def wave_bars(audio, nframes, fps, nb=40):
    """40 bars driven by the track: per-band spectrum × RMS, fast attack / slow release."""
    win = 2048
    edges = np.geomspace(35, 9000, nb + 1)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    idx = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(nb)]
    hann = np.hanning(win)
    bands, rms = np.zeros((nframes, nb)), np.zeros(nframes)
    for fi in range(nframes):
        c = int(fi / fps * SR)
        seg = audio[max(0, c - win // 2): c + win // 2]
        seg = np.pad(seg, (0, win - len(seg)))
        sp = np.abs(np.fft.rfft(seg * hann))
        bands[fi] = [sp[ix].mean() if len(ix) else 0 for ix in idx]
        rms[fi] = np.sqrt(np.mean(seg ** 2))
    bands = np.log1p(bands * 4)
    bands /= np.percentile(bands, 97, axis=0) + 1e-6
    rms /= rms.max() + 1e-9
    shape = 0.55 + 0.45 * np.abs(np.sin(np.arange(nb) * 0.9 + 0.5))
    v = np.clip((0.4 * np.clip(bands, 0, 1) + 0.6 * shape) * np.clip(rms[:, None] * 1.6, 0, 1) ** 0.7, 0, 1)
    sm = np.zeros_like(v)
    for fi in range(nframes):
        sm[fi] = v[fi] if fi == 0 else np.maximum(v[fi], sm[fi - 1] * 0.78)
    return sm


# ------------------------------------------------------------------ drawing
def init_grain(v):
    rng = np.random.default_rng(3)
    k = v.get("grain", 0.03)
    return [rng.normal(0, 255 * k, (v["height"], v["width"])).astype(np.float32)[..., None] for _ in range(6)]


def background(s, fi):
    col = np.array(s["th"]["bg"], np.float32)[None, None, :]
    return Image.fromarray(np.clip(col + GRAIN[fi % len(GRAIN)], 0, 255).astype(np.uint8), "RGB")


def draw_card(frame, s, lt):
    c = s["cfg"].get("card")
    if not c:
        return
    v = CFG["video"]
    ctx = dict(word_times=s["word_times"], beat=s["beat"], dur=s["dur"])
    card = cards.render_card(c, lt, ctx, v["card"]["width"], s["th"])
    t_in = lt - c.get("enter", 0.0)
    if t_in < 0:
        return
    e = ease_out_cubic(t_in / c.get("enter_dur", 0.3))
    a = clamp(t_in / c.get("enter_dur", 0.3))
    if a < 1:
        card.putalpha(card.getchannel("A").point(lambda x: int(x * a)))
    drift = c.get("drift_y", 0) * clamp(lt / s["dur"])
    x = (v["width"] - card.width) // 2
    y = int(v["card"]["center_y"] - card.height / 2 + (1 - e) * c.get("enter_dy", 40) + drift)
    frame.paste(card, (x, y), card)


def draw_head(d, x, y, seg, acc, size, fill):
    f = head_font(acc, size)
    w = cards.tracked(d, x, y, seg, f, fill, TRACK_EM)
    if acc:   # accent = ExtraBold + 3 px underline
        d.rectangle((x, y + size * 0.14, x + w, y + size * 0.14 + 3), fill=fill)
    return w


def draw_text(frame, s, lt):
    th = s["th"]
    d = ImageDraw.Draw(frame, "RGBA")
    for wi, w in enumerate(s["words"]):
        p = clamp((lt - s["word_times"][wi]) / 0.12)
        if p <= 0:
            continue
        x, base = s["layout"]["pos"][wi]
        y = base + (1 - ease_out_cubic(p)) * 10
        for seg, acc in w:
            x += draw_head(d, x, y, seg, acc or s["cfg"].get("center", False), s["size"], th["fg"] + (int(255 * p),))


def draw_chrome(frame, s, fi):
    v = CFG["video"]
    th = s["th"]
    W, H = v["width"], v["height"]
    d = ImageDraw.Draw(frame, "RGBA")
    cards.label(d, M, 108, f"{v.get('number', '01')} / {v.get('rubric', 'Мысль')}", th, color=th["fg"])
    cards.tracked(d, W - M, 110, v.get("wordmark", "SOVAKAV"), F("m700", 26), th["fg"], 0.24, "right")
    d.line((M, 134, W - M, 134), fill=th["line"], width=2)
    # footer block: bottom edge kept above the 320 px safe zone
    fy = H - SAFE_BOTTOM - 120
    d.line((M, fy, W - M, fy), fill=th["line"], width=2)
    cx, cy, r = M + 36, fy + 66, 36
    dark = th["name"] == "dark"
    # dark: black circle / white letters; light: inverted
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(10, 10, 10) if dark else (255, 255, 255),
              outline=th["line"], width=2)
    cards.tracked(d, cx + 1, cy + 9, "SV", F("m700", 24), (242, 242, 240) if dark else (10, 10, 10), 0.04, "center")
    cards.tracked(d, M + 96, cy + 8, " • ".join(v.get("tags", ["Сайты", "ИИ", "2026"])).upper(),
                  F("i500", 22), th["soft"], 0.18)
    vals = WAVE[min(fi, len(WAVE) - 1)]
    n = len(vals)
    x0, x1, base, hmax = 600, W - M, cy + 30, 56
    gap = (x1 - x0 - 3 * n) / (n - 1)
    for k, val in enumerate(vals):
        h = max(2, val * hmax)
        xx = x0 + k * (3 + gap)
        d.rectangle((xx, base - h, xx + 2, base), fill=th["fg"] + (220,))


def draw_cta(frame, s, lt):
    cta = s["cfg"].get("cta")
    if not cta:
        return
    t0 = cta["at"] if "at" in cta else s["dur"] - cta.get("last", 2.0)
    p = clamp((lt - t0) / 0.3)
    if p <= 0:
        return
    th = s["th"]
    d = ImageDraw.Draw(frame, "RGBA")
    e = ease_out_cubic(p)
    col = th["fg"] + (int(255 * p),)
    y = cta.get("y", 1340)
    cards.tracked(d, M, y + (1 - e) * 10, cta["text"].upper(), F("m800", 48), col, 0.02)
    drift = 6 * math.sin((lt - t0) * 2 * math.pi * 0.8)
    ox, oy = 720 + drift, y - 70 + drift
    L = 130 * e
    ex, ey = ox + L * 0.707, oy + L * 0.707
    d.line((ox, oy, ex, ey), fill=col, width=3)
    d.line((ex, ey, ex - 38, ey), fill=col, width=3)
    d.line((ex, ey, ex, ey - 38), fill=col, width=3)


def accent_state(t):
    """Signature move: 2 white-flash frames, then camera push 1.05 -> 1.00."""
    fps = CFG["video"]["fps"]
    for a in CFG["video"].get("accents", []):
        if a <= t < a + 2 / fps:
            return "flash", 0
        if a + 2 / fps <= t < a + 2 / fps + 0.35:
            return "push", ease_out_cubic((t - a - 2 / fps) / 0.35)
    return None, 0


def render_frame(fi):
    v = CFG["video"]
    t = fi / v["fps"]
    st, k = accent_state(t)
    s = TL[scene_at(t)]
    if st == "flash":
        frame = Image.new("RGB", (v["width"], v["height"]), (255, 255, 255))
        draw_chrome(frame, s, fi)
        return frame
    lt = t - s["start"]
    frame = background(s, fi)
    draw_card(frame, s, lt)
    draw_text(frame, s, lt)
    draw_cta(frame, s, lt)
    if st == "push":   # push the content, keep the brand frame steady
        z = lerp(1.05, 1.0, k)
        W, H = frame.size
        cw, ch = W / z, H / z
        frame = frame.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        d = ImageDraw.Draw(frame)   # clean header/footer zones before drawing chrome on top
        d.rectangle((0, 0, W, 150), fill=s["th"]["bg"])
        d.rectangle((0, H - SAFE_BOTTOM - 130, W, H), fill=s["th"]["bg"])
    draw_chrome(frame, s, fi)
    return frame


def render_bytes(fi):
    return render_frame(fi).tobytes()


# ------------------------------------------------------------------ QA
def qa_layout():
    v = CFG["video"]
    ok = True
    for s in TL:
        c = s["cfg"].get("card")
        lay = s["layout"]
        msg = f"   {s['cfg'].get('id', '?'):<10} {s['start']:6.2f}–{s['end']:5.2f}s  {s['th']['name']:<5} lines={lay['lines']}"
        right = max(p[0] + word_w(s["words"][i], s["size"]) for i, p in lay["pos"].items())
        if right > v["width"] - M + 1:
            print(f"!! {s['cfg'].get('id')}: текст выходит за правое поле ({right:.0f})"); ok = False
        if c:
            top = v["card"]["center_y"] - cards.HEIGHTS[c["type"]](c) / 2
            msg += f"  text bottom {lay['bottom']:.0f} / card top {top:.0f}"
            if lay["bottom"] > top:
                print(f"!! {s['cfg'].get('id')}: текст наезжает на карточку"); ok = False
        print(msg)
    return ok


def contact_sheet(path, nframes):
    fps = CFG["video"]["fps"]
    picks = list(np.linspace(0.5, total_len() - 0.1, 12))
    for a in CFG["video"].get("accents", [])[:2]:   # always show the accents
        picks[int(np.argmin([abs(p - a) for p in picks]))] = a + 0.5 / fps
    picks.sort()
    tw, th = 360, 640
    sheet = Image.new("RGB", (tw * 6, th * 2), (128, 128, 128))
    d = ImageDraw.Draw(sheet)
    for n, t in enumerate(picks):
        fi = min(nframes - 1, int(t * fps))
        sheet.paste(render_frame(fi).resize((tw - 4, th - 4), Image.LANCZOS), ((n % 6) * tw + 2, (n // 6) * th + 2))
        d.text(((n % 6) * tw + 10, (n // 6) * th + 150), f"{fi / fps:.2f}s", font=F("m700", 22), fill=(220, 30, 30))
    sheet.save(path)


# ------------------------------------------------------------------ main
def main():
    global CFG, TL, WAVE, GRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=os.path.join(ROOT, "scenes.json"))
    ap.add_argument("--contact", action="store_true")
    args = ap.parse_args()
    CFG = json.load(open(args.scenes, encoding="utf-8"))
    v = CFG["video"]
    TL = build_timeline(CFG)
    fps = v["fps"]
    nframes = int(round(total_len() * fps))
    print(f"duration {total_len():.2f}s, {nframes} frames")
    qa_layout()
    GRAIN = init_grain(v)
    n = int(round(nframes / fps * SR))
    track = os.path.join(ROOT, v["track"]) if v.get("track") else None
    audio = load_track(track, n) if track else synth_audio(CFG, n)
    WAVE = wave_bars(audio, nframes, fps)
    out = os.path.join(ROOT, v.get("output", "out/video.mp4"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    contact_sheet(os.path.join(os.path.dirname(out), v.get("contact", "contact.png")), nframes)
    if args.contact:
        print("contact sheet only"); return
    silent = out.replace(".mp4", "_silent.mp4")
    enc = ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-maxrate", "10M", "-bufsize", "20M",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"]
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{v['width']}x{v['height']}", "-r", str(fps), "-i", "-"] + enc + [silent],
                          stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for i, b in enumerate(pool.imap(render_bytes, range(nframes), chunksize=4)):
            ff.stdin.write(b)
    ff.stdin.close(); ff.wait()
    wav = os.path.join(os.path.dirname(out), "_audio.wav")
    write_wav(wav, audio)
    af = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"] if track else []
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy"] + af + ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-shortest",
                    "-movflags", "+faststart", out], check=True)
    os.remove(wav)
    print(f"done: {out}\n      {silent}")


if __name__ == "__main__":
    main()
