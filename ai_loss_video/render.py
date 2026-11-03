#!/usr/bin/env python3
"""Render a word-by-word TikTok visualizer from scenes.json.

    python render.py              -> out/<output>.mp4 + out/contact.png
    python render.py --contact    -> only out/contact.png (fast preview)
    python render.py --scenes other.json
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw

import cards
from cards import F, hex2rgb, clamp, lerp, ease_out, ease_out_back, ease_in_out

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100

# globals shared with forked workers
CFG = TL = WAVE = GRAIN = VIGN = None


# ------------------------------------------------------------------ timeline
def parse_markup(text):
    """'a *b c* d' -> [[('a',False)], [('b',True)], [('c',True)], [('d',False)]]"""
    words, cur, seg, acc = [], [], "", False
    for ch in text + " ":
        if ch == "*":
            if seg:
                cur.append((seg, acc))
                seg = ""
            acc = not acc
        elif ch == " ":
            if seg:
                cur.append((seg, acc))
                seg = ""
            if cur:
                words.append(cur)
                cur = []
        else:
            seg += ch
    return words


def build_timeline(cfg):
    v = cfg["video"]
    beat = 60.0 / v["bpm"]
    step = beat * v.get("word_step_beats", 0.5)
    tcfg = v["text"]
    scenes, t = [], 0.0
    for s in cfg["scenes"]:
        words = parse_markup(s["text"])
        wt = [i * step for i in range(len(words))]
        remark_t = wt[-1] + step if s.get("remark") else None
        last = remark_t if remark_t is not None else wt[-1]
        dur = max(s.get("beats", 0) * beat, last + s.get("hold", 1.0))
        size = s.get("size", tcfg["size"])
        layout = layout_text(words, s, size, tcfg, v)
        scenes.append(dict(cfg=s, start=t, end=t + dur, dur=dur, words=words, word_times=wt,
                           remark_t=remark_t, layout=layout, size=size, beat=beat,
                           accent=hex2rgb(s.get("accent", "#ffffff")),
                           bg=(hex2rgb(s["bg"][0]), hex2rgb(s["bg"][1]))))
        t += dur
    return scenes


def layout_text(words, s, size, tcfg, v):
    font = F("bold", size)
    x0, y0, maxw = tcfg["x"], tcfg["y"], tcfg["width"]
    lh = size * tcfg.get("line_height", 1.16)
    space = font.getlength(" ")
    lines, cur, cw = [], [], 0.0
    for wi, w in enumerate(words):
        ww = sum(font.getlength(seg) for seg, _ in w)
        if cur and cw + space + ww > maxw:
            lines.append((cur, cw))
            cur, cw = [], 0.0
        cur.append((wi, ww))
        cw += (space if len(cur) > 1 else 0) + ww
    lines.append((cur, cw))
    asc = font.getmetrics()[0]
    pos = {}
    if s.get("center"):
        top = v["height"] * 0.47 - len(lines) * lh / 2
    else:
        top = y0
    for li, (ln, lw) in enumerate(lines):
        x = (v["width"] - lw) / 2 if s.get("center") else x0
        base = top + asc * 0.9 + li * lh
        for wi, ww in ln:
            pos[wi] = (x, base)
            x += ww + space
    remark = None
    if s.get("remark"):
        rf = F("reg", 34)
        rw = rf.getlength(s["remark"])
        lx, lb = pos[len(words) - 1]
        lastw = lines[-1][1]
        end_x = (x0 + lastw) if not s.get("center") else lx + 0
        if end_x + 16 + rw <= x0 + maxw:
            remark = (end_x + 16, lb)
        else:
            remark = (x0, lb + lh * 0.7)
    bottom = (remark[1] if remark else top + asc * 0.9 + (len(lines) - 1) * lh) + size * 0.3
    return dict(pos=pos, remark=remark, bottom=bottom, lines=len(lines))


def scene_at(t):
    for i, s in enumerate(TL):
        if t < s["end"]:
            return i
    return len(TL) - 1


# ------------------------------------------------------------------ audio
def synth_audio(cfg, n):
    a = cfg["video"]["audio"]
    out = np.zeros(n, np.float32)

    def add(sig, t0, gain=1.0):
        i = int(round(t0 * SR))
        if i >= n:
            return
        j = min(n, i + len(sig))
        out[i:j] += sig[: j - i] * gain

    tk = np.arange(int(0.42 * SR)) / SR
    freq = 42 + 85 * np.exp(-tk * 28)
    kick = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-tk * 8.5)
    kick[: int(0.004 * SR)] += np.random.default_rng(1).normal(0, 0.4, int(0.004 * SR))

    tt = np.arange(int(0.045 * SR)) / SR
    tick = (np.sin(2 * np.pi * 2600 * tt) * 0.6 + np.sin(2 * np.pi * 5200 * tt) * 0.25) * np.exp(-tt * 110)

    th = np.arange(int(1.6 * SR)) / SR
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, len(th))
    noise = np.convolve(noise, np.ones(18) / 18, mode="same")  # crude lowpass
    hit = (np.sin(2 * np.pi * np.cumsum(30 + 110 * np.exp(-th * 14)) / SR) * np.exp(-th * 3.2) * 1.6
           + noise * np.exp(-th * 7) * 1.2)

    pads = a.get("pads", [[55, 82.4, 110]])
    pad_i = 0
    for s in TL:
        sc = s["cfg"]
        if sc.get("silent"):
            continue
        # kicks on every beat
        nb = int(math.floor(s["dur"] / s["beat"] + 1e-6))
        for b in range(nb):
            if b == 0 and sc.get("hit"):
                continue
            add(kick, s["start"] + b * s["beat"], a.get("kick", 0.9))
        if sc.get("hit"):
            add(hit, s["start"], 1.0)
        # low pad
        chord = pads[pad_i % len(pads)]
        pad_i += 1
        L = int(s["dur"] * SR)
        tp = np.arange(L) / SR
        sig = np.zeros(L)
        for k, f in enumerate(chord):
            sig += np.sin(2 * np.pi * f * tp + k) + 0.3 * np.sin(2 * np.pi * f * 1.003 * 2 * tp)
        sig /= len(chord)
        env = np.minimum(1, np.minimum(tp / 0.25, (s["dur"] - tp) / 0.2))
        sig *= np.clip(env, 0, 1) * (0.75 + 0.25 * np.sin(2 * np.pi * tp / (s["beat"] * 2)))
        add(sig.astype(np.float32), s["start"], a.get("pad", 0.15))
    # ticks on every word / remark
    for s in TL:
        for wt in s["word_times"]:
            add(tick, s["start"] + wt, a.get("tick", 0.35))
        if s["remark_t"] is not None:
            add(tick, s["start"] + s["remark_t"], a.get("tick", 0.35) * 0.7)
    out = np.tanh(out * 1.2)
    out *= a.get("peak", 0.3) / (np.abs(out).max() + 1e-9)
    return out


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(st.tobytes())


def wave_bars(audio, nframes, fps, nb=40):
    win = 2048
    edges = np.geomspace(35, 9000, nb + 1)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    hann = np.hanning(win)
    bands = np.zeros((nframes, nb))
    rms = np.zeros(nframes)
    for fi in range(nframes):
        c = int(fi / fps * SR)
        seg = audio[max(0, c - win // 2): c + win // 2]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        sp = np.abs(np.fft.rfft(seg * hann))
        for b in range(nb):
            m = (freqs >= edges[b]) & (freqs < edges[b + 1])
            bands[fi, b] = sp[m].mean() if m.any() else 0
        rms[fi] = np.sqrt(np.mean(seg ** 2))
    bands = np.log1p(bands * 4)
    bands /= np.percentile(bands, 97, axis=0) + 1e-6
    rms /= rms.max() + 1e-9
    shape = 0.55 + 0.45 * np.abs(np.sin(np.arange(nb) * 0.9 + 0.5))
    loud = np.clip(rms[:, None] * 1.8, 0, 1) ** 0.6
    v = np.clip((0.35 * np.clip(bands, 0, 1) + 0.65 * shape) * loud, 0, 1)  # silence -> flat
    sm = np.zeros_like(v)
    for fi in range(nframes):  # fast attack, slow release
        sm[fi] = v[fi] if fi == 0 else np.maximum(v[fi], sm[fi - 1] * 0.8)
    return sm


# ------------------------------------------------------------------ frame
def init_static(cfg):
    v = cfg["video"]
    W, H = v["width"], v["height"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.45) / (H * 0.62)) ** 2)
    vign = 1 - 0.55 * np.clip(d - 0.45, 0, 1) ** 1.6
    rng = np.random.default_rng(3)
    grain = [rng.normal(0, 5.5, (H, W)).astype(np.float32)[..., None] for _ in range(6)]
    return vign[..., None], grain


def background(t, fi):
    v = CFG["video"]
    W, H = v["width"], v["height"]
    half = v.get("transition", 0.4) / 2
    i = scene_at(t)
    c0, c1 = map(np.array, TL[i]["bg"])
    # blend with neighbour near boundaries
    if i + 1 < len(TL) and t > TL[i]["end"] - half:
        m = ease_in_out((t - (TL[i]["end"] - half)) / (2 * half))
        n0, n1 = map(np.array, TL[i + 1]["bg"])
        c0, c1 = c0 + (n0 - c0) * m, c1 + (n1 - c1) * m
    elif i > 0 and t < TL[i]["start"] + half:
        m = ease_in_out((t - (TL[i]["start"] - half)) / (2 * half))
        p0, p1 = map(np.array, TL[i - 1]["bg"])
        c0, c1 = p0 + (c0 - p0) * m, p1 + (c1 - p1) * m
    g = np.linspace(0, 1, H, dtype=np.float32) ** 1.3
    col = (c0[None, :] + (c1 - c0)[None, :] * g[:, None]).astype(np.float32)[:, None, :]
    img = col * VIGN + GRAIN[fi % len(GRAIN)]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def with_alpha(im, k):
    if k >= 0.999:
        return im
    im = im.copy()
    im.putalpha(im.getchannel("A").point(lambda a: int(a * k)))
    return im


def draw_card(frame, s, lt, fade):
    c = s["cfg"].get("card")
    if not c:
        return
    v = CFG["video"]
    cw = v["card"]["width"]
    ctx = dict(word_times=s["word_times"], accent=s["accent"], beat=s["beat"])
    card = cards.render_card(c, lt, ctx, cw)
    pad = 70
    sh = cards.shadow(card.width, card.height, pad)
    comp = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    comp.putalpha(sh)
    comp.alpha_composite(card, (pad, pad))
    e = clamp(lt / 0.35)
    breath = 1.0075 - 0.0075 * math.cos(2 * math.pi * lt / (s["beat"] * 4))
    scale = breath
    comp = comp.resize((int(comp.width * scale), int(comp.height * scale)), Image.BICUBIC)
    comp = comp.rotate(s["cfg"].get("tilt", 0), resample=Image.BICUBIC, expand=True)
    dy = (1 - ease_out_back(e)) * 900
    comp = with_alpha(comp, min(1.0, e * 2.5) * fade)
    x = (v["width"] - comp.width) // 2
    y = int(v["card"]["center_y"] - comp.height / 2 + dy)
    frame.paste(comp, (x, y), comp)


def draw_text(frame, s, lt, fade):
    d = ImageDraw.Draw(frame, "RGBA")
    font = F("bold", s["size"])
    space = font.getlength(" ")
    for wi, w in enumerate(s["words"]):
        p = clamp((lt - s["word_times"][wi]) / 0.12)
        if p <= 0:
            continue
        x, base = s["layout"]["pos"][wi]
        y = base + (1 - ease_out(p)) * 12
        a = int(255 * p * fade)
        for seg, acc in w:
            col = s["accent"] if acc else (255, 255, 255)
            d.text((x, y), seg, font=font, fill=col + (a,), anchor="ls")
            x += font.getlength(seg)
    if s["remark_t"] is not None:
        p = clamp((lt - s["remark_t"]) / 0.15)
        if p > 0:
            x, b = s["layout"]["remark"]
            d.text((x, b + (1 - ease_out(p)) * 8), s["cfg"]["remark"], font=F("reg", 34),
                   fill=(175, 175, 185, int(255 * p * fade)), anchor="ls")


def draw_bottom(frame, s, fi):
    v = CFG["video"]
    d = ImageDraw.Draw(frame, "RGBA")
    # avatar
    cx, cy, r = 70 + 60, 1690, 60
    acc = s["accent"]
    d.ellipse((cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5), fill=(255, 255, 255, 230))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=tuple(int(c * 0.55) for c in acc) + (255,))
    d.text((cx, cy + 2), v.get("avatar_initials", "I.P."), font=F("xbold", 40), fill=(255, 255, 255), anchor="mm")
    # waveform: 40 bars, bottom aligned
    vals = WAVE[min(fi, len(WAVE) - 1)]
    n = len(vals)
    x0, x1, base, hmax = 560, 1010, 1750, 150
    bw = 7
    gap = (x1 - x0 - bw * n) / (n - 1)
    for k, val in enumerate(vals):
        h = max(6, val * hmax)
        x = x0 + k * (bw + gap)
        d.rounded_rectangle((x, base - h, x + bw, base), 3, fill=(255, 255, 255, 215))


def draw_cta(frame, s, lt):
    cta = s["cfg"].get("cta")
    if not cta:
        return
    t0 = s["dur"] - cta.get("last", 2.0)
    p = clamp((lt - t0) / 0.3)
    if p <= 0:
        return
    d = ImageDraw.Draw(frame, "RGBA")
    a = int(255 * p)
    yel = s["accent"]
    d.text((70, 1395 + (1 - ease_out(p)) * 20), cta["text"], font=F("xbold", 64), fill=(255, 255, 255, a), anchor="ls")
    bob = 16 * math.sin((lt - t0) * 2 * math.pi * 1.5)
    ox, oy = 700 + bob * 0.7, 1420 + bob * 0.7
    L = 190 * ease_out_back(p)
    ex, ey = ox + L * 0.7, oy + L * 0.7
    d.line((ox, oy, ex, ey), fill=yel + (a,), width=22)
    for ang in (math.radians(180 + 45 - 40), math.radians(180 + 45 + 40)):
        hx, hy = ex + math.cos(ang) * 70, ey + math.sin(ang) * 70
        d.line((ex, ey, hx, hy), fill=yel + (a,), width=22)
    d.ellipse((ex - 11, ey - 11, ex + 11, ey + 11), fill=yel + (a,))


def render_frame(fi):
    fps = CFG["video"]["fps"]
    t = fi / fps
    i = scene_at(t)
    s = TL[i]
    lt = t - s["start"]
    fade = clamp((s["end"] - t) / 0.15) if i + 1 < len(TL) else 1.0
    frame = background(t, fi)
    draw_card(frame, s, lt, fade)
    draw_text(frame, s, lt, fade)
    draw_cta(frame, s, lt)
    draw_bottom(frame, s, fi)
    return frame


def render_bytes(fi):
    return render_frame(fi).tobytes()


# ------------------------------------------------------------------ QA
def qa_layout():
    ok = True
    v = CFG["video"]
    for s in TL:
        c = s["cfg"].get("card")
        if not c:
            continue
        # approx card top incl. tilt
        h = {"meter": 440, "people": 460, "finance": 470, "chat": 490,
             "bars": 130 + 150 * len(c.get("rows", [])), "notes": 210 + 82 * len(c.get("items", []))}[c["type"]]
        top = v["card"]["center_y"] - h / 2 - v["card"]["width"] / 2 * math.sin(math.radians(abs(s["cfg"].get("tilt", 0)))) - 8
        if s["layout"]["bottom"] > top:
            print(f"!! scene {s['cfg']['id']}: text bottom {s['layout']['bottom']:.0f} overlaps card top {top:.0f}")
            ok = False
        else:
            print(f"   scene {s['cfg']['id']:<12} {s['dur']:.2f}s  lines={s['layout']['lines']}  "
                  f"text bottom {s['layout']['bottom']:.0f} < card top {top:.0f}")
    return ok


def contact_sheet(path, nframes):
    fps = CFG["video"]["fps"]
    picks = []
    for s in TL:  # ~12 frames spread over scenes
        k = 2 if s["dur"] > 2 else 1
        for j in range(k):
            picks.append(s["start"] + s["dur"] * (0.55 + 0.38 * j if k == 2 else 0.7))
    total = TL[-1]["end"]
    picks = list(np.linspace(0.6, total - 0.4, 12))
    tw, th = 360, 640
    sheet = Image.new("RGB", (tw * 6, th * 2), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for n, t in enumerate(picks):
        fi = min(nframes - 1, int(t * fps))
        im = render_frame(fi).resize((tw, th), Image.LANCZOS)
        sheet.paste(im, ((n % 6) * tw, (n // 6) * th))
        d.text(((n % 6) * tw + 8, (n // 6) * th + 6), f"{t:.1f}s", font=F("bold", 24), fill=(255, 255, 0))
    sheet.save(path)


# ------------------------------------------------------------------ main
def main():
    global CFG, TL, WAVE, GRAIN, VIGN
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=os.path.join(ROOT, "scenes.json"))
    ap.add_argument("--contact", action="store_true", help="only render out/contact.png")
    args = ap.parse_args()
    CFG = json.load(open(args.scenes, encoding="utf-8"))
    v = CFG["video"]
    TL = build_timeline(CFG)
    fps = v["fps"]
    total = TL[-1]["end"]
    nframes = int(round(total * fps))
    print(f"duration {total:.2f}s, {nframes} frames")
    qa_layout()
    VIGN, GRAIN = init_static(CFG)
    audio = synth_audio(CFG, int(round(nframes / fps * SR)))
    WAVE = wave_bars(audio, nframes, fps)
    out = os.path.join(ROOT, v.get("output", "out/video.mp4"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    contact = os.path.join(os.path.dirname(out), "contact.png")
    contact_sheet(contact, nframes)
    print("contact sheet:", contact)
    if args.contact:
        return
    wav = os.path.join(os.path.dirname(out), "_audio.wav")
    write_wav(wav, audio)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{v['width']}x{v['height']}", "-r", str(fps), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           "-shortest", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ctx = mp.get_context("fork")
    with ctx.Pool(os.cpu_count()) as pool:
        for n, b in enumerate(pool.imap(render_bytes, range(nframes), chunksize=4)):
            ff.stdin.write(b)
            if n % 60 == 0:
                print(f"\r  frame {n}/{nframes}", end="", flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print(f"\ndone: {out}")


if __name__ == "__main__":
    main()
