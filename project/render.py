#!/usr/bin/env python3
"""Photo-dump on the beat.  python render.py  ->  out/video_silent.mp4, out/video_preview.mp4, out/contact.png"""
import glob
import json
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

import beats as B

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT, OUT = B.INPUT, B.OUT
W, H, FPS = 1080, 1920, 30
BG = (247, 245, 242)              # #f7f5f2
SAFE_BOTTOM, SAFE_RIGHT = 320, 140
DEFAULT_CFG = {"signature": "@ip.atelier", "final_line": "ты можешь так же", "pairs": [5, 9], "drop_time": None}


# ------------------------------------------------------------------ fonts
@lru_cache(None)
def font(kind, size):
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", {"hand": "Caveat-VF.ttf", "thin": "Manrope-VF.ttf"}[kind]), size)
    f.set_variation_by_axes([500 if kind == "hand" else 300])
    return f


# ------------------------------------------------------------------ photos
def load_photos():
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pillow_heif = None
    files = sorted(p for p in glob.glob(os.path.join(INPUT, "photos", "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp")))
    if not files:
        raise SystemExit("Нет фото: положи 12–20 файлов в input/photos/ (01.jpg, 02.jpg …)")
    if not 12 <= len(files) <= 20:
        print(f"!! фото: {len(files)} (рекомендуется 12–20)")
    photos = []
    for p in files:
        if p.lower().endswith((".heic", ".heif")) and pillow_heif is None:
            raise SystemExit("Для HEIC: pip install pillow-heif")
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        if min(im.size) < 1000:
            print(f"!! {os.path.basename(p)}: {im.width}×{im.height} — меньше 1000 px, может пикселить")
        photos.append((os.path.basename(p), grade(im)))
    return photos


def grade(im):
    """One look for the whole set: normalised exposure, contrast -5 %, lifted shadows, warm +3 %, film grain 4 %."""
    a = np.asarray(im, np.float32) / 255.0
    lum = (a @ np.array([0.299, 0.587, 0.114], np.float32))
    g = math.log(0.48) / math.log(max(1e-3, min(0.95, float(np.median(lum)))))   # median luminance -> 0.48
    a = np.clip(a, 1e-4, 1) ** float(np.clip(g, 0.7, 1.4))
    gray = (a @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    a = gray + (a - gray) * 0.9                       # slightly calmer saturation, unified
    a = 0.5 + (a - 0.5) * 0.95                        # contrast -5 %
    a = 0.045 + a * (1 - 0.045)                       # lifted shadows (faded blacks)
    a *= np.array([1.03, 1.0, 0.97], np.float32)      # warm +3 %
    rng = np.random.default_rng(hash(im.size) & 0xFFFF)
    a += rng.normal(0, 0.04 * 0.5, a.shape[:2])[..., None].astype(np.float32)   # grain 4 %
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def fit(im, single=True):
    """Target display size: landscape/square -> width 860, portrait -> height 1100; pairs -> width 760."""
    w, h = im.size
    if single:
        tw, th = (860, round(860 * h / w)) if w >= h else (round(1100 * w / h), 1100)
    else:
        tw, th = 760, round(760 * h / w)
        if th > 860:                                   # keep a stacked pair on screen
            tw, th = round(860 * w / h), 860
    return tw, th


@lru_cache(64)
def prepared(idx, tw, th):
    """Pre-scaled 1.03× source so the slow zoom never upsamples twice."""
    im = PHOTOS[idx][1]
    zw, zh = math.ceil(tw * 1.03), math.ceil(th * 1.03)
    if im.width < zw or im.height < zh:
        pass  # small source: will upscale (warned at load time)
    return im.resize((zw, zh), Image.LANCZOS)


def zoomed(idx, tw, th, z):
    """Crop-zoom 1.00 -> 1.03 inside a fixed frame."""
    src = prepared(idx, tw, th)
    cw, ch = src.width / z, src.height / z
    x, y = (src.width - cw) / 2, (src.height - ch) / 2
    return src.resize((tw, th), Image.BICUBIC, box=(x, y, x + cw, y + ch))


@lru_cache(64)
def shadow(tw, th):
    pad = 60
    m = Image.new("L", (tw + pad * 2, th + pad * 2), 0)
    ImageDraw.Draw(m).rectangle((pad, pad + 10, pad + tw, pad + th + 10), fill=int(255 * 0.12))
    return m.filter(ImageFilter.GaussianBlur(30)), pad


def place(frame, im, cx, cy):
    tw, th = im.size
    sh, pad = shadow(tw, th)
    x, y = int(cx - tw / 2), int(cy - th / 2)
    frame.paste((0, 0, 0), (x - pad, y - pad), sh)
    frame.paste(im, (x, y))


# ------------------------------------------------------------------ background / text
def paper():
    rng = np.random.default_rng(11)
    n = rng.normal(0, 255 * 0.03 * 0.5, (H // 2, W // 2)).astype(np.float32)
    n = np.asarray(Image.fromarray(n).resize((W, H), Image.BICUBIC))          # soft paper fibre
    n2 = rng.normal(0, 255 * 0.03 * 0.25, (H, W)).astype(np.float32)
    a = np.array(BG, np.float32)[None, None, :] + (n + n2)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@lru_cache(None)
def signature_img(text):
    f = font("hand", 54)
    l, t, r, b = f.getbbox(text)
    im = Image.new("RGBA", (r - l + 40, b - t + 40), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((20 - l, 20 - t), text, font=f, fill=(0, 0, 0, 204))   # black 80 %
    return im.rotate(4, resample=Image.BICUBIC, expand=True)                        # tilt −4° (rising)


def sig_pos(im, cx, cy):
    x, y = int(cx - im.width / 2), int(cy - im.height / 2)
    x = min(x, W - SAFE_RIGHT - im.width)
    y = min(y, H - SAFE_BOTTOM - im.height)
    return x, y


def draw_signature(frame, cx=None, cy=None):
    im = signature_img(CFG["signature"])
    cx = W * 0.64 if cx is None else cx
    cy = H * 0.5 if cy is None else cy
    frame.paste(im, sig_pos(im, cx, cy), im)


def draw_final(frame, t_in):
    text = CFG["final_line"]
    n = min(len(text), int(t_in / 0.04) + 1)
    f = font("thin", 64)
    full = f.getlength(text)
    if full > W - 2 * 90 - SAFE_RIGHT:
        print("!! final_line слишком длинная для безопасной зоны")
    x = (W - full) / 2
    d = ImageDraw.Draw(frame)
    d.text((x, H * 0.45), text[:n], font=f, fill=(20, 20, 20), anchor="ls")
    draw_signature(frame, W / 2 + 40, H * 0.45 + 80)


# ------------------------------------------------------------------ timeline
def build():
    global CFG, PHOTOS, SLOTS, LENGTH, DATA, NFR, ASSIGN, STEP
    cfgp = os.path.join(INPUT, "config.json")
    if not os.path.exists(cfgp):
        os.makedirs(INPUT, exist_ok=True)
        json.dump(DEFAULT_CFG, open(cfgp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("создан шаблон input/config.json")
    CFG = {**DEFAULT_CFG, **json.load(open(cfgp, encoding="utf-8"))}
    DATA = B.analyse(CFG)
    PHOTOS = load_photos()
    LENGTH, SLOTS, STEP = B.schedule(DATA, CFG, FPS)
    NFR = int(round(LENGTH * FPS))
    for s in SLOTS:
        s["frame"] = int(round(s["t"] * FPS))
    ASSIGN, k = [], 0
    for s in SLOTS:                                   # photos in file order, cycling if needed
        if s["kind"] == "photo":
            ASSIGN.append([k % len(PHOTOS)]); k += 1
        elif s["kind"] == "pair":
            ASSIGN.append([k % len(PHOTOS), (k + 1) % len(PHOTOS)]); k += 2
        else:
            ASSIGN.append([])
    if k > len(PHOTOS):
        print(f"   фото по кругу: {k} показов на {len(PHOTOS)} фото")


def slot_at(fi):
    j = 0
    for i, s in enumerate(SLOTS):
        if s["frame"] <= fi:
            j = i
    return j


def render_frame(fi, base_paper):
    j = slot_at(fi)
    s = SLOTS[j]
    if s["kind"] == "flash" and fi < s["frame"] + 2:
        frame = Image.new("RGB", (W, H), (255, 255, 255))
        draw_signature(frame)
        return frame
    frame = base_paper.copy()
    if s["kind"] == "final":
        draw_final(frame, (fi - s["frame"]) / FPS)
        return frame
    if s["kind"] == "flash":                           # (2 flash frames consumed; next slot starts right after)
        j += 1
        s = SLOTS[j]
    end = SLOTS[j + 1]["frame"] if j + 1 < len(SLOTS) else NFR
    p = (fi - s["frame"]) / max(1, end - s["frame"])
    z = 1.0 + 0.03 * p
    ids = ASSIGN[j]
    if s["kind"] == "pair":
        sizes = [fit(PHOTOS[i][1], single=False) for i in ids]
        total = sizes[0][1] + sizes[1][1] + 40
        y = (H - total) / 2
        for i, (tw, th) in zip(ids, sizes):
            place(frame, zoomed(i, tw, th, z), W / 2, y + th / 2)
            y += th + 40
    else:
        tw, th = fit(PHOTOS[ids[0]][1])
        place(frame, zoomed(ids[0], tw, th, z), W / 2, H / 2)
    draw_signature(frame)
    return frame


# ------------------------------------------------------------------ QA + output
def qa_table():
    print("\n  №  тип     бит (с)   склейка (с)  кадр  Δ кадров  фото")
    bad = 0
    for s, ids in zip(SLOTS, ASSIGN):
        cut = s["frame"] / FPS
        dfr = (cut - s["src_beat"]) * FPS
        if s.get("fast") and abs(s["t"] - s["src_beat"]) > 1e-6:
            dfr = (cut - (s["src_beat"] + 2 / FPS)) * FPS   # first post-drop photo follows the 2-frame flash
        bad += abs(dfr) > 1.0
        names = ", ".join(PHOTOS[i][0] for i in ids) or "—"
        print(f" {s['beat_no']:>2}  {s['kind']:<6} {s['src_beat']:8.3f}   {cut:9.3f}   {s['frame']:4d}   {dfr:+5.2f}    {names}")
    print("  всё в пределах 1 кадра" if not bad else f"!! {bad} склеек дальше 1 кадра")


def contact(path, base_paper):
    drop_f = next(s["frame"] for s in SLOTS if s["kind"] == "flash")
    fin_f = next(s["frame"] for s in SLOTS if s["kind"] == "final")
    picks = sorted(set([0, drop_f, drop_f + 3, NFR - 1] +
                       [int(x) for x in np.linspace(5, drop_f - 5, 6)] + [drop_f + 20, fin_f + 10]))[:12]
    tw, th = 270, 480
    sheet = Image.new("RGB", (tw * 6, th * 2 + 0), (200, 200, 200))
    d = ImageDraw.Draw(sheet)
    for n, fi in enumerate(picks):
        im = render_frame(fi, base_paper).resize((tw - 4, th - 4), Image.LANCZOS)
        sheet.paste(im, ((n % 6) * tw + 2, (n // 6) * th + 2))
        tag = f"{fi / FPS:.2f}s" + (" DROP" if fi == drop_f else "") + (" FINAL" if fi == NFR - 1 else "")
        d.text(((n % 6) * tw + 8, (n // 6) * th + 6), tag, fill=(220, 0, 0), font=font("thin", 22))
    sheet.save(path)


def main():
    build()
    base = paper()
    print(f"BPM {DATA['bpm']} (фото на {STEP} бит), drop {DATA['drop']} s, track_offset {DATA['track_offset']} s, "
          f"длина {LENGTH:.2f} s ({NFR} кадров), дроп на {100 * (DATA['drop'] - DATA['track_offset']) / LENGTH:.0f}%")
    qa_table()
    contact(os.path.join(OUT, "contact.png"), base)
    if "--contact" in sys.argv:
        return
    silent = os.path.join(OUT, "video_silent.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent], stdin=subprocess.PIPE)
    for fi in range(NFR):
        ff.stdin.write(render_frame(fi, base).tobytes())
    ff.stdin.close()
    ff.wait()
    preview = os.path.join(OUT, "video_preview.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-ss", str(DATA["track_offset"]), "-t", f"{LENGTH:.3f}",
                    "-i", os.path.join(ROOT, DATA["source"]), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                    "-shortest", "-movflags", "+faststart", preview], check=True)
    print(f"\n{silent}\n{preview}\n{os.path.join(OUT, 'contact.png')}")


if __name__ == "__main__":
    main()
