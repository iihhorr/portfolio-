#!/usr/bin/env python3
"""Ролик 06 / ЖИЗНЬ — фото-дамп SOVAKAV. Pillow + numpy + FFmpeg.

    python render.py --check    # out/check.png (t = 0.3, 2.88, 3.5, 9.95, 11.7)
    python render.py            # out/video_silent.mp4 + out/video_preview.mp4 + таблица склеек
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, LENGTH = 1080, 1920, 30, 12.0
NFR = int(round(LENGTH * FPS))
SR = 44100
TRACK = os.path.join(ROOT, "input", "track.mp4")
PHOTOS = os.path.join(ROOT, "input", "photos")
OUT = os.path.join(ROOT, "out")

BG = (239, 237, 232)          # #efede8
INK = (10, 10, 10)
GRAY = (107, 107, 107)        # #6b6b6b
LINE = (0, 0, 0, 38)          # чёрные 15 %
M = 70
AREA = (70, 200, 1010, 1500)  # область фото
FOOT_Y = 1540

SHOTS = [2.86, 2.96, 3.08, 3.34, 8.71, 8.94, 9.38, 9.61]
CASH, FINAL = 9.91, 10.31

# монтаж: (время начала, [фото])
EDIT = [
    (0.00, ["01"]), (1.18, ["02"]), (1.56, ["03"]), (1.93, ["04"]), (2.28, ["05"]), (2.65, ["06", "07"]),
    (2.86, ["25"]), (2.96, ["08"]), (3.08, ["09"]), (3.34, ["11"]),
    (4.85, ["10"]), (5.22, ["12"]), (5.60, ["13"]), (6.32, ["14"]), (6.69, ["15"]), (7.04, ["16"]),
    (7.41, ["17"]), (7.78, ["18", "19"]),
    (8.71, ["20"]), (8.94, ["21"]), (9.38, ["22"]), (9.61, ["23"]),
    (CASH, "cash"), (FINAL, "final"),
]
CUTS = [(e[0], int(round(e[0] * FPS)), e[1]) for e in EDIT]


# ------------------------------------------------------------------ шрифты / утилиты
@lru_cache(None)
def F(kind, size):
    """'m700' 'm800' Montserrat; 'i300' 'i500' Inter."""
    path = os.path.join(ROOT, "fonts", "Montserrat-VF.ttf" if kind[0] == "m" else "Inter-VF.ttf")
    f = ImageFont.truetype(path, size)
    w = int(kind[1:])
    f.set_variation_by_axes([w] if kind[0] == "m" else [min(32, max(14, size)), w])
    return f


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def tracked(d, x, y, text, font, fill, em, align="left"):
    w = sum(font.getlength(ch) for ch in text) + font.size * em * (len(text) - 1)
    x = x - w if align == "right" else (x - w / 2 if align == "center" else x)
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + font.size * em
    return w


# ------------------------------------------------------------------ звук → волна
def load_audio():
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", TRACK, "-vn", "-t", str(LENGTH), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)
    n = int(LENGTH * SR)
    return np.pad(a[:n], (0, max(0, n - len(a))))


def wave_bars(audio, nb=40):
    win = 2048
    edges = np.geomspace(40, 9000, nb + 1)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    idx = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(nb)]
    hann = np.hanning(win)
    bands, rms = np.zeros((NFR, nb)), np.zeros(NFR)
    for fi in range(NFR):
        c = int(fi / FPS * SR)
        seg = np.pad(audio[max(0, c - win // 2): c + win // 2], (0, win))[:win]
        sp = np.abs(np.fft.rfft(seg * hann))
        bands[fi] = [sp[ix].mean() if len(ix) else 0 for ix in idx]
        rms[fi] = np.sqrt(np.mean(seg ** 2))
    bands = np.log1p(bands * 4)
    bands /= np.percentile(bands, 97, axis=0) + 1e-6
    rms /= rms.max() + 1e-9
    v = np.clip((0.4 * np.clip(bands, 0, 1) + 0.6) * rms[:, None], 0, 1)
    out = np.zeros_like(v)
    for fi in range(NFR):
        out[fi] = v[fi] if fi == 0 else np.maximum(v[fi], out[fi - 1] * 0.8)
    return out


# ------------------------------------------------------------------ фото
@lru_cache(None)
def photo(name, h_max=None):
    """Ч/б: autocontrast + контраст +12 %; вписано в область без растяжения (с запасом 1.03 под зум)."""
    im = ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, f"{name}.jpg"))).convert("L")
    im = ImageEnhance.Contrast(ImageOps.autocontrast(im, cutoff=0.5)).enhance(1.12).convert("RGB")
    aw, ah = AREA[2] - AREA[0], (h_max or AREA[3] - AREA[1])
    s = min(aw / im.width, ah / im.height)
    tw, th = round(im.width * s), round(im.height * s)
    big = im.resize((math.ceil(tw * 1.03), math.ceil(th * 1.03)), Image.LANCZOS)
    return big, (tw, th)


def zoomed(name, z, h_max=None):
    big, (tw, th) = photo(name, h_max)
    # z=1.00 — всё фото целиком, z=1.03 — центральная часть на 3 % крупнее (без обрезки краёв на старте)
    cw, ch = big.width / z, big.height / z
    x, y = (big.width - cw) / 2, (big.height - ch) / 2
    return big.resize((tw, th), Image.BICUBIC, box=(x, y, x + cw, y + ch))


def framed(im):
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width - 1, im.height - 1), outline=INK, width=2)
    return im


def photo_layer(names, z):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if len(names) == 1:
        im = framed(zoomed(names[0], z))
        x = AREA[0] + (AREA[2] - AREA[0] - im.width) // 2
        y = AREA[1] + (AREA[3] - AREA[1] - im.height) // 2
        layer.paste(im, (x, y))
    else:                                            # пара: по 630 px в высоту, зазор 35
        ims = [framed(zoomed(n, z, 630)) for n in names]
        total = sum(i.height for i in ims) + 35
        y = AREA[1] + (AREA[3] - AREA[1] - total) // 2
        for im in ims:
            layer.paste(im, (AREA[0] + (AREA[2] - AREA[0] - im.width) // 2, y))
            y += im.height + 35
    return layer


# ------------------------------------------------------------------ рамка, текст
def grain_tiles():
    rng = np.random.default_rng(3)
    return [rng.normal(0, 255 * 0.035, (H, W)).astype(np.float32)[..., None] for _ in range(6)]


def add_grain(img, fi):
    a = np.asarray(img, np.float32) + GRAIN[fi % len(GRAIN)]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def chrome(img, fi, inverted=False):
    fg, gray = ((242, 242, 240), (180, 180, 180)) if inverted else (INK, GRAY)
    line = (255, 255, 255, 60) if inverted else LINE
    d = ImageDraw.Draw(img, "RGBA")
    tracked(d, M, 100, "06 / ЖИЗНЬ", F("i500", 22), gray, 0.18)
    tracked(d, W - M, 100, "SOVAKAV", F("m700", 26), fg, 0.24, "right")
    d.line((M, 145, W - M, 145), fill=line, width=1)
    d.line((M, FOOT_Y, W - M, FOOT_Y), fill=line, width=1)
    cx, cy, r = M + 36, FOOT_Y + 20 + 36, 36
    circ, letters = ((242, 242, 240), INK) if inverted else (INK, (242, 242, 240))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=circ)
    d.text((cx, cy), "SV", font=F("m700", 24), fill=letters, anchor="mm")
    tracked(d, M + 92, cy + 7, "САЙТЫ • ИИ • 2026", F("i500", 20), gray, 0.18)
    vals = WAVE[min(fi, NFR - 1)]
    x0, x1, base, hmax = 640, W - M, cy + 28, 56
    gap = (x1 - x0 - 2 * 40) / 39
    for k, v in enumerate(vals):
        h = max(2, v * hmax)
        x = x0 + k * (2 + gap)
        d.rectangle((x, base - h, x + 2, base), fill=fg + (220,))


def plate(img, box):
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rectangle(box, fill=BG + (int(255 * 0.92),))
    img.paste(ov, (0, 0), ov)


def hook(img):
    plate(img, (70, 170, 1010, 560))
    d = ImageDraw.Draw(img)
    f = F("m800", 108)
    asc = f.getmetrics()[0]
    for i, ln in enumerate(["ГОД НАЗАД", "Я ЖИЛ", "ПО-ДРУГОМУ."]):
        d.text((100, 170 + 22 + asc * 0.92 + i * 118), ln, font=f, fill=INK, anchor="ls")


def laptop_text(img, t):
    text = "всё это — с ноутбука"
    n = min(len(text), int((t - 3.34) / 0.03) + 1)
    plate(img, (70, 1370, 1010, 1470))
    ImageDraw.Draw(img).text((W / 2 - F("i300", 52).getlength(text) / 2, 1438), text[:n],
                             font=F("i300", 52), fill=INK, anchor="ls")


def cash(img):
    ImageDraw.Draw(img).text((W / 2, H / 2), "+1 КЛИЕНТ", font=F("m800", 120), fill=(242, 242, 240), anchor="mm")


def final(img, t):
    d = ImageDraw.Draw(img, "RGBA")
    f = F("m800", 118)
    lines = [("ХОЧЕШЬ", 10.33), ("ТАК ЖЕ?", 10.70), ("ПОЙДЁМ", 11.08), ("СО МНОЙ.", 11.45)]
    lh = 118 * 1.1
    top = 600
    for i, (ln, bt) in enumerate(lines):
        p = clamp((t - bt) / 0.12)
        if p > 0:
            d.text((W / 2, top + i * lh + (1 - ease(p)) * 10), ln, font=f, fill=INK + (int(255 * p),), anchor="ms")
    if t >= 11.45:
        p = clamp((t - 11.45) / 0.12)
        lw = max(f.getlength(ln) for ln, _ in lines)
        y = top + 3 * lh + 60
        d.rectangle((W / 2 - lw / 2, y, W / 2 + lw / 2, y + 3), fill=INK + (int(255 * p),))
        d.text((W / 2, y + 80), "сайты с ИИ — гайд в профиле", font=F("i500", 36), fill=GRAY + (int(255 * p),),
               anchor="ms")


# ------------------------------------------------------------------ кадр
def cut_index(fi):
    j = 0
    for i, (_, f0, _) in enumerate(CUTS):
        if f0 <= fi:
            j = i
    return j


def shot_fx(fi):
    """Для кадра: (белая вспышка 0..1, толчок-зум, сдвиг тряски)."""
    for s in SHOTS:
        f0 = int(round(s * FPS))
        k = fi - f0
        if 0 <= k < 2:
            return (0.85 if k == 0 else 0.45), 1.06, (0, 0)
        if 2 <= k < 2 + int(round(0.15 * FPS)) + 2:
            pz = 1.06 + (1.0 - 1.06) * ease((k - 2) / (0.15 * FPS))
            sh = (0, 0)
            if k - 2 < 4:                              # тряска ±6 px на 4 кадра
                rng = np.random.default_rng(f0 * 10 + k)
                sh = tuple(int(v) for v in rng.choice([-6, -4, 4, 6], 2))
            return 0.0, pz, sh
    return 0.0, 1.0, (0, 0)


def render_frame(fi):
    t = fi / FPS
    j = cut_index(fi)
    start, f0, what = CUTS[j]
    f_end = CUTS[j + 1][1] if j + 1 < len(CUTS) else NFR
    if what == "cash":
        img = Image.new("RGB", (W, H), INK)
        cash(img)
        img = add_grain(img, fi)
        chrome(img, fi, inverted=True)
        return img
    img = Image.new("RGB", (W, H), BG)
    if what == "final":
        final(img, t)
    else:
        z = 1.0 + 0.03 * (fi - f0) / max(1, f_end - f0)     # медленный зум 1.00→1.03
        flash, push, (sx, sy) = shot_fx(fi)
        layer = photo_layer(what, z)
        if push != 1.0:
            cw, ch = W / push, H / push
            layer = layer.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        img.paste(layer, (sx, sy), layer)                 # тряска = сдвиг слоя
        if what == ["01"]:
            hook(img)
        if what == ["11"]:
            laptop_text(img, t)
        if flash > 0:                                     # белая вспышка 2 кадра
            img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), flash)
        d = ImageDraw.Draw(img)                           # рамка и подвал не трясутся
        d.rectangle((0, 0, W, 190), fill=BG)
        d.rectangle((0, FOOT_Y - 30, W, H), fill=BG)
    img = add_grain(img, fi)
    chrome(img, fi)
    return img


def render_bytes(fi):
    return render_frame(fi).tobytes()


# ------------------------------------------------------------------ main
def main():
    global WAVE, GRAIN
    os.makedirs(OUT, exist_ok=True)
    GRAIN = grain_tiles()
    WAVE = wave_bars(load_audio())
    if "--check" in sys.argv:
        ts = [0.3, 2.88, 3.5, 9.95, 11.7]
        sheet = Image.new("RGB", (5 * 540 + 40, 960), (128, 128, 128))
        for i, t in enumerate(ts):
            fi = int(round(t * FPS))
            im = render_frame(fi)
            im.save(os.path.join(OUT, f"check_{t}.png"))
            im = im.resize((540, 960), Image.LANCZOS)
            ImageDraw.Draw(im).text((12, 200), f"{t}s (кадр {fi})", font=F("m700", 24), fill=(200, 30, 30))
            sheet.paste(im, (i * 550, 0))
        sheet.save(os.path.join(OUT, "check.png"))
        print(os.path.join(OUT, "check.png"))
        return
    silent = os.path.join(OUT, "video_silent.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-frames:v", str(NFR), "-c:v", "libx264", "-preset", "medium",
                           "-crf", "20", "-maxrate", "12M", "-bufsize", "24M", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", silent], stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        for b in pool.imap(render_bytes, range(NFR), chunksize=4):
            ff.stdin.write(b)
    ff.stdin.close(); ff.wait()
    preview = os.path.join(OUT, "video_preview.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-t", str(LENGTH), "-i", TRACK,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-t", str(LENGTH), "-movflags", "+faststart",
                    preview], check=True)
    print("\n  событие  пик (с)  кадр склейки  время склейки  Δ кадров")
    for s in SHOTS + [CASH, FINAL]:
        f0 = int(round(s * FPS))
        print(f"  {'SHOT' if s in SHOTS else ('CASH' if s == CASH else 'FINAL'):<7} {s:7.2f}  {f0:12d}  {f0 / FPS:13.3f}"
              f"  {(f0 / FPS - s) * FPS:+8.2f}")
    print(silent); print(preview)


if __name__ == "__main__":
    main()
