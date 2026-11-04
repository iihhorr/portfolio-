#!/usr/bin/env python3
"""Ролик 05 / МЫСЛЬ — SOVAKAV. Pillow + numpy + FFmpeg.

    python render.py --check     # шаг 1: out/check_*.png + out/check.png
    python render.py             # шаг 2: out/video_silent.mp4 + out/video_preview.mp4
"""
import math
import multiprocessing as mp
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, LENGTH = 1080, 1920, 30, 14.5
NFR = int(round(LENGTH * FPS))
SR = 44100
TRACK = os.path.join(ROOT, "input", "track.wav")
SHOT = os.path.join(ROOT, "input", "client_msg_blurred.png")
OUT = os.path.join(ROOT, "out")

BEATS = [0.07, 0.60, 1.11, 1.65, 2.18, 2.74, 3.27, 3.83, 4.39, 4.95, 5.46, 5.94, 6.46, 6.94, 7.45, 7.99,
         8.54, 9.10, 9.66, 10.22, 10.75, 11.28, 11.84, 12.38, 12.93, 13.47, 14.00]
FLASHES = [5.46, 11.28]

# палитра
DARK = dict(bg=(10, 10, 10), fg=(242, 242, 240), gray=(138, 138, 138), line=(255, 255, 255, 38))
LIGHT = dict(bg=(239, 237, 232), fg=(10, 10, 10), gray=(138, 138, 138), line=(0, 0, 0, 38))

M = 70                                   # поля
CARD = (70, 820, 1010, 1300)             # x0, y0, x1, y1
FOOT_Y = 1540


# ------------------------------------------------------------------ шрифты
@lru_cache(None)
def F(kind, size):
    """'m800' 'm700' (Montserrat) | 'i400' 'i500' (Inter)."""
    path = os.path.join(ROOT, "fonts", "Montserrat-VF.ttf" if kind[0] == "m" else "Inter-VF.ttf")
    f = ImageFont.truetype(path, size)
    w = int(kind[1:])
    f.set_variation_by_axes([w] if kind[0] == "m" else [min(32, max(14, size)), w])
    return f


TNUM = ["tnum"]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(t):                              # easeOutCubic
    t = clamp(t)
    return 1 - (1 - t) ** 3


def rgba(c, a):
    base = c[3] if len(c) == 4 else 255
    return tuple(c[:3]) + (int(clamp(a) * base),)


def caps_label(d, x, y, text, th, size=20, align="left", a=1.0, color=None, track=0.18):
    f = F("i500", size)
    text = text.upper()
    w = sum(f.getlength(ch) for ch in text) + f.size * track * (len(text) - 1)
    x = x - w if align == "right" else x
    for ch in text:
        d.text((x, y), ch, font=f, fill=rgba(color or th["gray"], a), anchor="ls")
        x += f.getlength(ch) + f.size * track
    return w


# ------------------------------------------------------------------ сцены
# words: (текст, акцент); times — появление слов (на биты)
SCENES = [
    dict(start=0.00, end=1.11, th=DARK, kind="hook"),
    dict(start=1.11, end=3.27, th=DARK, words=[("РАНЬШЕ:", True)], times=[1.11], card="time"),
    dict(start=3.27, end=4.39, th=DARK, words=[("0 КЛИЕНТОВ.", False), ("€0.", True)], times=[3.27, 3.83],
         remark=("(работал официантом)", 3.83), card="zero"),
    dict(start=4.39, end=5.46, th=DARK, kind="center",
         words=[("А", False), ("ПОТОМ", False), ("Я", False), ("ПОНЯЛ", False), ("ОДНУ", True), ("ВЕЩЬ", True)],
         times=[4.39, 4.50, 4.62, 4.73, 4.84, 4.95]),
    dict(start=5.46, end=8.54, th=LIGHT, words=[("МОЙ", False), ("ПЛАН", True)], times=[5.46, 5.94], card="plan"),
    dict(start=8.54, end=11.28, th=DARK,
         words=[("КЛИЕНТЫ", False), ("СО", True), ("ВСЕГО", True), ("МИРА", True)],
         times=[8.54, 9.10, 9.66, 10.22], card="shot"),
    dict(start=11.28, end=14.50, th=LIGHT,
         words=[("КАК", False), ("Я", False), ("ЭТО", False), ("СДЕЛАЛ", True)],
         times=[11.28, 11.84, 12.38, 12.93], card="list"),
]


def scene_at(t):
    for s in SCENES:
        if t < s["end"]:
            return s
    return SCENES[-1]


# ------------------------------------------------------------------ звук → волна
def load_audio():
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", TRACK, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)[: int(LENGTH * SR)]
    return np.pad(a, (0, max(0, int(LENGTH * SR) - len(a))))


def wave_bars(audio, nb=40):
    """Высота столбиков от громкости трека (RMS) × лёгкая спектральная форма."""
    win = 2048
    edges = np.geomspace(40, 9000, nb + 1)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    idx = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(nb)]
    hann = np.hanning(win)
    bands, rms = np.zeros((NFR, nb)), np.zeros(NFR)
    for fi in range(NFR):
        c = int(fi / FPS * SR)
        seg = audio[max(0, c - win // 2): c + win // 2]
        seg = np.pad(seg, (0, win - len(seg)))
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


# ------------------------------------------------------------------ рисование
def grain_tiles():
    rng = np.random.default_rng(3)
    return [rng.normal(0, 255 * 0.03, (H, W)).astype(np.float32)[..., None] for _ in range(6)]


def background(th, fi):
    a = np.array(th["bg"], np.float32)[None, None, :] + GRAIN[fi % len(GRAIN)]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def draw_frame_chrome(img, th, fi):
    d = ImageDraw.Draw(img, "RGBA")
    caps_label(d, M, 110, "05 / Мысль", th, size=22)
    f = F("m700", 26)
    d.text((W - M, 110), "SOVAKAV", font=f, fill=th["fg"], anchor="rs")
    d.line((M, 134, W - M, 134), fill=th["line"], width=1)
    d.line((M, FOOT_Y, W - M, FOOT_Y), fill=th["line"], width=1)
    cx, cy, r = M + 36, FOOT_Y + 20 + 36, 36
    dark = th is DARK
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(10, 10, 10) if dark else (242, 242, 240), outline=th["line"])
    d.text((cx, cy), "SV", font=F("m700", 24), fill=(242, 242, 240) if dark else (10, 10, 10), anchor="mm")
    caps_label(d, M + 92, cy + 7, "Сайты • ИИ • 2026", th, size=20)
    vals = WAVE[min(fi, NFR - 1)]
    x0, x1, base, hmax = 640, W - M, cy + 28, 56
    gap = (x1 - x0 - 2 * 40) / 39
    for k, v in enumerate(vals):
        h = max(2, v * hmax)
        x = x0 + k * (2 + gap)
        d.rectangle((x, base - h, x + 2, base), fill=th["fg"] + (220,))


def headline(img, s, t, x0=M, y0=200, size=88, center=False):
    """Слова по одному на биты: fade 0.12 с + подъём 10 px. Акцент — основной цвет, остальное — серый."""
    th = s["th"]
    d = ImageDraw.Draw(img, "RGBA")
    f = F("m800", size)
    space = f.getlength(" ")
    lines, cur, cw = [], [], 0
    for i, (w, acc) in enumerate(s["words"]):
        ww = f.getlength(w)
        if cur and cw + space + ww > (830 if center else 940):   # с запасом под зум 1.05
            lines.append((cur, cw)); cur, cw = [], 0
        cur.append((i, ww)); cw += (space if len(cur) > 1 else 0) + ww
    lines.append((cur, cw))
    lh = size * 1.1
    asc = f.getmetrics()[0]
    top = (H * 0.47 - len(lines) * lh / 2) if center else y0
    for li, (ln, lw) in enumerate(lines):
        x = (W - lw) / 2 if center else x0
        base = top + asc + li * lh
        for i, ww in ln:
            p = clamp((t - s["times"][i]) / 0.12)
            if p > 0:
                w, acc = s["words"][i]
                col = th["fg"] if acc else th["gray"]
                d.text((x, base + (1 - ease(p)) * 10), w, font=f, fill=col + (int(255 * p),), anchor="ls")
            x += ww + space
    return top + asc + (len(lines) - 1) * lh


def card_box(th):
    """1.5 px обводка (рисуется в 2× и уменьшается), без заливки, скругление 6."""
    w, h = CARD[2] - CARD[0], CARD[3] - CARD[1]
    big = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle((1, 1, w * 2 - 2, h * 2 - 2), 12, outline=th["line"], width=3)
    return big.resize((w, h), Image.LANCZOS)


def bar(d, x, y, w, frac, th):
    d.rectangle((x, y, x + w, y + 10), fill=th["line"])
    if frac > 0:
        d.rectangle((x, y, x + w * clamp(frac), y + 10), fill=th["fg"])


def fmt(v):
    return f"{int(round(v)):,}".replace(",", " ")


def card_time(c, th, lt, t):
    d = ImageDraw.Draw(c, "RGBA")
    w = c.width
    caps_label(d, 40, 60, "Время на 1 сайт", th)
    caps_label(d, w - 40, 60, "Один проект", th, align="right")
    d.text((36, 290), "2 НЕДЕЛИ", font=F("m700", 150), fill=th["fg"], anchor="ls")
    frac = 0.0
    for k, bt in enumerate([1.65, 2.18, 2.74]):          # три рывка до 90 %
        frac += 0.3 * ease((t - bt) / 0.15)
    bar(d, 40, 380, w - 80, frac, th)


def card_zero(c, th, lt, t):
    d = ImageDraw.Draw(c, "RGBA")
    w = c.width
    caps_label(d, 40, 60, "Клиенты и доход", th)
    caps_label(d, w - 40, 60, "В месяц", th, align="right")
    rows = [("Клиенты в месяц", "0–1"), ("Доход", "€0")]
    for i, (lab, val) in enumerate(rows):
        y = 250 + i * 190
        caps_label(d, 40, y - 50, lab, th, size=20)
        d.text((w - 40, y), val, font=F("m700", 150), fill=th["fg"], anchor="rs")
        if i == 0:
            d.line((40, y + 30, w - 40, y + 30), fill=th["line"], width=1)


def card_plan(c, th, lt, t):
    d = ImageDraw.Draw(c, "RGBA")
    w = c.width
    caps_label(d, 40, 60, "Мой план", th)
    caps_label(d, w - 40, 60, "3 месяца", th, align="right")
    vals = [500, 1000, 1500]
    heights = [130, 255, 380]                      # мин ~130, макс ~380 px
    times = [5.46, 5.94, 6.46]
    base_y, cw, step = 440, 130, 210
    for i, (v, hmax, bt) in enumerate(zip(vals, heights, times)):
        x = 85 + i * step
        p = ease((t - bt) / 0.45)
        hh = hmax * p
        if p > 0:
            d.rectangle((x, base_y - hh, x + cw, base_y), fill=th["fg"])
            d.text((x + cw / 2, base_y - hh - 14), "+€" + fmt(v * p), font=F("m700", 56),
                   fill=rgba(th["fg"], clamp(p * 3)), anchor="ms", features=TNUM)
        d.line((x, base_y, x + cw, base_y), fill=th["line"], width=1)
        caps_label(d, x, base_y + 28, f"Месяц {i + 1}", th)
    if t >= 6.94:                                  # цель — внизу справа
        a = ease((t - 6.94) / 0.3)
        p = clamp((t - 6.94) / (7.99 - 6.94))
        p = 1 - (1 - p) ** 2
        caps_label(d, w - 40, 370, "Цель накоплений", th, a=a, color=th["fg"], align="right")
        d.text((w - 40, 450), "€" + fmt(5000 * p), font=F("m700", 72), fill=rgba(th["fg"], a), anchor="rs",
               features=TNUM)
        caps_label(d, w - 40, 470, "план", th, a=a, size=18, align="right")


@lru_cache(None)
def shot_bw():
    im = Image.open(SHOT).convert("L")
    im = ImageEnhance.Contrast(im).enhance(1.1).convert("RGB")
    return im.resize((940, round(im.height * 940 / im.width)), Image.LANCZOS)


def draw_shot(img, s, lt):
    """Ч/б скриншот 940 px, рамка 1.5 px; увеличение 1.00→1.06 внутри рамки (рамка не выходит за поля)."""
    th = s["th"]
    src = shot_bw()
    z = 1.0 + 0.06 * clamp(lt / (s["end"] - s["start"]))
    cw, ch = src.width / z, src.height / z
    box = ((src.width - cw) / 2, (src.height - ch) / 2, (src.width + cw) / 2, (src.height + ch) / 2)
    im = src.resize(src.size, Image.BICUBIC, box=box)
    e = ease(lt / 0.3)
    a = clamp(lt / 0.3)
    x = M
    y = int((CARD[1] + CARD[3]) / 2 - im.height / 2 + (1 - e) * 40)
    HS = 90                                          # шапка рамки
    lw, lh = im.width, im.height + HS
    y -= HS // 2
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    layer.paste(im, (0, HS))
    big = Image.new("RGBA", (lw * 2, lh * 2), (0, 0, 0, 0))
    ImageDraw.Draw(big).rectangle((1, 1, lw * 2 - 2, lh * 2 - 2), outline=th["line"], width=3)
    layer.alpha_composite(big.resize((lw, lh), Image.LANCZOS))
    ld = ImageDraw.Draw(layer)
    caps_label(ld, 40, 56, "Переписка", th)
    caps_label(ld, lw - 40, 56, "Upwork", th, align="right")
    ld.line((0, HS - 1, lw, HS - 1), fill=th["line"], width=1)
    if a < 1:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
    img.paste(layer, (x, y), layer)


def check_icon(d, x, y, th, a):
    d.rectangle((x, y, x + 40, y + 40), outline=rgba(th["fg"], a), width=2)
    d.line([(x + 9, y + 21), (x + 17, y + 30), (x + 32, y + 11)], fill=rgba(th["fg"], a), width=2)


def card_list(c, th, lt, t):
    d = ImageDraw.Draw(c, "RGBA")
    w = c.width
    caps_label(d, 40, 60, "Чек-лист", th)
    caps_label(d, w - 40, 60, "С нуля", th, align="right")
    items = ["ИИ-ИНСТРУМЕНТЫ", "ПРОМПТЫ", "ПЕРВЫЕ КЛИЕНТЫ"]
    for i, (it, bt) in enumerate(zip(items, [11.28, 11.84, 12.38])):
        a = ease((t - bt) / 0.12)
        if a <= 0:
            continue
        y = 100 + i * 120
        if i:
            d.line((40, y - 25, w - 40, y - 25), fill=rgba(th["line"], a), width=1)
        check_icon(d, 40, y + 20 + (1 - a) * 10, th, a)
        d.text((110, y + 58 + (1 - a) * 10), it, font=F("m800", 52), fill=rgba(th["fg"], a), anchor="ls")


CARDS = {"time": card_time, "zero": card_zero, "plan": card_plan, "list": card_list}


def draw_card(img, s, t):
    lt = t - s["start"]
    th = s["th"]
    if s.get("card") == "shot":
        draw_shot(img, s, lt)
        return
    if not s.get("card"):
        return
    c = card_box(th)
    CARDS[s["card"]](c, th, lt, t)
    e, a = ease(lt / 0.3), clamp(lt / 0.3)
    if a < 1:
        c.putalpha(c.getchannel("A").point(lambda v: int(v * a)))
    img.paste(c, (CARD[0], int(CARD[1] + (1 - e) * 40)), c)


def draw_cta(img, s, t):
    if t < 12.93:
        return
    th = s["th"]
    d = ImageDraw.Draw(img, "RGBA")
    p = clamp((t - 12.93) / 0.12)
    y = (1 - ease(p)) * 10
    f = F("m800", 60)
    col = th["fg"] + (int(255 * p),)
    d.text((M, 1405 + y), "БЕСПЛАТНЫЙ ГАЙД —", font=f, fill=col, anchor="ls")
    d.text((M, 1480 + y), "ССЫЛКА В ПРОФИЛЕ", font=f, fill=col, anchor="ls")
    # стрелка ↘ линиями 90×90, 6 px; мигает на 13.47 и 14.00
    blink = 1.0
    for bt in (13.47, 14.00):
        if bt <= t < bt + 0.2:
            blink = 0.15
    ac = th["fg"] + (int(255 * p * blink),)
    ax = M + f.getlength("ССЫЛКА В ПРОФИЛЕ") + 24
    ay = 1480 + y - 82
    d.line((ax, ay, ax + 90, ay + 90), fill=ac, width=6)
    d.line((ax + 90, ay + 90, ax + 30, ay + 90), fill=ac, width=6)
    d.line((ax + 90, ay + 90, ax + 90, ay + 30), fill=ac, width=6)


def draw_hook(img, th):
    d = ImageDraw.Draw(img)
    f = F("m800", 130)
    lines = ["МНЕ 19", "ЕВРОПА", "САЙТЫ С ИИ"]
    lh = 130 * 1.1
    asc = f.getmetrics()[0]
    top = H * 0.46 - len(lines) * lh / 2
    for i, ln in enumerate(lines):
        d.text((W / 2, top + asc + i * lh), ln, font=f, fill=th["fg"], anchor="ms")


def render_frame(fi):
    t = fi / FPS
    s = scene_at(t)
    th = s["th"]
    for fl in FLASHES:                                   # белая вспышка 2 кадра
        if round(fl * FPS) <= fi < round(fl * FPS) + 2:
            img = Image.new("RGB", (W, H), (255, 255, 255))
            draw_frame_chrome(img, LIGHT, fi)
            return img
    img = background(th, fi)
    kind = s.get("kind")
    if kind == "hook":
        draw_hook(img, th)
    elif kind == "center":
        layer = background(th, fi)
        headline(layer, s, t, center=True)
        z = 1.0 + 0.05 * clamp((t - s["start"]) / (s["end"] - s["start"]))   # медленный зум 1.00→1.05
        cw, ch = W / z, H / z
        img = layer.resize((W, H), Image.BICUBIC, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
    else:
        draw_card(img, s, t)
        bottom = headline(img, s, t)
        if s.get("remark"):
            text, rt = s["remark"]
            p = clamp((t - rt) / 0.12)
            if p > 0:
                ImageDraw.Draw(img, "RGBA").text((M, bottom + 70 + (1 - ease(p)) * 10), text,
                                                 font=F("i400", 34), fill=th["gray"] + (int(255 * p),), anchor="ls")
        if s is SCENES[-1]:
            draw_cta(img, s, t)
    draw_frame_chrome(img, th, fi)
    return img


def render_bytes(fi):
    return render_frame(fi).tobytes()


# ------------------------------------------------------------------ main
def main():
    global WAVE, GRAIN
    os.makedirs(OUT, exist_ok=True)
    GRAIN = grain_tiles()
    WAVE = wave_bars(load_audio())
    if "--contact" in sys.argv:
        ts = [0.0, 2.9, 4.8, 5.5, 7.0, 9.5, 11.5, 14.0]
        sheet = Image.new("RGB", (4 * 540 + 30, 2 * 960 + 10), (128, 128, 128))
        for i, t in enumerate(ts):
            im = render_frame(min(NFR - 1, int(round(t * FPS)))).resize((540, 960), Image.LANCZOS)
            ImageDraw.Draw(im).text((12, 300), f"{t:.1f}s", font=F("m700", 26), fill=(200, 30, 30))
            sheet.paste(im, ((i % 4) * 550, (i // 4) * 970))
        sheet.save(os.path.join(OUT, "contact.png"))
        return
    if "--check" in sys.argv:
        ts = [0.0, 2.5, 6.8, 13.5]
        ims = []
        for t in ts:
            im = render_frame(int(round(t * FPS)))
            im.save(os.path.join(OUT, f"check_{t:.1f}.png"))
            ims.append(im)
        sheet = Image.new("RGB", (W * 4 + 30, H), (128, 128, 128))
        for i, im in enumerate(ims):
            sheet.paste(im, (i * (W + 10), 0))
        sheet.resize((sheet.width // 2, H // 2), Image.LANCZOS).save(os.path.join(OUT, "check.png"))
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
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-ss", "0", "-t", str(LENGTH), "-i", TRACK,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-shortest", "-movflags", "+faststart", preview],
                   check=True)
    print(silent); print(preview)


if __name__ == "__main__":
    main()
