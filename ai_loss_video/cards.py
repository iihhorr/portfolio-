"""SOVAKAV-style cards: monochrome, thin lines, caps labels with wide tracking.
No glass, no colour, no emoji, no springs. Each card is an RGBA image; render.py places it."""
import math
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------ palette (DESIGN.md §3)
INK = (10, 10, 10)          # #0A0A0A
PAPER = (244, 242, 238)     # #F4F2EE
GRAPHITE = (107, 104, 98)   # #6B6862  — light bg, large text only
MUTED = (154, 148, 138)     # #9A948A  — captions on dark
DIM = (28, 27, 25)          # #1C1B19
SOFT = (138, 134, 126)      # #8A867E  — "lighter tone" line of a headline


def theme(name):
    if name == "light":
        return dict(name="light", bg=PAPER, fg=INK, small=INK, soft=GRAPHITE,
                    line=INK + (89,), track=INK + (28,), ghost=INK + (40,))
    return dict(name="dark", bg=INK, fg=PAPER, small=MUTED, soft=SOFT,
                line=PAPER + (89,), track=DIM + (255,), ghost=PAPER + (40,))


# ------------------------------------------------------------ fonts (DESIGN.md §4)
@lru_cache(None)
def F(kind, size):
    """kind: 'reg' 400, 'med' 500, 'bold' 700 (Montserrat) or 'mark' (Archivo 600, width 125)."""
    if kind == "mark":
        f = ImageFont.truetype(os.path.join(ROOT, "fonts", "Archivo-VF.ttf"), size)
        f.set_variation_by_axes([600, 125])
        return f
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", "Montserrat-VF.ttf"), size)
    f.set_variation_by_axes([{"reg": 400, "med": 500, "bold": 700}[kind]])
    return f


# ------------------------------------------------------------ easing (power3/expo out)
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def power3_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def expo_out(t):
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def anim(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ev_time(cfg, ctx, key="at", word_key="at_word", off_key="offset"):
    if word_key in cfg:
        idx = min(cfg[word_key], len(ctx["word_times"]) - 1)
        return ctx["word_times"][idx] + cfg.get(off_key, 0.0)
    return cfg.get(key, 0.0)


def rgba(c, a):
    return tuple(c[:3]) + (int(clamp(a) * (c[3] if len(c) == 4 else 255)),)


# ------------------------------------------------------------ text helpers
def tracked_width(text, font, em):
    if not text:
        return 0
    return sum(font.getlength(ch) for ch in text) + font.size * em * (len(text) - 1)


def tracked(d, x, y, text, font, fill, em=0.35, align="left"):
    """Caps label with letter spacing; (x, y) = baseline. align left|right|center."""
    w = tracked_width(text, font, em)
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + font.size * em
    return w


def label(d, x, y, text, th, a=1.0, align="left", size=20):
    tracked(d, x, y, text.upper(), F("med", size), rgba(th["small"], a), 0.35, align)


def diamond(d, cx, cy, s, fill):
    r = s / math.sqrt(2)
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill)


def fmt_int(v):
    return f"{int(round(v)):,}".replace(",", " ")


def bar(d, x, y, w, h, frac, th, a=1.0):
    d.rectangle((x, y, x + w, y + h), fill=rgba(th["track"], a))
    fw = clamp(frac) * w
    if fw >= 1:
        d.rectangle((x, y, x + fw, y + h), fill=rgba(th["fg"], a))


# ------------------------------------------------------------ frame of a card
def base(w, h, th, title, tag=None):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w - 1, h - 1), outline=th["line"], width=2)
    label(d, 40, 58, title, th)
    if tag:
        label(d, w - 40, 58, tag, th, align="right")
    d.line((40, 84, w - 40, 84), fill=th["line"], width=1)
    return img, d


# ------------------------------------------------------------ cards
def card_meter(c, t, ctx, w, th):
    h = 430
    img, d = base(w, h, th, c["title"], c.get("tag"))
    p = power3_out(anim(t, ev_time(c, ctx), c.get("dur", 0.8)))
    v = lerp(c["from"], c["to"], p)
    txt = fmt_int(v)
    big = F("bold", 160)
    d.text((34, 262), txt, font=big, fill=th["fg"], anchor="ls")
    d.text((34 + big.getlength(txt) + 10, 262), c.get("suffix", ""), font=F("bold", 72), fill=th["fg"], anchor="ls")
    delta = c["to"] - c["from"]
    if p > 0.02:
        s = ("−" if delta < 0 else "+") + fmt_int(abs(delta) * p) + c.get("suffix", "")
        d.text((w - 40, 250), s, font=F("med", 44), fill=rgba(th["soft"], clamp(p * 2)), anchor="rs")
    bar(d, 40, 302, w - 80, 10, v / 100.0, th)
    label(d, 40, 380, c.get("caption", ""), th)
    return img


def card_bars(c, t, ctx, w, th):
    rows = c["rows"]
    h = 120 + 150 * len(rows)
    img, d = base(w, h, th, c["title"], c.get("tag"))
    p = power3_out(anim(t, ev_time(c, ctx), c.get("dur", 1.0)))
    for i, r in enumerate(rows):
        y = 150 + i * 150
        label(d, 40, y + 14, r["label"], th, size=22)
        val = fmt_int(lerp(0, r["value"], p)) + r.get("unit", "")
        col = th["fg"] if r.get("strong") else th["soft"]
        d.text((w - 40, y + 22), val, font=F("bold", 64), fill=col, anchor="rs")
        bar(d, 40, y + 50, w - 80, 10, r["frac"] * p, th)
    return img


def person(d, x, y, col):
    d.ellipse((x - 22, y, x + 22, y + 44), outline=col, width=3)
    d.rectangle((x - 38, y + 58, x + 38, y + 116), outline=col, width=3)


def card_people(c, t, ctx, w, th):
    h = 450
    img, d = base(w, h, th, c["title"], c.get("tag"))
    total, remain = c["total"], c["remain"]
    t0, step = ev_time(c, ctx), c.get("step", 0.45)
    gone = 0
    for i in range(total):
        k = total - 1 - i
        fade = 0.0
        if k < total - remain:
            ts = t0 + k * step
            fade = power3_out(anim(t, ts, 0.5))
            gone += t >= ts + 0.2
        cx, cy = 90 + (i % 3) * 120, 128 + (i // 3) * 150
        person(d, cx, cy, rgba(th["fg"], lerp(1.0, 0.18, fade)))
    n = total - gone
    label(d, 470, 150, c.get("label", ""), th)
    d.text((462, 330), str(n), font=F("bold", 160), fill=th["fg"], anchor="ls")
    if n <= remain:
        tb = t0 + (total - remain - 1) * step + 0.2
        a = power3_out(anim(t, tb, 0.6))
        f = F("med", 20)
        txt = c.get("badge", "").upper()
        bw = tracked_width(txt, f, 0.35) + 44
        bx, by = 470, 370 + (1 - a) * 16
        d.rectangle((bx, by, bx + bw, by + 44), fill=rgba(th["fg"], a))  # the one accent
        tracked(d, bx + 22, by + 29, txt, f, rgba(th["bg"], a), 0.35)
    return img


def card_finance(c, t, ctx, w, th):
    h = 440
    img, d = base(w, h, th, c["title"], c.get("tag"))
    for i, r in enumerate(c["rows"]):
        y = 150 + i * 130
        p = power3_out(anim(t, ev_time(r, ctx), r.get("dur", 0.8)))
        v = lerp(r["from"], r["to"], p)
        label(d, 40, y + 14, r["label"], th, size=22)
        col = th["fg"] if r.get("strong") else th["soft"]
        d.text((w - 40, y + 22), fmt_int(v) + "%", font=F("bold", 60), fill=col, anchor="rs")
        bar(d, 40, y + 46, w - 80, 10, v / 100.0, th)
    b = c.get("blink")
    if b:
        a = power3_out(anim(t, ev_time(b, ctx), 0.8))
        if a > 0:
            d.line((40, 340, w - 40, 340), fill=th["line"], width=1)
            d.text((40, 400 + (1 - a) * 20), b["text"], font=F("bold", 52), fill=rgba(th["fg"], a), anchor="ls")
    return img


def card_chat(c, t, ctx, w, th):
    h = 470
    img, d = base(w, h, th, c["name"], c.get("status"))
    tm = ev_time(c, ctx)
    a = power3_out(anim(t, tm, 0.7))
    if a > 0:
        f = F("med", 40)
        y = 140 + (1 - a) * 24
        tw = f.getlength(c["message"])
        d.rectangle((40, y, 40 + tw + 64, y + 96), outline=rgba(th["fg"], a * 0.6), width=2)
        d.text((72, y + 62), c["message"], font=f, fill=rgba(th["fg"], a), anchor="ls")
    tt = ev_time({"at_word": c["typing_at_word"], "offset": c.get("typing_offset", 0)}, ctx) \
        if "typing_at_word" in c else c.get("typing_at", 2.0)
    at = power3_out(anim(t, tt, 0.5))
    if at > 0:
        y = 270
        for k in range(3):
            ph = 0.5 + 0.5 * math.sin((t - tt) * 2 * math.pi * 1.2 - k * 0.9)
            cx = 52 + k * 24
            d.ellipse((cx - 6, y - 6, cx + 6, y + 6), fill=rgba(th["fg"], at * (0.25 + 0.75 * ph)))
        label(d, 130, y + 7, "печатает", th, a=at)
    pills = c.get("pills", [])
    if pills:
        gap = 16
        pw = (w - 80 - gap * (len(pills) - 1)) / len(pills)
        d.line((40, 320, w - 40, 320), fill=th["line"], width=1)
        for i, p in enumerate(pills):
            ft = power3_out(anim(t, p["at"], 0.7))
            x = 40 + i * (pw + gap)
            label(d, x, 360, p.get("label", ""), th)
            f = F("bold", 44)
            if ft < 1:
                d.text((x, 425 - ft * 20), p["from"], font=f, fill=rgba(th["soft"], 1 - ft), anchor="ls")
            if ft > 0:
                d.text((x, 445 - ft * 20), p["to"], font=f, fill=rgba(th["fg"], ft), anchor="ls")
    return img


def card_notes(c, t, ctx, w, th):
    items = c["items"]
    h = 200 + 84 * len(items)
    img, d = base(w, h, th, c.get("label", "Заметки"), c.get("tag"))
    d.text((40, 160), c["title"].upper(), font=F("bold", 52), fill=th["fg"], anchor="ls")
    t0, step = ev_time(c, ctx), c.get("step", 0.6)
    for i, it in enumerate(items):
        a = power3_out(anim(t, t0 + i * step, 0.6))
        if a <= 0:
            continue
        y = 200 + i * 84
        d.line((40, y, w - 40, y), fill=rgba(th["line"], a), width=1)
        bx, by = 40, y + 24
        d.rectangle((bx, by, bx + 36, by + 36), outline=rgba(th["fg"], a), width=2)
        ck = power3_out(anim(t, t0 + i * step + 0.25, 0.4))
        if ck > 0:
            pts = [(bx + 8, by + 19), (bx + 15, by + 27), (bx + 29, by + 10)]
            d.line(pts if ck > 0.5 else pts[:2], fill=rgba(th["fg"], a), width=4)
        d.text((100 - (1 - a) * 20, y + 56), it, font=F("med", 40), fill=rgba(th["fg"], a), anchor="ls")
    return img


RENDERERS = {"meter": card_meter, "bars": card_bars, "people": card_people,
             "finance": card_finance, "chat": card_chat, "notes": card_notes}

HEIGHTS = {"meter": lambda c: 430, "people": lambda c: 450, "finance": lambda c: 440,
           "chat": lambda c: 470, "bars": lambda c: 120 + 150 * len(c["rows"]),
           "notes": lambda c: 200 + 84 * len(c["items"])}


def render_card(cfg, t, ctx, width, th):
    return RENDERERS[cfg["type"]](cfg, t, ctx, width, th)
