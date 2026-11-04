"""SOVAKAV v2 cards: black/white editorial, 1.5 px outlines, radius 6, 40 px padding,
thin 10 px bars, line icons, tabular digits. One red mark (#c8102e) per video at most."""
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
RED = (200, 16, 46)        # #c8102e — the single allowed red sign
PAD = 40
RADIUS = 6


def theme(name):
    if name == "light":
        return dict(name="light", bg=(239, 237, 232), fg=(10, 10, 10), soft=(107, 107, 107),
                    line=(0, 0, 0, 38), fill=None, inv_bg=(10, 10, 10), inv_fg=(239, 237, 232))
    return dict(name="dark", bg=(10, 10, 10), fg=(242, 242, 240), soft=(138, 138, 138),
                line=(255, 255, 255, 38), fill=(255, 255, 255, 8), inv_bg=(242, 242, 240), inv_fg=(10, 10, 10))


# ------------------------------------------------------------ fonts
@lru_cache(None)
def F(kind, size):
    """Montserrat: 'm300' 'm400' 'm700' 'm800'; Inter: 'i400' 'i500'."""
    fam = "Montserrat-VF.ttf" if kind[0] == "m" else "Inter-VF.ttf"
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", fam), size)
    w = int(kind[1:])
    f.set_variation_by_axes([w] if kind[0] == "m" else [min(32, max(14, size)), w])
    return f


TNUM = ["tnum"]


# ------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out_cubic(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def anim(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ev_time(cfg, ctx, key="at", word_key="at_word", off_key="offset"):
    if word_key in cfg:
        idx = min(cfg[word_key], len(ctx["word_times"]) - 1)
        return ctx["word_times"][idx] + cfg.get(off_key, 0.0)
    return cfg.get(key, 0.0)


def rgba(c, a=1.0):
    base = c[3] if len(c) == 4 else 255
    return tuple(c[:3]) + (int(clamp(a) * base),)


# ------------------------------------------------------------ text helpers
def tracked_width(text, font, em, features=None):
    if not text:
        return 0
    return sum(font.getlength(ch, features=features) for ch in text) + font.size * em * (len(text) - 1)


def tracked(d, x, y, text, font, fill, em=0.18, align="left", features=None):
    w = tracked_width(text, font, em, features)
    x -= w if align == "right" else (w / 2 if align == "center" else 0)
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls", features=features)
        x += font.getlength(ch, features=features) + font.size * em
    return w


def label(d, x, y, text, th, a=1.0, align="left", color=None):
    """Service caption: Inter Medium 22, caps, tracking +18 %."""
    tracked(d, x, y, text.upper(), F("i500", 22), rgba(color or th["soft"], a), 0.18, align)


def num(d, x, y, text, size, fill, anchor="ls"):
    d.text((x, y), text, font=F("m700", size), fill=fill, anchor=anchor, features=TNUM)
    return F("m700", size).getlength(text, features=TNUM)


def fmt_int(v):
    return f"{int(round(v)):,}".replace(",", " ")


def bar(d, x, y, w, frac, th, a=1.0, h=10):
    d.rectangle((x, y, x + w, y + h), fill=rgba(th["line"], a))
    fw = clamp(frac) * w
    if fw >= 1:
        d.rectangle((x, y, x + fw, y + h), fill=rgba(th["fg"], a))


# ------------------------------------------------------------ card frame
def frame_outline(img, th, alpha=1.0, extra=0.0):
    """1.5 px outline (drawn at 2× then downsampled) + optional pulse glow."""
    w, h = img.size
    big = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    col = th["line"] if extra <= 0 else tuple(int(lerp(a, b, extra)) for a, b in zip(th["line"], th["fg"] + (255,)))
    bd.rounded_rectangle((1, 1, w * 2 - 2, h * 2 - 2), RADIUS * 2, outline=rgba(col, alpha), width=3)
    img.alpha_composite(big.resize((w, h), Image.LANCZOS))


def base(w, h, th, left, right=None):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if th["fill"]:
        d.rounded_rectangle((0, 0, w - 1, h - 1), RADIUS, fill=th["fill"])
    frame_outline(img, th)
    d = ImageDraw.Draw(img)
    label(d, PAD, PAD + 18, left, th)
    if right:
        label(d, w - PAD, PAD + 18, right, th, align="right")
    return img, d


# ------------------------------------------------------------ cards
def card_meter(c, t, ctx, w, th):
    h = 420
    img, d = base(w, h, th, c["title"], c.get("tag"))
    p = ease_out_cubic(anim(t, ev_time(c, ctx), c.get("dur", 0.6)))
    v = lerp(c["from"], c["to"], p)
    txt = f"{int(round(v)):>3d}".replace(" ", " ")   # figure spaces keep width fixed
    x = PAD - 6
    x += num(d, x, 250, txt, 160, th["fg"])
    d.text((x + 8, 250), c.get("suffix", ""), font=F("m700", 70), fill=th["fg"], anchor="ls")
    delta = c["to"] - c["from"]
    if p > 0.02:
        s = fmt_int(abs(delta) * p) + c.get("suffix", "")
        f = F("m700", 48)
        sw = f.getlength(s, features=TNUM)
        d.text((w - PAD, 240), s, font=f, fill=th["fg"], anchor="rs", features=TNUM)
        sign_col = RED if c.get("red_sign") else th["fg"]
        d.text((w - PAD - sw - 4, 240), "−" if delta < 0 else "+", font=f, fill=sign_col, anchor="rs")
    bar(d, PAD, 290, w - 2 * PAD, v / 100.0, th)
    label(d, PAD, 370, c.get("caption", ""), th)
    return img


def card_bars(c, t, ctx, w, th):
    rows = c["rows"]
    h = 130 + 140 * len(rows)
    img, d = base(w, h, th, c["title"], c.get("tag"))
    p = ease_out_cubic(anim(t, ev_time(c, ctx), c.get("dur", 0.8)))
    for i, r in enumerate(rows):
        y = 150 + i * 140
        d.text((PAD, y + 20), r["label"], font=F("i500", 40), fill=th["fg"], anchor="ls")
        val = f"{int(round(lerp(0, r['value'], p))):>2d}".replace(" ", " ") + r.get("unit", "")
        d.text((w - PAD, y + 24), val, font=F("m700", 60), fill=th["fg"] if r.get("strong") else th["soft"],
               anchor="rs", features=TNUM)
        bar(d, PAD, y + 54, w - 2 * PAD, r["frac"] * p, th)
    return img


def person(d, x, y, col):
    d.ellipse((x - 22, y, x + 22, y + 44), outline=col, width=2)
    d.rounded_rectangle((x - 38, y + 58, x + 38, y + 116), 6, outline=col, width=2)


def card_people(c, t, ctx, w, th):
    h = 440
    img, d = base(w, h, th, c["title"], c.get("tag"))
    total, remain = c["total"], c["remain"]
    t0, step = ev_time(c, ctx), c.get("step", 0.3)
    gone = 0
    for i in range(total):
        k = total - 1 - i
        fade = 0.0
        if k < total - remain:
            ts = t0 + k * step
            fade = ease_out_cubic(anim(t, ts, 0.25))
            gone += t >= ts + 0.12
        person(d, 90 + (i % 3) * 120, 120 + (i // 3) * 150, rgba(th["fg"], lerp(1.0, 0.15, fade)))
    n = total - gone
    label(d, 470, 140, c.get("label", ""), th)
    num(d, 462, 320, str(n), 160, th["fg"])
    if n <= remain:
        a = ease_out_cubic(anim(t, t0 + (total - remain - 1) * step + 0.12, 0.3))
        f = F("i500", 22)
        txt = c.get("badge", "").upper()
        bw = tracked_width(txt, f, 0.18) + 40
        by = 352 + (1 - a) * 10
        d.rounded_rectangle((470, by, 470 + bw, by + 44), RADIUS, fill=rgba(th["fg"], a))
        tracked(d, 490, by + 30, txt, f, rgba(th["bg"], a), 0.18)
    return img


def wallet(d, x, y, col, s=1.0):
    """Line wallet icon, 2 px stroke."""
    d.rounded_rectangle((x, y + 16 * s, x + 120 * s, y + 96 * s), 8, outline=col, width=2)
    d.line((x + 10 * s, y + 16 * s, x + 86 * s, y), fill=col, width=2)
    d.line((x + 86 * s, y, x + 96 * s, y + 16 * s), fill=col, width=2)
    d.rounded_rectangle((x + 78 * s, y + 42 * s, x + 120 * s, y + 70 * s), 6, outline=col, width=2)
    d.ellipse((x + 90 * s, y + 52 * s, x + 98 * s, y + 60 * s), outline=col, width=2)


def card_columns(c, t, ctx, w, th):
    h = 520
    img, d = base(w, h, th, c["title"], c.get("tag"))
    cols = c["columns"]
    area_top, area_bot = 110, 400
    cw, gap = 110, 60
    for i, col in enumerate(cols):
        p = ease_out_cubic(anim(t, col["at"], 0.45))
        x = PAD + i * (cw + gap)
        d.rectangle((x, area_top, x + cw, area_bot), outline=rgba(th["line"]), width=2)
        hh = (area_bot - area_top) * col["frac"] * p
        if hh > 1:
            d.rectangle((x, area_bot - hh, x + cw, area_bot), fill=rgba(th["fg"]))
        label(d, x, area_bot + 44, col["label"], th, a=clamp(p * 3), color=th["fg"])
    wl = c.get("wallet")
    if wl:
        a = ease_out_cubic(anim(t, wl["at"], 0.3))
        if a > 0:
            wx, wy = w - PAD - 230, 170 + (1 - a) * 40
            wallet(d, wx + 50, wy, rgba(th["fg"], a), 1.1)
            for j, line in enumerate(wl["text"]):
                d.text((wx + 115, wy + 175 + j * 44), line, font=F("i500", 34), fill=rgba(th["fg"], a), anchor="ms")
    return img


@lru_cache(None)
def load_bw(path, width):
    im = Image.open(path).convert("L")
    im = ImageEnhance.Contrast(im).enhance(1.10)
    a = np.asarray(im, np.float32)
    a += np.random.default_rng(5).normal(0, 255 * 0.03 * 0.5, a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
    h = round(im.height * width / im.width)
    return im.resize((width, h), Image.LANCZOS)


def card_screenshot(c, t, ctx, w, th):
    """B/W screenshot, 1.5 px frame, no rounding; zoom + slow scroll; outline pulses on given times."""
    p = clamp(t / ctx["dur"])
    z = lerp(*c.get("zoom", [1.0, 1.08]), p)
    src = load_bw(os.path.join(ROOT, c["image"]), int(c.get("width", w) * 1.08) + 2)
    tw = int(c.get("width", w) * z)
    im = src.resize((tw, round(src.height * tw / src.width)), Image.BICUBIC)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.paste(im, (0, 0))
    pulse = 0.0
    for pt in c.get("pulse", []):
        if t >= pt:
            pulse = max(pulse, math.exp(-(t - pt) / 0.18) * clamp((t - pt) / 0.06))
    d = ImageDraw.Draw(out)
    col = tuple(int(lerp(a, b, pulse)) for a, b in zip(th["line"][:3], th["fg"]))
    d.rectangle((0, 0, out.width - 1, out.height - 1), outline=col + (int(lerp(38, 255, pulse)),), width=2)
    if pulse > 0.02:
        d.rectangle((-0, -0, out.width - 1, out.height - 1), outline=th["fg"] + (int(90 * pulse),), width=5)
    return out


def card_notes(c, t, ctx, w, th):
    items = c["items"]
    h = 200 + 84 * len(items)
    img, d = base(w, h, th, c.get("label", "Заметки"), c.get("tag"))
    d.text((PAD, 160), c["title"].upper(), font=F("m800", 50), fill=th["fg"], anchor="ls")
    for i, it in enumerate(items):
        ti = c["at"][i] if isinstance(c.get("at"), list) else ev_time(c, ctx) + i * c.get("step", 0.6)
        a = ease_out_cubic(anim(t, ti, 0.25))
        if a <= 0:
            continue
        y = 196 + i * 84
        d.line((PAD, y, w - PAD, y), fill=rgba(th["line"], a), width=2)
        bx, by = PAD, y + 24
        d.rectangle((bx, by, bx + 36, by + 36), outline=rgba(th["fg"], a), width=2)
        ck = ease_out_cubic(anim(t, ti + 0.08, 0.2))
        if ck > 0:
            pts = [(bx + 8, by + 19), (bx + 15, by + 27), (bx + 29, by + 10)]
            d.line(pts if ck > 0.5 else pts[:2], fill=rgba(th["fg"], a), width=3)
        d.text((100 - (1 - a) * 10, y + 56), it, font=F("i500", 40), fill=rgba(th["fg"], a), anchor="ls")
    return img


def card_chat(c, t, ctx, w, th):
    h = 440
    img, d = base(w, h, th, c["name"], c.get("status"))
    a = ease_out_cubic(anim(t, ev_time(c, ctx), 0.3))
    if a > 0:
        f = F("i500", 40)
        y = 120 + (1 - a) * 10
        d.rounded_rectangle((PAD, y, PAD + f.getlength(c["message"]) + 64, y + 96), RADIUS,
                            outline=rgba(th["fg"], a * 0.6), width=2)
        d.text((PAD + 32, y + 62), c["message"], font=f, fill=rgba(th["fg"], a), anchor="ls")
    return img


def card_finance(c, t, ctx, w, th):
    h = 440
    img, d = base(w, h, th, c["title"], c.get("tag"))
    for i, r in enumerate(c["rows"]):
        y = 150 + i * 130
        p = ease_out_cubic(anim(t, ev_time(r, ctx), r.get("dur", 0.8)))
        v = lerp(r["from"], r["to"], p)
        d.text((PAD, y + 20), r["label"], font=F("i500", 40), fill=th["fg"], anchor="ls")
        d.text((w - PAD, y + 24), f"{int(round(v)):>3d}".replace(" ", "\u2007") + "%", font=F("m700", 56),
               fill=th["fg"] if r.get("strong") else th["soft"], anchor="rs", features=TNUM)
        bar(d, PAD, y + 50, w - 2 * PAD, v / 100.0, th)
    b = c.get("blink")
    if b:
        a = ease_out_cubic(anim(t, ev_time(b, ctx), 0.3))
        if a > 0:
            d.line((PAD, 340, w - PAD, 340), fill=rgba(th["line"]), width=2)
            d.text((PAD, 400 + (1 - a) * 10), b["text"], font=F("m700", 50), fill=rgba(th["fg"], a), anchor="ls")
    return img


RENDERERS = {"finance": card_finance, "meter": card_meter, "bars": card_bars, "people": card_people, "columns": card_columns,
             "screenshot": card_screenshot, "notes": card_notes, "chat": card_chat}

HEIGHTS = {"finance": lambda c: 440, "meter": lambda c: 420, "people": lambda c: 440, "columns": lambda c: 520, "chat": lambda c: 440,
           "bars": lambda c: 130 + 140 * len(c["rows"]), "notes": lambda c: 200 + 84 * len(c["items"]),
           "screenshot": lambda c: 300}


def render_card(cfg, t, ctx, width, th):
    return RENDERERS[cfg["type"]](cfg, t, ctx, width, th)
