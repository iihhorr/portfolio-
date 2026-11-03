"""Card drawing for the TikTok visualizer. Every card is rendered as an RGBA image
of fixed width; render.py handles placement, tilt, breathing and shadow."""
import math
import os
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT_FILES = {
    "reg": "Inter-Regular.ttf",
    "semi": "Inter-SemiBold.ttf",
    "bold": "Inter-Bold.ttf",
    "xbold": "Inter-ExtraBold.ttf",
}
GLASS = (21, 22, 28)          # #15161c
GLASS_ALPHA = 235             # 92 %
OUTLINE = (255, 255, 255, 26)  # #ffffff1a
RADIUS = 36
GRAY = (139, 141, 152)
WHITE = (255, 255, 255)
GREEN, YELLOW, RED = (51, 209, 122), (245, 197, 66), (255, 77, 79)


# ---------------------------------------------------------------- helpers
@lru_cache(None)
def F(kind, size):
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(ROOT, "fonts", FONT_FILES[kind]), size)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(c1, c2, t):
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


def ease_in_out(t):
    t = clamp(t)
    return 3 * t * t - 2 * t * t * t if t < 1 else 1.0


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def value_color(good):
    """good: 1 -> green, 0.5 -> yellow, 0 -> red"""
    good = clamp(good)
    if good >= 0.5:
        return lerp_col(YELLOW, GREEN, (good - 0.5) * 2)
    return lerp_col(RED, YELLOW, good * 2)


def ev_time(cfg, ctx, key="at", word_key="at_word", off_key="offset"):
    """Resolve an event time (seconds from scene start)."""
    if word_key in cfg:
        idx = min(cfg[word_key], len(ctx["word_times"]) - 1)
        return ctx["word_times"][idx] + cfg.get(off_key, 0.0)
    return cfg.get(key, 0.0)


def anim(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


@lru_cache(None)
def round_mask(w, h, r):
    big = Image.new("L", (w * 3, h * 3), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * 3 - 1, h * 3 - 1), r * 3, fill=255)
    return big.resize((w, h), Image.LANCZOS)


@lru_cache(None)
def shadow(w, h, pad):
    im = Image.new("L", (w + pad * 2, h + pad * 2), 0)
    ImageDraw.Draw(im).rounded_rectangle((pad, pad + 18, pad + w, pad + h + 18), RADIUS, fill=150)
    return im.filter(ImageFilter.GaussianBlur(pad / 2.2))


def is_emoji(ch):
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or o in (0xFE0F,)


@lru_cache(None)
def emoji_img(ch, size):
    from PIL import ImageFont
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", "NotoColorEmoji.ttf"), 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((int(im.width * size / im.height), size), Image.LANCZOS)


def rich_runs(text):
    runs, cur = [], ""
    for ch in text:
        if ch == "️":
            continue
        if is_emoji(ch):
            if cur:
                runs.append(("t", cur))
                cur = ""
            runs.append(("e", ch))
        else:
            cur += ch
    if cur:
        runs.append(("t", cur))
    return runs


def rich_width(text, font):
    return sum(font.getlength(s) if k == "t" else font.size * 1.1 for k, s in rich_runs(text))


def draw_rich(img, d, x, y, text, font, fill):
    """Text with color emoji. (x, y) = left baseline."""
    for k, s in rich_runs(text):
        if k == "t":
            d.text((x, y), s, font=font, fill=fill, anchor="ls")
            x += font.getlength(s)
        else:
            e = emoji_img(s, int(font.size * 0.95))
            if len(fill) == 4 and fill[3] < 255:
                e = e.copy()
                e.putalpha(e.getchannel("A").point(lambda a: a * fill[3] // 255))
            img.paste(e, (int(x + font.size * 0.07), int(y - font.size * 0.82)), e)
            x += font.size * 1.1
    return x


def bar(d, x, y, w, h, frac, color, track=(255, 255, 255, 28)):
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=track)
    fw = max(0.0, min(1.0, frac)) * w
    if fw > 0.5:
        fw = max(fw, h)
        d.rounded_rectangle((x, y, x + fw, y + h), h // 2, fill=color)


def header(d, w, title, tag=None):
    d.text((48, 52), title, font=F("semi", 30), fill=GRAY, anchor="ls")
    if tag:
        d.text((w - 48, 52), tag, font=F("semi", 26), fill=GRAY, anchor="rs")


def base(w, h):
    img = Image.new("RGB", (w, h), GLASS)
    d = ImageDraw.Draw(img, "RGBA")
    # soft top highlight to sell the "glass"
    for i in range(120):
        d.line((0, i, w, i), fill=(255, 255, 255, int(10 * (1 - i / 120))))
    return img, d


def finish(img):
    w, h = img.size
    out = img.convert("RGBA")
    m = round_mask(w, h, RADIUS).point(lambda a: a * GLASS_ALPHA // 255)
    out.putalpha(m)
    ImageDraw.Draw(out).rounded_rectangle((0, 0, w - 1, h - 1), RADIUS, outline=(255, 255, 255, 40), width=2)
    return out


def fmt_int(v):
    return f"{int(round(v)):,}".replace(",", " ")


# ---------------------------------------------------------------- cards
def card_meter(c, t, ctx, w):
    h = 440
    img, d = base(w, h)
    header(d, w, c["title"], c.get("tag"))
    t0 = ev_time(c, ctx)
    p = ease_in_out(anim(t, t0, c.get("dur", 0.6)))
    v = lerp(c["from"], c["to"], p)
    col = value_color(v / 100.0)
    txt = fmt_int(v)
    d.text((40, 250), txt, font=F("xbold", 160), fill=WHITE, anchor="ls")
    x2 = 40 + F("xbold", 160).getlength(txt) + 6
    d.text((x2, 250), c.get("suffix", ""), font=F("bold", 90), fill=WHITE, anchor="ls")
    delta = c["to"] - c["from"]
    if p > 0.02:
        a = int(255 * clamp(p * 2))
        dc = RED if delta < 0 else GREEN
        d.text((w - 48, 230), ("−" if delta < 0 else "+") + fmt_int(abs(delta) * p) + c.get("suffix", ""),
               font=F("bold", 54), fill=dc + (a,), anchor="rs")
    bar(d, 48, 300, w - 96, 26, v / 100.0, col)
    d.text((48, 385), c.get("caption", ""), font=F("semi", 28), fill=GRAY, anchor="ls")
    return finish(img)


def card_bars(c, t, ctx, w):
    rows = c["rows"]
    h = 130 + 150 * len(rows)
    img, d = base(w, h)
    header(d, w, c["title"], c.get("tag"))
    p = ease_in_out(anim(t, ev_time(c, ctx), c.get("dur", 0.6)))
    for i, r in enumerate(rows):
        y = 140 + i * 150
        col = hex2rgb(r["color"])
        d.text((48, y + 20), r["label"], font=F("bold", 44), fill=WHITE, anchor="ls")
        val = fmt_int(lerp(0, r["value"], p)) + r.get("unit", "")
        d.text((w - 48, y + 22), val, font=F("xbold", 64), fill=col, anchor="rs")
        bar(d, 48, y + 52, w - 96, 26, r["frac"] * p, col)
    return finish(img)


def person(d, x, y, col, s=1.0):
    r = 24 * s
    d.ellipse((x - r, y, x + r, y + 2 * r), outline=col, width=5)
    d.rounded_rectangle((x - 40 * s, y + 2 * r + 10 * s, x + 40 * s, y + 2 * r + 72 * s), 28 * s, outline=col, width=5)


def card_people(c, t, ctx, w):
    h = 460
    img, d = base(w, h)
    header(d, w, c["title"], c.get("tag"))
    total, remain = c["total"], c["remain"]
    t0, step = ev_time(c, ctx), c.get("step", 0.45)
    gone = 0
    for i in range(total):
        idx_from_end = total - 1 - i
        fade = 0.0
        if idx_from_end < total - remain:
            ts = t0 + idx_from_end * step
            fade = ease_in_out(anim(t, ts, 0.3))
            if t >= ts + 0.15:
                gone += 1
        cx = 100 + (i % 3) * 130
        cy = 110 + (i // 3) * 165
        a = int(lerp(255, 40, fade))
        colr = lerp_col(WHITE, RED, fade) + (a,)
        person(d, cx, cy, colr)
    n = total - gone
    d.text((520, 135), c.get("label", ""), font=F("semi", 32), fill=GRAY, anchor="ls")
    good = (n - remain) / max(1, total - remain)
    d.text((515, 300), str(n), font=F("xbold", 160), fill=value_color(good), anchor="ls")
    if n <= remain:
        tb = t0 + (total - remain - 1) * step + 0.15
        pop = ease_out_back(anim(t, tb, 0.3))
        f = F("bold", 30)
        bw = f.getlength(c.get("badge", "")) + 48
        bx, by = 520, 340
        sc = max(0.01, pop)
        bh = 58 * sc
        d.rounded_rectangle((bx, by + (58 - bh) / 2, bx + bw * sc, by + (58 + bh) / 2), int(29 * sc), fill=(229, 72, 77))
        if pop > 0.6:
            d.text((bx + 24, by + 40), c.get("badge", ""), font=f, fill=WHITE, anchor="ls")
    return finish(img)


def card_finance(c, t, ctx, w):
    h = 470
    img, d = base(w, h)
    header(d, w, c["title"], c.get("tag"))
    for i, r in enumerate(c["rows"]):
        y = 140 + i * 140
        p = ease_in_out(anim(t, ev_time(r, ctx), r.get("dur", 0.6)))
        v = lerp(r["from"], r["to"], p)
        col = hex2rgb(r["color"])
        d.text((48, y + 20), r["label"], font=F("bold", 42), fill=WHITE, anchor="ls")
        d.text((w - 48, y + 22), fmt_int(v) + "%", font=F("xbold", 60), fill=col, anchor="rs")
        bar(d, 48, y + 50, w - 96, 26, v / 100.0, col)
    b = c.get("blink")
    if b:
        tb = ev_time(b, ctx)
        if t >= tb:
            a = 0.25 + 0.75 * (0.5 + 0.5 * math.cos((t - tb) * 2 * math.pi * 1.5))
            a *= clamp((t - tb) / 0.15)
            d.text((48, 425), b["text"], font=F("xbold", 58), fill=RED + (int(255 * a),), anchor="ls")
    return finish(img)


def pill(text, bg, w, h):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), h // 2, fill=bg)
    f = F("bold", 30)
    while f.getlength(text) > w - 24:
        f = F("bold", f.size - 2)
    d.text((w / 2, h / 2), text, font=f, fill=WHITE, anchor="mm")
    return im


def card_chat(c, t, ctx, w):
    h = 490
    img, d = base(w, h)
    acc = ctx["accent"]
    d.ellipse((48, 34, 120, 106), fill=acc)
    d.text((84, 70), "ИИ", font=F("bold", 28), fill=WHITE, anchor="mm")
    d.text((140, 66), c["name"], font=F("bold", 36), fill=WHITE, anchor="ls")
    d.ellipse((140, 84, 154, 98), fill=GREEN)
    d.text((164, 99), c.get("status", ""), font=F("semi", 26), fill=GREEN, anchor="ls")
    # message bubble
    tm = ev_time(c, ctx)
    pm = ease_out_back(anim(t, tm, 0.35))
    am = clamp(anim(t, tm, 0.2))
    if am > 0:
        f = F("semi", 42)
        tw = rich_width(c["message"], f)
        bx, by, bw, bh = 48, 140 + (1 - pm) * 30, tw + 64, 92
        d.rounded_rectangle((bx, by, bx + bw, by + bh), 30, fill=(48, 50, 64, int(255 * am)))
        draw_rich(img, d, bx + 32, by + 60, c["message"], f, (255, 255, 255, int(255 * am)))
    # typing indicator
    tt = ev_time({"at_word": c.get("typing_at_word", 0), "offset": c.get("typing_offset", 0)}, ctx) \
        if "typing_at_word" in c else c.get("typing_at", 2.0)
    at = anim(t, tt, 0.2)
    if at > 0:
        by = 254
        d.rounded_rectangle((48, by, 168, by + 60), 30, fill=(48, 50, 64, int(255 * at)))
        for k in range(3):
            ph = (t - tt) * 2 * math.pi * 1.6 - k * 0.9
            dy = -6 * max(0.0, math.sin(ph))
            a = int(255 * at * (0.45 + 0.55 * max(0.0, math.sin(ph))))
            cx = 80 + k * 28
            d.ellipse((cx - 8, by + 30 - 8 + dy, cx + 8, by + 30 + 8 + dy), fill=(220, 220, 230, a))
        d.text((186, by + 40), "печатает…", font=F("semi", 28), fill=GRAY + (int(255 * at),), anchor="ls")
    # flipping mini pills
    pills = c.get("pills", [])
    if pills:
        gap = 18
        pw = int((w - 96 - gap * (len(pills) - 1)) / len(pills))
        ph_ = 76
        for i, p in enumerate(pills):
            ft = anim(t, p["at"], 0.45)
            if ft < 0.5:
                im = pill(p["from"], (150, 42, 48), pw, ph_)
                sy = math.cos(ft * math.pi)
            else:
                im = pill(p["to"], (34, 160, 92), pw, ph_)
                sy = -math.cos(ft * math.pi)
            sh = max(2, int(ph_ * sy))
            im = im.resize((pw, sh), Image.BICUBIC)
            x = 48 + i * (pw + gap)
            y = 370 + (ph_ - sh) // 2
            img.paste(im, (x, y), im)
    return finish(img)


def check_icon(d, cx, cy, r, s):
    if s <= 0:
        return
    rr = r * s
    d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=GREEN)
    if s > 0.6:
        d.line([(cx - r * .45, cy + r * .02), (cx - r * .1, cy + r * .38), (cx + r * .5, cy - r * .35)],
               fill=WHITE, width=6, joint="curve")


def card_notes(c, t, ctx, w):
    items = c["items"]
    h = 210 + 82 * len(items)
    img, d = base(w, h)
    d.rounded_rectangle((48, 32, 76, 60), 7, fill=(255, 204, 0))
    d.text((90, 57), c.get("label", "Заметки"), font=F("bold", 30), fill=(255, 204, 0), anchor="ls")
    d.text((48, 140), c["title"], font=F("xbold", 58), fill=WHITE, anchor="ls")
    t0, step = ev_time(c, ctx), c.get("step", 0.6)
    for i, it in enumerate(items):
        ti = t0 + i * step
        a = ease_out(anim(t, ti, 0.25))
        if a <= 0:
            continue
        y = 205 + i * 82
        check_icon(d, 74, y + 22, 24, ease_out_back(anim(t, ti + 0.1, 0.3)))
        d.text((118 - (1 - a) * 30, y + 38), it, font=F("semi", 44), fill=(255, 255, 255, int(255 * a)), anchor="ls")
    return finish(img)


RENDERERS = {
    "meter": card_meter,
    "bars": card_bars,
    "people": card_people,
    "finance": card_finance,
    "chat": card_chat,
    "notes": card_notes,
}


def render_card(cfg, t, ctx, width=900):
    return RENDERERS[cfg["type"]](cfg, t, ctx, width)
