"""
Flowers of Spring - painted with code, not with pixels.

A small watercolour engine written in the spirit of the p5.brush experiments
(huggingface.co/blog/train-to-paint-with-code): every mark is a polygon stamped
many times over, each copy nudged a little further out by a random walk. That
walk is the bleed of a wet edge. Pigment composites multiplicatively, the way
layered washes actually darken on paper, and a single noise field standing in
for the grain of the sheet decides where the pigment settles and where it
skips. There is no diffusion model here and no photograph - the picture below
is the return value of this program.

    python3 art/flowers_of_spring.py

Writes blog/assets/images/flowers-of-spring.jpg
"""

import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1500, 1000
SS = 2                                  # supersample, then downsample to W x H
RW, RH = W * SS, H * SS

SEED = 1953
random.seed(SEED)
np.random.seed(SEED)

PAPER = (0.988, 0.974, 0.949)

SKY_ORANGE = (0.97, 0.60, 0.26)
SKY_AMBER = (0.99, 0.79, 0.44)
SKY_ROSE = (0.95, 0.55, 0.47)
HILL = (0.74, 0.75, 0.58)
FIELD = (0.62, 0.71, 0.45)
FIELD_DEEP = (0.40, 0.54, 0.33)
GRASS = (0.47, 0.60, 0.35)
BLUE = (0.30, 0.37, 0.72)
RED = (0.82, 0.25, 0.28)
PURPLE = (0.50, 0.29, 0.61)
BROWN = (0.50, 0.37, 0.25)
STEM = (0.55, 0.63, 0.38)


def _noise_field(scale, blur):
    """One tileable-enough sheet of grain, reused by every mark."""
    n = np.random.rand(RH // scale + 2, RW // scale + 2).astype(np.float32)
    im = Image.fromarray((n * 255).astype(np.uint8)).resize((RW, RH), Image.BICUBIC)
    return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255.0


class Canvas:
    def __init__(self):
        self.px = np.full((RH, RW, 3), PAPER, np.float32)
        self.grain = _noise_field(4, 1.3)
        self.px = np.clip(self.px + (self.grain[:, :, None] - 0.5) * 0.05, 0, 1)

    # -- geometry -----------------------------------------------------------
    @staticmethod
    def _walk(pts, amount, wander):
        """Random-walk the vertices outward: the bleed of a wet edge."""
        out = []
        ang = random.uniform(0, math.tau)
        for x, y in pts:
            ang += random.uniform(-wander, wander)
            r = amount * random.uniform(0.35, 1.0)
            out.append((x + math.cos(ang) * r, y + math.sin(ang) * r))
        return out

    def _accumulate(self, pts, bleed, layers, alpha, blur, wander):
        """Stamp the polygon `layers` times inside its own bounding box."""
        size = _extent(pts)
        pad = int(bleed * size + blur * 3 + 4)
        x0 = max(0, int(min(p[0] for p in pts)) - pad)
        y0 = max(0, int(min(p[1] for p in pts)) - pad)
        x1 = min(RW, int(max(p[0] for p in pts)) + pad)
        y1 = min(RH, int(max(p[1] for p in pts)) + pad)
        if x1 <= x0 or y1 <= y0:
            return None, None
        w, h = x1 - x0, y1 - y0
        acc = np.zeros((h, w), np.float32)
        im = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(im)
        for i in range(layers):
            t = (i + 1) / layers
            poly = self._walk(pts, bleed * size * (t ** 0.65), wander)
            draw.rectangle((0, 0, w, h), fill=0)
            draw.polygon([(x - x0, y - y0) for x, y in poly], fill=255)
            src = im.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else im
            acc += np.asarray(src, np.float32) * (alpha / 255.0)
        return acc, (x0, y0, x1, y1)

    # -- paint --------------------------------------------------------------
    def fill(self, pts, color, bleed=0.22, layers=22, alpha=0.05,
             texture=0.55, blur=1.6, wander=0.9):
        """Transparent pigment: multiply, so washes darken where they overlap."""
        acc, box = self._accumulate(pts, bleed, layers, alpha, blur, wander)
        if acc is None:
            return
        x0, y0, x1, y1 = box
        if texture > 0:
            acc *= 1.0 - texture * (1.0 - self.grain[y0:y1, x0:x1])
        a = np.clip(acc, 0, 1)[:, :, None]
        self.px[y0:y1, x0:x1] *= 1.0 - a * (1.0 - np.array(color, np.float32))

    def lift(self, pts, color, amount=0.55, bleed=0.12, layers=8,
             alpha=0.09, blur=1.2):
        """Opaque body colour - the only way white shows over a dark wash."""
        acc, box = self._accumulate(pts, bleed, layers, alpha, blur, 0.7)
        if acc is None:
            return
        x0, y0, x1, y1 = box
        a = np.clip(acc * amount, 0, 1)[:, :, None]
        self.px[y0:y1, x0:x1] += a * (np.array(color, np.float32) - self.px[y0:y1, x0:x1])

    def stroke(self, path, color, width, **kw):
        """A tapered brush stroke: the path walked out one side and back."""
        n = len(path)
        left, right = [], []
        for i, (x, y) in enumerate(path):
            t = i / max(n - 1, 1)
            w = width * math.sin(math.pi * min(1.0, 0.12 + t * 0.95)) ** 0.6
            px, py = path[min(i + 1, n - 1)]
            qx, qy = path[max(i - 1, 0)]
            dx, dy = px - qx, py - qy
            m = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / m, dx / m
            left.append((x + nx * w, y + ny * w))
            right.append((x - nx * w, y - ny * w))
        self.fill(left + right[::-1], color, **kw)

    def save(self, path):
        im = Image.fromarray((np.clip(self.px, 0, 1) * 255).astype(np.uint8))
        im.resize((W, H), Image.LANCZOS).save(path, quality=86, optimize=True, subsampling=1)


def _extent(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0


def blob(cx, cy, rx, ry, n=20, rough=0.16):
    return [(cx + math.cos(math.tau * i / n) * rx * (1 + random.uniform(-rough, rough)),
             cy + math.sin(math.tau * i / n) * ry * (1 + random.uniform(-rough, rough)))
            for i in range(n)]


def band(y, h, wob=40, n=24):
    top = [(RW * i / n, y + math.sin(i * 0.7) * wob * 0.5
            + random.uniform(-wob, wob) * 0.4) for i in range(n + 1)]
    return top + [(x, y + h + random.uniform(-wob, wob) * 0.3) for x, _ in reversed(top)]


# ---------------------------------------------------------------- composition
c = Canvas()
HORIZON = RH * 0.46

# sky - amber ground, then orange and rose floated on top of it wet-in-wet
c.fill(band(-RH * 0.10, HORIZON + RH * 0.12, wob=70), SKY_AMBER,
       bleed=0.04, layers=18, alpha=0.060, texture=0.30, blur=10)
for i in range(12):
    y = random.uniform(-RH * 0.06, HORIZON * 0.85)
    weight = 1.0 - y / (HORIZON * 1.4)          # heavier colour up top
    c.fill(blob(random.uniform(0, RW), y, random.uniform(340, 700),
                random.uniform(55, 150), rough=0.30),
           SKY_ORANGE if i % 2 else SKY_ROSE,
           bleed=0.20, layers=14, alpha=0.018 + 0.030 * weight,
           texture=0.40, blur=13)

# a low sun, lifted back out of the wash
c.lift(blob(RW * 0.745, HORIZON * 0.60, 112, 106, rough=0.05), (1.0, 0.95, 0.78),
       amount=0.55, bleed=0.26, layers=10, alpha=0.07, blur=16)

# far hills, sitting on the horizon and holding the sky off the field
c.fill(band(HORIZON - RH * 0.055, RH * 0.075, wob=22), HILL,
       bleed=0.08, layers=16, alpha=0.055, texture=0.45, blur=6)
c.fill(band(HORIZON - RH * 0.015, RH * 0.05, wob=14), FIELD_DEEP,
       bleed=0.09, layers=12, alpha=0.035, texture=0.5, blur=6)

# meadow - lighter where the light lands, deepening toward the foreground
c.fill(band(HORIZON, RH * 0.62, wob=16), FIELD,
       bleed=0.05, layers=20, alpha=0.055, texture=0.45, blur=6)
c.fill(band(RH * 0.72, RH * 0.34, wob=40), FIELD_DEEP,
       bleed=0.14, layers=16, alpha=0.035, texture=0.55, blur=10)
for i in range(14):
    y = random.uniform(HORIZON + 20, RH * 0.99)
    d = (y - HORIZON) / (RH - HORIZON)
    c.fill(blob(random.uniform(0, RW), y, random.uniform(220, 560),
                random.uniform(40, 110), rough=0.28),
           FIELD_DEEP if d > 0.5 else (0.72, 0.74, 0.46),
           bleed=0.22, layers=12, alpha=0.030, texture=0.55, blur=9)

# the trampled patch - blue and red gone to purple, purple going back to brown
BOOT_X, BOOT_Y = RW * 0.36, RH * 0.88
for k in range(3):
    c.fill(blob(BOOT_X + random.uniform(-160, 160), BOOT_Y + random.uniform(-30, 30),
                random.uniform(150, 260), random.uniform(55, 85), rough=0.36), BROWN,
           bleed=0.24, layers=14, alpha=0.030, texture=0.60, blur=9)

# grass
for i in range(620):
    x = random.uniform(-40, RW + 40)
    depth = random.random() ** 1.5
    y = HORIZON + 14 + depth * (RH - HORIZON)
    h = 24 + depth * 200
    lean = random.uniform(-0.55, 0.55)
    path = [(x + lean * h * (t / 6) ** 2, y - h * t / 6) for t in range(7)]
    c.stroke(path, GRASS if random.random() > 0.4 else FIELD_DEEP,
             1.2 + depth * 3.2, bleed=0.10, layers=5, alpha=0.17,
             texture=0.40, blur=0.9, wander=0.4)

# wildflowers - blue, red, and the purples they blend into
for i in range(165):
    depth = random.random() ** 1.35 if i < 150 else random.uniform(0.85, 1.0)
    x = random.uniform(0, RW)
    y = HORIZON + 25 + depth * (RH - HORIZON - 40)
    r = 5 + depth * 26
    trampled = abs(x - BOOT_X) < 360 and abs(y - BOOT_Y) < 130
    if trampled:
        col = BROWN if random.random() < 0.6 else PURPLE
    else:
        col = random.choice([BLUE, RED, PURPLE, PURPLE, BLUE, RED])
    head = y - r * 2.8
    c.stroke([(x, y), (x, head)], STEM, 1.4 + depth * 2.2,
             bleed=0.10, layers=4, alpha=0.11, texture=0.45, blur=1.4)
    for p in range(5):
        a = math.tau * p / 5 + random.uniform(-0.2, 0.2)
        c.fill(blob(x + math.cos(a) * r * 0.72, head + math.sin(a) * r * 0.72,
                    r * 0.66, r * 0.62, n=12, rough=0.26), col,
               bleed=0.18, layers=8, alpha=0.095, texture=0.40, blur=1.8)


def dandelion(x, y, r, seeds=70, stem_to=None):
    """A clock: stem, a shadow of pigment behind it, filaments lifted in white."""
    stem_to = stem_to if stem_to is not None else RH + 20
    sway = random.uniform(-0.22, 0.22)
    span = stem_to - y
    c.stroke([(x + sway * span * ((1 - t / 7) ** 2), y + span * t / 7)
              for t in range(8)], (0.42, 0.50, 0.30), 3.0,
             bleed=0.09, layers=6, alpha=0.14, texture=0.40, blur=1.6)
    # a soft grey halo so the white filaments have something to sit against
    c.fill(blob(x, y, r * 1.02, r * 1.02, n=18, rough=0.16), (0.78, 0.80, 0.76),
           bleed=0.30, layers=12, alpha=0.028, texture=0.5, blur=8)
    for i in range(seeds):
        a = math.tau * i / seeds + random.uniform(-0.04, 0.04)
        rr = r * random.uniform(0.68, 1.0)
        ex, ey = x + math.cos(a) * rr, y + math.sin(a) * rr
        ix, iy = x + math.cos(a) * r * 0.10, y + math.sin(a) * r * 0.10
        c.lift([(ix, iy), (ex - math.sin(a) * 1.2, ey + math.cos(a) * 1.2),
                (ex + math.sin(a) * 1.2, ey - math.cos(a) * 1.2)],
               (1.0, 1.0, 0.98), amount=0.72, bleed=0.03, layers=4,
               alpha=0.16, blur=1.1)
        c.lift(blob(ex, ey, r * 0.075, r * 0.075, n=8, rough=0.3), (1.0, 1.0, 0.99),
               amount=0.85, bleed=0.08, layers=3, alpha=0.22, blur=1.0)
    c.fill(blob(x, y, r * 0.07, r * 0.07, n=8), (0.55, 0.55, 0.48),
           bleed=0.2, layers=6, alpha=0.10, texture=0.3, blur=1.4)


dandelion(RW * 0.125, RH * 0.520, 118)
dandelion(RW * 0.255, RH * 0.640, 92)
dandelion(RW * 0.855, RH * 0.575, 74)

# seeds gone off on the wind, heading for the orange
for i in range(30):
    t = i / 29
    x = RW * (0.28 + 0.66 * t) + random.uniform(-45, 45)
    y = RH * (0.50 - 0.42 * t ** 1.25) + random.uniform(-50, 50)
    s = 17 - 8.5 * t
    c.lift(blob(x, y, s * 0.24, s * 0.24, n=8, rough=0.3), (1.0, 1.0, 0.99),
           amount=0.90, bleed=0.06, layers=3, alpha=0.26, blur=1.0)
    for k in range(9):
        a = math.tau * k / 9 + t
        c.lift([(x, y), (x + math.cos(a) * s - math.sin(a), y + math.sin(a) * s + math.cos(a)),
                (x + math.cos(a) * s + math.sin(a), y + math.sin(a) * s - math.cos(a))],
               (1.0, 1.0, 0.98), amount=0.62, bleed=0.03, layers=3,
               alpha=0.18, blur=0.9)

out = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "blog", "assets", "images", "flowers-of-spring.jpg"))
c.save(out)
print("wrote", out)
