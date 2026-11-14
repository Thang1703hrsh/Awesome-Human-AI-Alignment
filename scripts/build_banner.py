#!/usr/bin/env python3
"""Draw the README banner (Figure/banner.svg) as a self-contained SVG with text converted to outlines.

Fonts: the variable TTFs of Inter and JetBrains Mono from github.com/google/fonts (SIL Open Font License),
placed in the directory given by --fonts. Requires ``fonttools``.

The four dimension colors follow the lifecycle taxonomy figure (Figure/Taxonomy.pdf):
specification green, supervision blue, mechanisms red, assurance purple.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 400
LEFT = 68
TEXT_MAX = 580
TITLE_Y = 123

GREEN, BLUE, RED, PURPLE = "#4fd1a1", "#6ea8ff", "#ff8a8a", "#b18cff"
PINK, CYAN, AMBER = "#f78fb3", "#8be9ff", "#ffd166"

DIMENSIONS = (
    ("Specification", "specify", GREEN),
    ("Supervision", "supervise", BLUE),
    ("Mechanisms", "train & steer", RED),
    ("Assurance", "assure", PURPLE),
)


def num(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


class Face:
    def __init__(self, path: Path, **axes: float):
        font = TTFont(path)
        tags = {a.axisTag for a in font["fvar"].axes}
        self.font = instantiateVariableFont(font, {k: v for k, v in axes.items() if k in tags})
        self.upem = self.font["head"].unitsPerEm
        self.cmap = self.font.getBestCmap()
        self.glyphs = self.font.getGlyphSet()
        self.hmtx = self.font["hmtx"]
        self.pairs = self._pair_subtables()

    def _pair_subtables(self):
        if "GPOS" not in self.font:
            return []
        gpos = self.font["GPOS"].table
        indices = sorted({i for fr in gpos.FeatureList.FeatureRecord if fr.FeatureTag == "kern" for i in fr.Feature.LookupListIndex})
        out = []
        for i in indices:
            lookup = gpos.LookupList.Lookup[i]
            for st in lookup.SubTable:
                if lookup.LookupType == 9:
                    st = st.ExtSubTable
                if st.LookupType == 2:
                    out.append(st)
        return out

    def kern(self, left: str, right: str) -> float:
        for st in self.pairs:
            cov = st.Coverage.glyphs
            if left not in cov:
                continue
            if st.Format == 1:
                for rec in st.PairSet[cov.index(left)].PairValueRecord:
                    if rec.SecondGlyph == right:
                        return getattr(rec.Value1, "XAdvance", 0) or 0
            else:
                c1 = st.ClassDef1.classDefs.get(left, 0)
                c2 = st.ClassDef2.classDefs.get(right, 0)
                value = st.Class1Record[c1].Class2Record[c2].Value1
                adv = getattr(value, "XAdvance", 0) if value is not None else 0
                if adv:
                    return adv
        return 0

    def _advances(self, text: str, size: float, tracking: float):
        s = size / self.upem
        names = [self.cmap[ord(c)] for c in text]
        for i, name in enumerate(names):
            adv = self.hmtx[name][0]
            if i + 1 < len(names):
                adv += self.kern(name, names[i + 1])
            yield name, adv * s + tracking

    def width(self, text: str, size: float, tracking: float = 0) -> float:
        return sum(adv for _, adv in self._advances(text, size, tracking)) - tracking

    def path(self, text: str, x: float, y: float, size: float, tracking: float = 0) -> str:
        s = size / self.upem
        pen = SVGPathPen(self.glyphs, ntos=num)
        for name, adv in self._advances(text, size, tracking):
            self.glyphs[name].draw(TransformPen(pen, (s, 0, 0, -s, x, y)))
            x += adv
        return pen.getCommands()


def check(x: float, y: float, r: float, color: str) -> str:
    """Colored badge with a check mark."""
    d = f"M{num(x - r * 0.42)} {num(y + r * 0.02)}L{num(x - r * 0.1)} {num(y + r * 0.34)}L{num(x + r * 0.45)} {num(y - r * 0.32)}"
    return (
        f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="{color}"/>'
        f'<path d="{d}" fill="none" stroke="#0b1020" stroke-width="{num(r * 0.3)}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def polar(cx: float, cy: float, r: float, degrees: float) -> tuple[float, float]:
    """Point on a circle; angles grow clockwise on screen, starting from the positive x axis."""
    a = math.radians(degrees)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def human(x: float, y: float) -> str:
    """A person standing at (x, y), right arm raised toward the lifecycle ring."""
    skin, hair, shirt, legs = "#f6d7bd", "#3a2a4a", "url(#shirt)", "#8e9ad0"
    parts = [
        '<ellipse cx="0" cy="4" rx="42" ry="7" fill="#000" fill-opacity=".35"/>',
        f'<rect x="-19" y="-46" width="14" height="44" rx="7" fill="{legs}"/>',
        f'<rect x="5" y="-46" width="14" height="44" rx="7" fill="{legs}"/>',
        f'<ellipse cx="-13" cy="-3" rx="11" ry="5.5" fill="{BLUE}"/>',
        f'<ellipse cx="13" cy="-3" rx="11" ry="5.5" fill="{BLUE}"/>',
        f'<path d="M-27 -98L-41 -60" stroke="{GREEN}" stroke-width="12" stroke-linecap="round"/>',
        f'<circle cx="-42" cy="-55" r="7.5" fill="{skin}"/>',
        f'<path d="M27 -98L49 -124" stroke="{GREEN}" stroke-width="12" stroke-linecap="round"/>',
        f'<circle cx="52" cy="-128" r="7.5" fill="{skin}"/>',
        f'<rect x="-31" y="-112" width="62" height="74" rx="22" fill="{shirt}"/>',
        f'<rect x="-7" y="-124" width="14" height="14" rx="4" fill="{skin}"/>',
        f'<circle cx="0" cy="-146" r="25" fill="{skin}"/>',
        f'<path d="M-26 -144C-28 -166 -12 -175 2 -174C18 -173 29 -162 26 -144C19 -155 8 -158 -4 -156C-13 -155 -21 -151 -26 -144Z" fill="{hair}"/>',
        '<g class="blink">'
        '<circle cx="-9" cy="-143" r="3" fill="#0d1328"/><circle cx="9" cy="-143" r="3" fill="#0d1328"/>'
        "</g>",
        '<path d="M-7 -133Q0 -127 7 -133" fill="none" stroke="#0d1328" stroke-width="2.4" stroke-linecap="round"/>',
        f'<ellipse cx="-16" cy="-134" rx="5" ry="2.8" fill="{PINK}" fill-opacity=".55"/>',
        f'<ellipse cx="16" cy="-134" rx="5" ry="2.8" fill="{PINK}" fill-opacity=".55"/>',
        f'<circle cx="0" cy="-86" r="9" fill="#0d1328" fill-opacity=".35"/>',
        f'<path d="M0 -81C-7 -86 -7 -93 -3 -94C-1 -95 0 -93 0 -92C0 -93 1 -95 3 -94C7 -93 7 -86 0 -81Z" fill="{PINK}" class="beat"/>',
    ]
    return f'<g transform="translate({num(x)} {num(y)})">' + "".join(parts) + "</g>"


def robot(x: float, y: float) -> str:
    """A small robot standing at (x, y), mirrored so that its raised arm points left toward the ring."""
    body, limb = 'fill="url(#robotBody)"', "#b9c7ec"
    parts = [
        '<ellipse cx="0" cy="4" rx="46" ry="7" fill="#000" fill-opacity=".35"/>',
        f'<rect x="-21" y="-28" width="13" height="24" rx="6" fill="{limb}"/>',
        f'<rect x="8" y="-28" width="13" height="24" rx="6" fill="{limb}"/>',
        f'<ellipse cx="-15" cy="-3" rx="11" ry="5.5" fill="{BLUE}"/>',
        f'<ellipse cx="15" cy="-3" rx="11" ry="5.5" fill="{BLUE}"/>',
        f'<path d="M-33 -68L-49 -42" stroke="{limb}" stroke-width="12" stroke-linecap="round"/>',
        f'<circle cx="-50" cy="-40" r="7.5" fill="{BLUE}"/>',
        f'<path d="M33 -68L57 -90" stroke="{limb}" stroke-width="12" stroke-linecap="round"/>',
        f'<circle cx="59" cy="-92" r="7.5" fill="{BLUE}"/>',
        f'<rect x="-35" y="-84" width="70" height="60" rx="18" {body}/>',
        '<circle cx="0" cy="-54" r="10" fill="#0d1328"/>',
        f'<circle cx="0" cy="-54" r="5" fill="{PURPLE}" class="beat"/>',
        f'<rect x="-10" y="-92" width="20" height="10" rx="3" fill="{limb}"/>',
        f'<rect x="-57" y="-138" width="11" height="28" rx="5" fill="{BLUE}"/>',
        f'<rect x="46" y="-138" width="11" height="28" rx="5" fill="{BLUE}"/>',
        f'<rect x="-49" y="-162" width="98" height="74" rx="26" {body}/>',
        '<rect x="-39" y="-152" width="78" height="54" rx="18" fill="#0d1328"/>',
        '<g class="blink">'
        f'<rect x="-23" y="-138" width="13" height="19" rx="6.5" fill="{CYAN}"/><rect x="10" y="-138" width="13" height="19" rx="6.5" fill="{CYAN}"/>'
        '<circle cx="-14" cy="-133" r="2.6" fill="#fff"/><circle cx="19" cy="-133" r="2.6" fill="#fff"/>'
        "</g>",
        f'<path d="M-9 -110Q0 -103 9 -110" fill="none" stroke="{CYAN}" stroke-width="2.6" stroke-linecap="round"/>',
        f'<ellipse cx="-28" cy="-110" rx="5.5" ry="3" fill="{PINK}" fill-opacity=".6"/>',
        f'<ellipse cx="28" cy="-110" rx="5.5" ry="3" fill="{PINK}" fill-opacity=".6"/>',
        f'<path d="M0 -162V-180" stroke="{limb}" stroke-width="4" stroke-linecap="round"/>',
        f'<circle cx="0" cy="-186" r="13" fill="{AMBER}" opacity=".25" class="halo"/>',
        f'<circle cx="0" cy="-186" r="7" fill="{AMBER}"/>',
    ]
    return f'<g transform="translate({num(x)} {num(y)}) scale(-1 1)">' + "".join(parts) + "</g>"


def build(fonts: Path) -> str:
    inter = fonts / "Inter[opsz,wght].ttf"
    mono = fonts / "JetBrainsMono[wght].ttf"
    title_face = Face(inter, wght=760, opsz=32)
    body_face = Face(inter, wght=420, opsz=20)
    pill_face = Face(inter, wght=600, opsz=14)
    mono_face = Face(mono, wght=500)

    out: list[str] = []
    add = out.append

    # Title block.
    title_size = 58
    lines = ["Awesome Human–AI", "Alignment"]
    while max(title_face.width(t, title_size, -1.2) for t in lines) > TEXT_MAX:
        title_size -= 1
    add(f'<path fill="#f4f6fb" d="{title_face.path(lines[0], LEFT, TITLE_Y, title_size, -1.2)}"/>')
    add(f'<path fill="url(#titleGrad)" d="{title_face.path(lines[1], LEFT, TITLE_Y + title_size * 1.08, title_size, -1.2)}"/>')
    sub = ["A lifecycle survey of specifying, supervising, implementing,", "and assuring the alignment of AI systems with people"]
    sub_size = 20
    while max(body_face.width(t, sub_size) for t in sub) > TEXT_MAX:
        sub_size -= 0.5
    sub_y = TITLE_Y + title_size * 1.08 + 48
    for i, text in enumerate(sub):
        add(f'<path fill="#aab4cc" d="{body_face.path(text, LEFT, sub_y + i * sub_size * 1.4, sub_size)}"/>')

    # Dimension pills, colored as in the lifecycle taxonomy figure.
    x, top, h = LEFT, sub_y + sub_size * 1.4 + 26, 32
    for label, _, color in DIMENSIONS:
        w = pill_face.width(label, 15) + 32
        add(f'<rect x="{num(x)}" y="{num(top)}" width="{num(w)}" height="{h}" rx="16" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-opacity=".55"/>')
        add(f'<path fill="{color}" d="{pill_face.path(label, x + 16, top + 21.3, 15)}"/>')
        x += w + 10
    assert x - 10 < LEFT + TEXT_MAX + 40, f"pills overflow: {x}"
    assert top + h < H - 24, "pills overflow"

    # Illustration: a person's intent passes through the four-stage lifecycle to an aligned model.
    cx, cy, r = 942, 196, 96
    add(f'<circle cx="{cx}" cy="{cy}" r="{r + 26}" fill="url(#ringGlow)"/>')
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#ffffff" stroke-opacity=".06" stroke-width="22"/>')
    gap = 9
    for k, (_, verb, color) in enumerate(DIMENSIONS):
        start, end = 180 + 90 * k + gap, 270 + 90 * k - gap - 6
        (x0, y0), (x1, y1) = polar(cx, cy, r, start), polar(cx, cy, r, end)
        tip = polar(cx, cy, r, end + 7)
        a = math.radians(end + 7)
        tx, ty = -math.sin(a), math.cos(a)  # clockwise tangent at the tip
        nx, ny = math.cos(a), math.sin(a)
        bx, by = tip[0] - tx * 13, tip[1] - ty * 13
        head = f"M{num(tip[0])} {num(tip[1])}L{num(bx + nx * 11)} {num(by + ny * 11)}L{num(bx - nx * 11)} {num(by - ny * 11)}Z"
        add(f'<g class="stage s{k}">'
            f'<path d="M{num(x0)} {num(y0)}A{r} {r} 0 0 1 {num(x1)} {num(y1)}" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"/>'
            f'<path d="{head}" fill="{color}" stroke="{color}" stroke-width="3" stroke-linejoin="round"/></g>')
        # Stage label outside the ring, on the diagonal of its quadrant.
        lx, ly = polar(cx, cy, r + 34, 225 + 90 * k)
        lw = mono_face.width(verb, 13, 0.4)
        anchor_x = lx - lw if lx < cx else lx
        add(f'<path fill="{color}" fill-opacity=".9" d="{mono_face.path(verb, anchor_x, ly + (4 if ly < cy else 9), 13, 0.4)}"/>')

    # Inside the ring: the overlap between human intent and model behavior, marked as aligned.
    vr, dx = 34, 22
    add(f'<circle cx="{cx - dx}" cy="{cy}" r="{vr}" fill="{GREEN}" fill-opacity=".14" stroke="{GREEN}" stroke-opacity=".7" stroke-width="2"/>')
    add(f'<circle cx="{cx + dx}" cy="{cy}" r="{vr}" fill="{BLUE}" fill-opacity=".14" stroke="{BLUE}" stroke-opacity=".7" stroke-width="2"/>')
    add(f'<clipPath id="lens"><circle cx="{cx - dx}" cy="{cy}" r="{vr}"/></clipPath>')
    add(f'<circle cx="{cx + dx}" cy="{cy}" r="{vr}" fill="url(#lensGrad)" clip-path="url(#lens)"/>')
    add(f'<g class="pop">{check(cx, cy, 11, "#f4f6fb")}</g>')

    # The person on the left and the robot on the right, linked to the ring.
    hx, rx_ = 726, 1190
    assert rx_ + 49 < W - 36, "robot overflows"
    assert hx - 42 > LEFT + TEXT_MAX, "person overlaps the title block"
    add(human(hx, 336))
    add(robot(rx_, 336))
    hand_h = (hx + 52 + 8, 336 - 128)
    ring_l = (cx - r - 14, cy)
    add(f'<path d="M{num(hand_h[0])} {num(hand_h[1])}C{num(hand_h[0] + 22)} {num(hand_h[1])} {num(ring_l[0] - 22)} {cy} {num(ring_l[0])} {cy}" fill="none" stroke="{GREEN}" stroke-opacity=".85" stroke-width="2" stroke-linecap="round" class="flow"/>')
    hand_r = (rx_ - 59 - 8, 336 - 92)
    ring_r = (cx + r + 14, cy)
    add(f'<path d="M{num(ring_r[0])} {cy}C{num(ring_r[0] + 22)} {cy} {num(hand_r[0] - 22)} {num(hand_r[1])} {num(hand_r[0])} {num(hand_r[1])}" fill="none" stroke="{PURPLE}" stroke-opacity=".85" stroke-width="2" stroke-linecap="round" class="flow"/>')

    # Captions under the two figures.
    for text, x_mid, color in (("human intent", hx, GREEN), ("aligned model", rx_, "#c9b6ff")):
        add(f'<path fill="{color}" fill-opacity=".85" d="{mono_face.path(text, x_mid - mono_face.width(text, 13, 0.4) / 2, 368, 13, 0.4)}"/>')

    defs = f"""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#0a0f1f"/><stop offset=".55" stop-color="#10163a"/><stop offset="1" stop-color="#1b1238"/>
</linearGradient>
<radialGradient id="glow" cx="{cx}" cy="210" r="440" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="#3f63e0" stop-opacity=".30"/><stop offset="1" stop-color="#3f63e0" stop-opacity="0"/>
</radialGradient>
<radialGradient id="glow2" cx="1180" cy="40" r="300" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="{PURPLE}" stop-opacity=".18"/><stop offset="1" stop-color="{PURPLE}" stop-opacity="0"/>
</radialGradient>
<radialGradient id="ringGlow" cx="{cx}" cy="{cy}" r="{r + 26}" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="{BLUE}" stop-opacity=".20"/><stop offset=".75" stop-color="{BLUE}" stop-opacity=".06"/><stop offset="1" stop-color="{BLUE}" stop-opacity="0"/>
</radialGradient>
<linearGradient id="lensGrad" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="{GREEN}" stop-opacity=".75"/><stop offset="1" stop-color="{BLUE}" stop-opacity=".75"/>
</linearGradient>
<linearGradient id="titleGrad" x1="{LEFT}" y1="0" x2="{LEFT + 330}" y2="0" gradientUnits="userSpaceOnUse">
<stop offset="0" stop-color="{GREEN}"/><stop offset=".35" stop-color="{BLUE}"/><stop offset=".7" stop-color="{PURPLE}"/><stop offset="1" stop-color="{RED}"/>
</linearGradient>
<pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
<circle cx="1" cy="1" r="1" fill="#ffffff" fill-opacity=".07"/>
</pattern>
<linearGradient id="gridFade" x1="0" y1="0" x2="1" y2="0">
<stop offset="0" stop-color="#fff" stop-opacity=".25"/><stop offset=".6" stop-color="#fff" stop-opacity="1"/>
</linearGradient>
<mask id="gridMask"><rect width="{W}" height="{H}" fill="url(#gridFade)"/></mask>
<linearGradient id="robotBody" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#f7f9ff"/><stop offset="1" stop-color="#c4d0f0"/>
</linearGradient>
<linearGradient id="shirt" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#7fe3bd"/><stop offset="1" stop-color="#3fae86"/>
</linearGradient>
<clipPath id="frame"><rect width="{W}" height="{H}" rx="24"/></clipPath>
</defs>"""

    style = """<style>
.flow{stroke-dasharray:3 6;animation:flow 1.4s linear infinite}
@keyframes flow{to{stroke-dashoffset:-18}}
.halo,.blink,.beat,.pop{transform-box:fill-box;transform-origin:center}
.halo{animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.15;transform:scale(.8)}50%{opacity:.45;transform:scale(1.25)}}
.blink{animation:blink 4.5s infinite}
@keyframes blink{0%,93%,100%{transform:scaleY(1)}96%{transform:scaleY(.12)}}
.beat{animation:beat 1.6s ease-in-out infinite}
@keyframes beat{0%,100%{opacity:.55}50%{opacity:1}}
.stage{animation:stage 4.8s infinite both}
.s1{animation-delay:1.2s}.s2{animation-delay:2.4s}.s3{animation-delay:3.6s}
@keyframes stage{0%,100%{opacity:.55}8%,22%{opacity:1}35%{opacity:.55}}
.pop{animation:pop 4.8s infinite both}
@keyframes pop{0%,70%{transform:scale(1)}78%{transform:scale(1.2)}86%,100%{transform:scale(1)}}
@media (prefers-reduced-motion:reduce){.flow,.halo,.blink,.beat,.stage,.pop{animation:none}.stage{opacity:1}}
</style>"""

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Awesome Human-AI Alignment: a lifecycle survey of specifying, supervising, implementing, and assuring the alignment of AI systems with people">',
            "<title>Awesome Human-AI Alignment</title>",
            defs,
            style,
            '<g clip-path="url(#frame)">',
            f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
            f'<rect width="{W}" height="{H}" fill="url(#glow)"/>',
            f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>',
            f'<rect width="{W}" height="{H}" fill="url(#grid)" mask="url(#gridMask)"/>',
            *out,
            "</g>",
            f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="23.5" fill="none" stroke="#ffffff" stroke-opacity=".08"/>',
            "</svg>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "Figure" / "banner.svg")
    args = parser.parse_args()
    args.out.write_text(build(args.fonts), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
