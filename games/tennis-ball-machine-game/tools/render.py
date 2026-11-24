"""Render one Proton-style card (PNG) per level + a combined PDF.
usage: python3 render.py <game_dir> [level numbers...]"""
import csv, math, os, re, sys, html

GAME = sys.argv[1]
ONLY = [int(x) for x in sys.argv[2:]]
OUT = os.path.join(GAME, "level-cards")
os.makedirs(OUT, exist_ok=True)

SPEED = {"S1": 60, "S2": 70, "S3": 80, "S4": 90}
SPIN = {"Flat": 0, "Top 1": 500, "Top 2": 1000, "Top 3": 2000, "Slice 1": -500}
MIX_CYCLE = ["Top 1", "Top 2", "Slice 1", "Top 3"]
ROW_T = {"D": 0.05, "M": 0.165, "S": 0.40}           # player's side, 0 = his baseline
ROW_NAME = {"D": "back row", "M": "middle row", "S": "front row"}
COL_U = {"FW": 0.1, "F": 0.3, "C": 0.5, "B": 0.7, "BW": 0.9}  # screen left -> right
COL_NAME = {"FW": "1st", "F": "2nd", "C": "3rd", "B": "4th", "BW": "5th"}
LANE_WORD = {"FW": "wide forehand", "F": "forehand", "C": "center", "B": "backhand", "BW": "wide backhand"}
TARGET_TEXT = {
    "A": "anywhere in the singles court",
    "B": "past the service line",
    "C": "crosscourt half",
    "D": "down-the-line half",
    "E": "crosscourt corner box (cones)",
    "F": "down-the-line corner box (cones)",
    "G": "middle deep box (cones)",
}

# ---------- perspective ----------
W = 900
CX = W / 2
# near baseline d=1, far d=1.75
K, YH, HW = 2170.0, -1170.0, 400.0
TOP = 0  # vertical offset of the court in the card


def P(u, t):
    d = 1 + (1 - t) * 0.75
    return CX + (u - 0.5) * 2 * HW / d, TOP + YH + K / d


def poly(pts, **attrs):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" {a}/>'


def rect_uv(u0, u1, t0, t1, **a):
    return poly([P(u0, t0), P(u1, t0), P(u1, t1), P(u0, t1)], **a)


def line_uv(u0, t0, u1, t1, w=4, c="#fff"):
    (x0, y0), (x1, y1) = P(u0, t0), P(u1, t1)
    return f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{c}" stroke-width="{w}"/>'


def scale_at(t):
    return 1 / (1 + (1 - t) * 0.75)


# ---------- parsing ----------
SHOT_RE = re.compile(r"^([SMD])-(BW|B|C|FW|F)(?:\s*\((.*?)\))?$")


def parse(row):
    s = row["shots"].strip()
    rnd = s.startswith("RANDOM:")
    if rnd:
        s = s[len("RANDOM:"):].strip()
    if s == "all 15 spots":
        toks = [f"{r}-{c}" for r in "DMS" for c in ["FW", "F", "C", "B", "BW"]]
    else:
        toks = [t.strip() for t in re.split(r"->|,", s) if t.strip()]
    shots = []
    mixed = row["spin"].startswith("Mixed")
    for i, tok in enumerate(toks):
        m = SHOT_RE.match(tok)
        assert m, (row["level"], tok)
        depth, lane, extra = m.groups()
        spin, riser = row["spin"], False
        if extra in ("lob",):
            riser, spin = True, "Flat"
        elif extra == "high":
            riser, spin = True, "Top 2"
        elif extra:
            spin = extra
        if spin.startswith("Mixed"):
            spin = MIX_CYCLE[i % len(MIX_CYCLE)]
        shots.append(dict(depth=depth, lane=lane, spin=spin, riser=riser,
                          speed=SPEED[row["speed"]], interval=float(row["interval"])))
    return rnd, shots


def spin_txt(name):
    v = SPIN[name]
    if v == 0:
        return "0 rpm"
    return f"+{v} rpm topspin" if v > 0 else f"{v} rpm backspin"


# ---------- target zones ----------
BOX_U, BOX_T0 = 3 / 8.23, 1 - 3 / 23.77
SVC_NEAR = 1 - 6.40 / 23.77


def sides_of(shots):
    s = set()
    for sh in shots:
        if sh["lane"] in ("F", "FW"):
            s.add("FH")
        elif sh["lane"] in ("B", "BW"):
            s.add("BH")
        else:
            s.update(["FH", "BH"])
    return [x for x in ("FH", "BH") if x in s]


def zone_rects(letter, side):
    """return list of (u0,u1,t0,t1). FH hits from screen-left, so crosscourt = screen-right."""
    if letter == "A":
        return [(0, 1, 0.5, 1)]
    if letter == "B":
        return [(0, 1, SVC_NEAR, 1)]
    if letter == "G":
        return [(0.5 - 1.5 / 8.23, 0.5 + 1.5 / 8.23, SVC_NEAR + 0.03, 0.95)]
    cross_right = side == "FH"
    if letter in "CE":
        right = cross_right
    else:  # D, F
        right = not cross_right
    if letter in "CD":
        return [(0.5, 1, 0.5, 1)] if right else [(0, 0.5, 0.5, 1)]
    return [(1 - BOX_U, 1, BOX_T0, 1)] if right else [(0, BOX_U, BOX_T0, 1)]


ZONE_COL = ["#ffe94d", "#7fd3ff"]


def draw_targets(target, shots, rnd):
    parts = [p.strip() for p in target.split("/")]
    groups = []  # (letter, shots for it, colour index, label override)
    if len(parts) == 1:
        groups = [(parts[0], shots, 0, None)]
    elif "A" in parts:
        other = [p for p in parts if p != "A"][0]
        groups = [(other, shots, 0, None)]
    elif set(parts) == {"E", "F"}:
        groups = [("E", [dict(lane="F")], 0, "box"), ("E", [dict(lane="B")], 0, "box")]
    elif not rnd and len(shots) % 2 == 0:
        groups = [(parts[0], shots[0::2], 0, None), (parts[1], shots[1::2], 1, None)]
    else:
        groups = [(parts[0], shots, 0, None), (parts[1], shots, 1, None)]
    out, labels, seen = [], [], set()
    for letter, sh, ci, override in groups:
        col = ZONE_COL[ci]
        if letter in "ABG":
            rects = [(r, None) for r in zone_rects(letter, None)]
        else:
            rects = [(r, sd) for sd in sides_of(sh) for r in zone_rects(letter, sd)]
        for (u0, u1, t0, t1), sd in rects:
            key = (round(u0, 3), round(u1, 3), round(t0, 3), round(t1, 3))
            if key in seen:
                continue
            seen.add(key)
            tag = override or (letter if sd is None else f"{letter}·{sd}")
            out.append(rect_uv(u0, u1, t0, t1, fill=col, fill_opacity="0.45", stroke=col,
                               stroke_width="5", stroke_dasharray="14 8"))
            labels.append(((u0 + u1) / 2, (t0 + t1) / 2, tag, col))
    for u, t, tag, col in labels:
        x, y = P(u, t)
        w = 22 + 17 * len(tag)
        out.append(f'<rect x="{x - w / 2:.1f}" y="{y - 22:.1f}" width="{w}" height="44" rx="22" fill="#1f2a44" stroke="{col}" stroke-width="3"/>'
                   f'<text x="{x:.1f}" y="{y + 10:.1f}" text-anchor="middle" font-size="30" font-weight="700" fill="{col}">{html.escape(tag)}</text>')
    return "".join(out)


# ---------- card ----------
def card(row):
    rnd, shots = parse(row)
    n = int(row["level"])
    boss = n % 10 == 0
    balls = int(row["balls"])
    per_loop = len(shots)
    loops = max(1, math.ceil(balls / per_loop))
    riser_any = any(s["riser"] for s in shots)

    svg = []
    # court surface
    svg.append(rect_uv(-0.04, 1.04, -0.03, 1.03, fill="#57b97c"))
    svg.append(draw_targets(row["target"], shots, rnd))
    # lines (singles)
    for a, b in [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 0), (0, 1)), ((1, 0), (1, 1))]:
        svg.append(line_uv(a[0], a[1], b[0], b[1], 6))
    svg.append(line_uv(0, 1 - SVC_NEAR, 1, 1 - SVC_NEAR, 4))
    svg.append(line_uv(0, SVC_NEAR, 1, SVC_NEAR, 4))
    svg.append(line_uv(0.5, 1 - SVC_NEAR, 0.5, SVC_NEAR, 4))
    # net
    (lx, ly), (rx, ry) = P(-0.07, 0.5), P(1.07, 0.5)
    nh = 46
    svg.append(f'<rect x="{lx:.1f}" y="{ly - nh:.1f}" width="{rx - lx:.1f}" height="{nh}" fill="url(#mesh)" stroke="#e8e8e8" stroke-width="3"/>')
    svg.append(f'<rect x="{lx - 4:.1f}" y="{ly - nh - 4:.1f}" width="8" height="{nh + 4}" fill="#eee"/><rect x="{rx - 4:.1f}" y="{ly - nh - 4:.1f}" width="8" height="{nh + 4}" fill="#eee"/>')
    # center mark on player's baseline
    svg.append(line_uv(0.5, 0, 0.5, 0.012, 5))
    # spot rings
    used = {}
    for i, s in enumerate(shots, 1):
        used.setdefault((s["depth"], s["lane"]), []).append(i)
    for d, t in ROW_T.items():
        for c, u in COL_U.items():
            x, y = P(u, t)
            k = scale_at(t)
            r = 34 * k + 8
            if (d, c) in used:
                continue
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" stroke="#fff" stroke-width="5"/>'
                       f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r * 0.5:.1f}" fill="#fff"/>')
    for (d, c), idx in used.items():
        x, y = P(COL_U[c], ROW_T[d])
        k = scale_at(ROW_T[d])
        r = 40 * k + 12
        svg.append(f'<ellipse cx="{x + 3:.1f}" cy="{y + r * 0.85:.1f}" rx="{r * 0.9:.1f}" ry="{r * 0.25:.1f}" fill="#000" opacity=".25"/>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="url(#ball)"/>')
        lab = "?" if rnd else ",".join(map(str, idx))
        fs = 26 if len(lab) < 4 else 20
        svg.append(f'<text x="{x:.1f}" y="{y + fs * 0.36:.1f}" text-anchor="middle" font-size="{fs}" font-weight="800" fill="#1f2a44">{lab}</text>')
    # machine
    mx, my = P(0.5, 1)
    svg.append(f'<g transform="translate({mx:.1f},{my:.1f})"><path d="M-36,-34 L36,-34 L26,-14 L-26,-14Z" fill="#2c4a7a"/>'
               f'<rect x="-20" y="-16" width="40" height="64" rx="6" fill="#111"/><circle cx="0" cy="4" r="7" fill="#555"/><circle cx="0" cy="26" r="9" fill="#333" stroke="#777"/></g>')
    # side labels
    fx, fy = P(0.08, -0.02)
    bx, by = P(0.92, -0.02)
    svg.append(f'<text x="{fx:.1f}" y="{fy - 14:.1f}" text-anchor="middle" font-size="22" fill="#c8d3ea">forehand side</text>'
               f'<text x="{bx:.1f}" y="{by - 14:.1f}" text-anchor="middle" font-size="22" fill="#c8d3ea">backhand side</text>')
    cxm, cym = P(0.5, -0.02)
    svg.append(f'<text x="{cxm:.1f}" y="{cym - 14:.1f}" text-anchor="middle" font-size="22" fill="#c8d3ea">▲ player</text>')
    court_h = P(0.5, 1)[1] + 70

    # shot list
    list_rows = []
    show = shots if len(shots) <= 6 or not rnd else shots
    for i, s in enumerate(show, 1):
        where = f'{ROW_NAME[s["depth"]]}, {COL_NAME[s["lane"]]} from left'
        riser = " · Riser ON" if s["riser"] else ""
        list_rows.append(
            f'<div class="shot"><div class="num">{"?" if rnd else i}</div>'
            f'<div class="where"><b>{s["depth"]}-{s["lane"]}</b> <span>{LANE_WORD[s["lane"]]}</span><small>{where}</small></div>'
            f'<div class="vals"><b>{s["speed"]} km/h</b><small>{spin_txt(s["spin"])}{riser}</small><small>delay {s["interval"]:g} s</small></div></div>')
    if row["shots"] == "RANDOM: all 15 spots":
        list_rows = [f'<div class="shot"><div class="num">?</div><div class="where"><b>All 15 spots</b><small>add one shot per circle; spins rotate +500 / +1000 / −500 / +2000 rpm</small></div>'
                     f'<div class="vals"><b>{SPEED[row["speed"]]} km/h</b><small>delay {float(row["interval"]):g} s</small></div></div>']

    parts = [p.strip() for p in row["target"].split("/")]
    tdesc = " &nbsp;/&nbsp; ".join(
        f'<span style="color:{ZONE_COL[i % 2]}">■</span> <b>{p}</b> {TARGET_TEXT[p]}' for i, p in enumerate(parts))
    notes = html.escape(row["notes"]) if row["notes"] else ""
    lefty = "Left-handed? Tap <b>Mirror</b>."
    stars = f'★ {row["star1"]} &nbsp; ★★ {row["star2"]} &nbsp; ★★★ {row["star3"]} <small>of {balls}</small>'

    return f"""
<section class="card{' boss' if boss else ''}" id="L{n:03d}">
  <header>
    <div class="lvl">LEVEL {n}{' · BOSS' if boss else ''}</div>
    <h1>{html.escape(row['name'])}</h1>
    <div class="sub">World {row['world']}: {html.escape(row['world_name'])} &nbsp;·&nbsp; save as <b>{row['proton_name']}</b></div>
  </header>
  <svg viewBox="0 0 {W} {court_h:.0f}" width="{W}" height="{court_h:.0f}">
    <defs>
      <pattern id="mesh" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M0 0H10M0 0V10" stroke="#8a8f99" stroke-width="2"/></pattern>
      <radialGradient id="ball" cx="35%" cy="30%" r="70%"><stop offset="0" stop-color="#f5ff6a"/><stop offset=".7" stop-color="#d7e21d"/><stop offset="1" stop-color="#a8b000"/></radialGradient>
    </defs>
    {''.join(svg)}
  </svg>
  <div class="bar">
    <div><span>Loops</span><b>{loops}</b></div>
    <div><span>Order</span><b>{'Random 🔀' if rnd else 'In order ➜'}</b></div>
    <div><span>Riser</span><b>{'ON' if riser_any else 'Off'}</b></div>
    <div><span>Balls</span><b>{balls}</b></div>
  </div>
  <div class="shots">{''.join(list_rows)}</div>
  <div class="info">
    <div class="target"><span class="k">HIT TO</span> {tdesc}</div>
    <div class="stars"><span class="k">STARS</span> {stars}</div>
    {f'<div class="note"><span class="k">RULE</span> {notes}</div>' if notes else ''}
    <div class="foot">{lefty} Proton location: baseline center · Stand Off.{' Stop the machine after ' + str(balls) + ' balls.' if loops * per_loop != balls else ''}</div>
  </div>
</section>"""


CSS = """
*{box-sizing:border-box} body{margin:0;background:#2a3550;font-family:Roboto,'Helvetica Neue',Arial,'DejaVu Sans',sans-serif;color:#fff}
.card{width:900px;background:#3d4b6b;padding:0 0 28px;page-break-after:always;break-after:page}
.card.boss header{background:linear-gradient(90deg,#7a3b12,#b8641f)}
header{padding:26px 34px 18px;background:#24304b}
.lvl{font-size:22px;letter-spacing:3px;color:#ffe94d;font-weight:700}
h1{margin:4px 0 6px;font-size:46px;font-weight:600}
.sub{font-size:22px;color:#c8d3ea}
svg{display:block}
.bar{display:grid;grid-template-columns:repeat(4,1fr);background:#24304b;text-align:center;padding:14px 0}
.bar span{display:block;font-size:20px;color:#c8d3ea}.bar b{font-size:30px}
.shots{padding:18px 34px 4px}
.shot{display:flex;align-items:center;background:#e9edf5;color:#1f2a44;border-radius:22px;padding:14px 22px;margin-bottom:12px}
.num{font-size:38px;width:56px;font-weight:600}
.where{flex:1}.where b{font-size:28px}.where span{font-size:22px;margin-left:6px}
.where small,.vals small{display:block;font-size:19px;color:#4a5876}
.vals{text-align:right}.vals b{font-size:28px}
.info{padding:10px 34px 0;font-size:24px;line-height:1.45}
.info>div{margin-top:10px}
.k{display:inline-block;min-width:110px;font-size:18px;letter-spacing:2px;color:#c8d3ea;font-weight:700}
.stars{color:#ffe94d;font-size:28px}.stars small{color:#c8d3ea;font-size:20px}
.foot{font-size:18px;color:#aab6d0;margin-top:16px}
"""


def main():
    rows = list(csv.DictReader(open(os.path.join(GAME, "levels.csv"))))
    if ONLY:
        rows = [r for r in rows if int(r["level"]) in ONLY]
    doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(card(r) for r in rows)}</body></html>"
    path = os.path.abspath(os.path.join(OUT, "_all.html"))
    open(path, "w").write(doc)
    import json, subprocess
    job = dict(html=path, out=os.path.abspath(OUT), ids=[f"L{int(r['level']):03d}" for r in rows],
               pdf=None)
    subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "shoot.js"), json.dumps(job)], check=True)
    os.remove(path)
    print("rendered", len(rows))


main()
