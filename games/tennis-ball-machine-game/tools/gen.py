import csv, os, sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

WORLDS = {
    1: ("Rally Rookie", "Slow balls straight to you. Just get it in the court."),
    2: ("Forehand & Backhand", "A bit faster, a bit quicker. Patterns of forehands and backhands."),
    3: ("Deep Zone", "Push the ball deep, past the service line."),
    4: ("Fast Feet", "Wider and shorter balls. Recover to the center mark after every shot."),
    5: ("Spin Lab", "Handle topspin that kicks up and slice that stays low."),
    6: ("Aim Master", "Hit crosscourt and down the line, then into the corner boxes."),
    7: ("Net Attack", "Volleys, approach shots, and moving forward to the net."),
    8: ("Sky Balls", "Overheads and high balls. Use the lob riser."),
    9: ("Chaos Mode", "RANDOM order turned on. You don't know where the next ball is going."),
    10: ("Match Play", "Faster, heavier, smaller targets. Play like it's a real match."),
}

# (name, shots, speed, spin, interval_s, target, rule)
# Shots use the court grid: Depth S/M/D + Lane BW/B/C/F/FW. "->" = fixed order, "RANDOM:" = randomize on.
L = [
 # World 1
 ("Forehand Start", "M-F", "S1", "Flat", 6, "A", "Forehands only"),
 ("Backhand Start", "M-B", "S1", "Flat", 6, "A", "Backhands only"),
 ("Ping-Pong", "M-F -> M-B", "S1", "Flat", 6, "A", "Alternate forehand / backhand"),
 ("Forehand Spin-Up", "M-F", "S2", "Top 1", 5, "A", ""),
 ("Backhand Spin-Up", "M-B", "S2", "Top 1", 5, "A", ""),
 ("Ping-Pong 2", "M-F -> M-B", "S2", "Top 1", 5, "A", ""),
 ("Deep Forehand", "D-F", "S2", "Top 1", 5, "A", "Ball bounces deeper: start 1 step behind baseline"),
 ("Deep Backhand", "D-B", "S2", "Top 1", 5, "A", "Start 1 step behind baseline"),
 ("Deep Ping-Pong", "D-F -> D-B", "S2", "Top 1", 5, "A", ""),
 ("BOSS: Rookie Rally", "M-F -> M-B -> D-F -> D-B", "S2", "Top 1", 5, "A", "20 balls"),
 # World 2
 ("Faster Forehand", "M-F", "S3", "Top 1", 5, "A", ""),
 ("Faster Backhand", "M-B", "S3", "Top 1", 5, "A", ""),
 ("Fast Ping-Pong", "M-F -> M-B", "S3", "Top 1", 5, "A", ""),
 ("Two Forehands", "M-F -> M-F -> M-B", "S3", "Top 1", 5, "A", ""),
 ("Two Backhands", "M-B -> M-B -> M-F", "S3", "Top 1", 5, "A", ""),
 ("Quick Ping-Pong", "M-F -> M-B", "S3", "Top 1", 4, "A", "Less time between balls"),
 ("Body Ball", "M-C", "S3", "Top 1", 5, "A", "Ball comes at you: step away, pick forehand or backhand"),
 ("Three Spots", "M-F -> M-C -> M-B", "S3", "Top 1", 4, "A", ""),
 ("Speed Step", "M-F -> M-B", "S4", "Top 1", 5, "A", ""),
 ("BOSS: Pattern Pro", "M-F -> M-B -> M-F -> M-C -> M-B", "S3", "Top 1", 4, "A", "20 balls"),
 # World 3
 ("Push Deep Forehand", "M-F", "S2", "Top 1", 5, "B", "Swing low to high, aim high over the net"),
 ("Push Deep Backhand", "M-B", "S2", "Top 1", 5, "B", ""),
 ("Push Deep Ping-Pong", "M-F -> M-B", "S2", "Top 1", 5, "B", ""),
 ("Deep Forehand 2", "M-F", "S3", "Top 1", 5, "B", ""),
 ("Deep Backhand 2", "M-B", "S3", "Top 1", 5, "B", ""),
 ("Deep Ping-Pong 2", "M-F -> M-B", "S3", "Top 1", 5, "B", ""),
 ("Middle Box Forehand", "M-F", "S3", "Top 1", 5, "G", ""),
 ("Middle Box Backhand", "M-B", "S3", "Top 1", 5, "G", ""),
 ("Middle Box Ping-Pong", "M-F -> M-B", "S3", "Top 1", 4.5, "G", ""),
 ("BOSS: Deep Diver", "D-F -> D-B", "S3", "Top 2", 4.5, "B", "20 balls"),
 # World 4
 ("Wide Forehand", "M-FW", "S2", "Top 1", 6, "A", "Recover to center mark after every shot"),
 ("Wide Backhand", "M-BW", "S2", "Top 1", 6, "A", "Recover to center mark"),
 ("Side to Side", "M-FW -> M-BW", "S2", "Top 1", 6, "A", "Recover to center mark"),
 ("Deep Wide Forehand", "D-FW", "S3", "Top 1", 5.5, "A", "Recover to center mark"),
 ("Deep Wide Backhand", "D-BW", "S3", "Top 1", 5.5, "A", "Recover to center mark"),
 ("Big Side to Side", "D-FW -> D-BW", "S3", "Top 1", 5.5, "A", "Recover to center mark"),
 ("In and Out Forehand", "S-F -> D-F", "S2", "Top 1", 6, "A", "Run in for the short ball, back for the deep one"),
 ("In and Out Backhand", "S-B -> D-B", "S2", "Top 1", 6, "A", "Run in, run back"),
 ("Four Corners", "M-FW -> S-C -> M-BW -> D-C", "S3", "Top 1", 5.5, "A", "Recover to center mark"),
 ("BOSS: Iron Legs", "M-FW -> M-BW", "S3", "Top 1", 5, "B", "20 balls. Recover to center mark"),
 # World 5
 ("Topspin Forehand", "M-F", "S3", "Top 2", 5, "A", "Ball kicks up: move back, hit it at waist height"),
 ("Topspin Backhand", "M-B", "S3", "Top 2", 5, "A", ""),
 ("Topspin Deep", "M-F -> M-B", "S3", "Top 2", 5, "B", ""),
 ("Slice Forehand", "M-F", "S3", "Slice 1", 5, "A", "Ball stays low: bend your knees"),
 ("Slice Backhand", "M-B", "S3", "Slice 1", 5, "A", "Bend your knees"),
 ("Slice Ping-Pong", "M-F -> M-B", "S3", "Slice 1", 5, "A", ""),
 ("Heavy Forehand", "D-F", "S3", "Top 3", 5, "A", "High bouncer: step back or take it early"),
 ("Heavy Backhand", "D-B", "S3", "Top 3", 5, "A", ""),
 ("Spin Switch", "M-F (Top 2) -> M-B (Slice 1) -> M-F (Slice 1) -> M-B (Top 2)", "S3", "Mixed (per shot)", 5, "B", ""),
 ("BOSS: Spin Doctor", "M-F (Top 3) -> M-B (Slice 1) -> D-F (Slice 1) -> D-B (Top 2)", "S3", "Mixed (per shot)", 5, "B", "20 balls"),
 # World 6
 ("Crosscourt Forehand", "M-F", "S3", "Top 1", 5, "C", ""),
 ("Crosscourt Backhand", "M-B", "S3", "Top 1", 5, "C", ""),
 ("Down-the-Line Forehand", "M-F", "S3", "Top 1", 5, "D", "Hit later, point the racket straight"),
 ("Down-the-Line Backhand", "M-B", "S3", "Top 1", 5, "D", ""),
 ("Crosscourt Ping-Pong", "M-F -> M-B", "S3", "Top 1", 5, "C", "Both sides crosscourt"),
 ("Switch It", "M-F", "S3", "Top 1", 5, "C / D", "Odd balls crosscourt (C), even balls down the line (D)"),
 ("Corner Box Forehand", "M-F", "S3", "Top 1", 5, "E", ""),
 ("Corner Box Backhand", "M-B", "S3", "Top 1", 5, "E", ""),
 ("Corner Box Ping-Pong", "M-F -> M-B", "S3", "Top 1", 5, "E", ""),
 ("BOSS: Sniper", "M-F -> M-B", "S3", "Top 2", 4.5, "E / D", "20 balls. Odd balls crosscourt box (E), even balls down the line (D)"),
 # World 7
 ("Forehand Volley", "S-F", "S2", "Flat", 6, "A", "Stand 2 m from the net. Catch the ball in the air"),
 ("Backhand Volley", "S-B", "S2", "Flat", 6, "A", "Stand 2 m from the net"),
 ("Volley Ping-Pong", "S-F -> S-B", "S2", "Flat", 6, "A", ""),
 ("Deep Volleys", "S-F -> S-B", "S3", "Flat", 5, "B", ""),
 ("Forehand Approach", "S-F", "S2", "Top 1", 6, "B", "Start at baseline, run in, hit, keep going to the net"),
 ("Backhand Approach", "S-B", "S2", "Slice 1", 6, "B", "Start at baseline, run in"),
 ("Approach + Volley", "S-F -> S-B", "S2", "Flat", 6, "A", "Ball 1: forehand approach from baseline. Ball 2: backhand volley"),
 ("Approach + Volley 2", "S-B -> S-F", "S2", "Flat", 6, "A", "Ball 1: backhand approach. Ball 2: forehand volley"),
 ("Low Volleys", "S-F -> S-B", "S2", "Slice 1", 6, "A", "Bend your knees, don't swing, punch"),
 ("BOSS: Net Ninja", "S-C -> S-F -> S-B", "S2", "Flat", 6, "A", "20 balls. Approach, then 2 volleys. Start each pattern at baseline"),
 # World 8
 ("First Overhead", "M-C (lob)", "S2", "Flat", 8, "A", "Lob riser on. Stand at service line. Point at the ball with your free hand"),
 ("Forehand-Side Overhead", "M-F (lob)", "S2", "Flat", 8, "A", "Lob riser on"),
 ("Moving Overhead", "M-F (lob) -> M-B (lob)", "S2", "Flat", 8, "A", "Turn sideways and shuffle"),
 ("Deep Overhead", "M-C (lob)", "S2", "Flat", 8, "B", ""),
 ("Backward Overhead", "D-C (lob)", "S2", "Flat", 8, "A", "Start at the net, turn and run back"),
 ("Moonball Forehand", "D-F (high)", "S2", "Top 2", 6, "B", "Lob riser on. Baseline. Hit the high ball back deep"),
 ("Moonball Backhand", "D-B (high)", "S2", "Top 2", 6, "B", "Baseline"),
 ("Volley + Overhead", "S-F -> M-C (lob)", "S2", "Flat", 8, "A", "Volley, then step back for the overhead"),
 ("Crosscourt Overhead", "M-C (lob)", "S2", "Flat", 8, "C", ""),
 ("BOSS: Sky Walker", "M-C (lob) -> D-F (high) -> M-F (lob) -> D-B (high)", "S2", "Mixed", 8, "A", "20 balls"),
 # World 9
 ("Guess It", "RANDOM: M-F, M-B", "S3", "Top 1", 5, "A", "Randomize ON"),
 ("Guess It Deep", "RANDOM: M-F, M-B", "S3", "Top 1", 5, "B", "Randomize ON"),
 ("Three-Way Guess", "RANDOM: M-F, M-C, M-B", "S3", "Top 1", 5, "B", "Randomize ON"),
 ("Short or Long FH", "RANDOM: S-F, M-F, D-F", "S3", "Top 1", 5, "A", "Randomize ON. Forehands only"),
 ("Short or Long BH", "RANDOM: S-B, M-B, D-B", "S3", "Top 1", 5, "A", "Randomize ON. Backhands only"),
 ("Four-Way Guess", "RANDOM: M-FW, M-BW, D-F, D-B", "S3", "Top 1", 5, "B", "Randomize ON. Recover to center mark"),
 ("Spin Guess", "RANDOM: M-F (Top 1), M-F (Top 3), M-F (Slice 1), M-B (Top 1), M-B (Top 3), M-B (Slice 1)", "S3", "Mixed (per shot)", 5, "B", "Randomize ON"),
 ("Six-Way Guess", "RANDOM: M-FW, M-F, M-C, M-B, M-BW, D-C", "S3", "Top 1", 5, "A", "Randomize ON"),
 ("Crosscourt Chaos", "RANDOM: M-F, M-B", "S3", "Top 2", 5, "C", "Randomize ON. Every ball crosscourt"),
 ("BOSS: Chaos Master", "RANDOM: M-FW, M-F, M-B, M-BW, D-F, D-B", "S3", "Mixed (per shot)", 4.5, "B", "Randomize ON. 20 balls"),
 # World 10
 ("Match Speed", "M-F -> M-B", "S4", "Top 2", 4.5, "B", ""),
 ("Match Speed Box", "M-F -> M-B", "S4", "Top 2", 4.5, "G", ""),
 ("Crosscourt Battle", "RANDOM: M-F, M-B", "S4", "Top 2", 4.5, "E", "Randomize ON"),
 ("Point Builder", "M-F -> S-C -> S-B", "S3", "Top 1", 5, "B / A", "Rally ball (B), attack the short ball (B), finish with a volley (A)"),
 ("Defend!", "D-FW -> D-BW", "S4", "Top 2", 5, "B", "Recover to center mark. Hit high and deep to get back in the point"),
 ("Attack!", "RANDOM: S-F, S-B", "S3", "Top 1", 5, "E / F", "Randomize ON. Short ball: hit into either corner box"),
 ("Match Chaos", "RANDOM: M-FW, M-F, M-C, M-B, M-BW, D-C", "S4", "Top 2", 4.5, "B", "Randomize ON"),
 ("Heavy Match", "RANDOM: D-F, D-B", "S4", "Top 3", 5, "B", "Randomize ON"),
 ("Everything", "RANDOM: all 15 spots", "S4", "Mixed (per shot)", 4.5, "A", "Randomize ON"),
 ("FINAL BOSS: Champion", "RANDOM: M-FW, M-F, M-C, M-B, M-BW, D-F, D-C, D-B, S-C", "S4", "Mixed (per shot)", 4.5, "B", "Randomize ON. 30 balls"),
]
assert len(L) == 100, len(L)

# pass marks: (balls, 1-star, 2-star, 3-star) by target difficulty
def marks(n, target, rule):
    balls = 30 if n == 100 else (20 if n % 10 == 0 else 10)
    t = target.replace(" ", "")
    if any(x in t for x in "EFG"):
        base = (4, 5, 6)      # small boxes
    elif t == "A":
        base = (7, 8, 9)
    else:
        base = (6, 7, 8)      # half court / deep zone
    k = balls // 10
    return balls, tuple(b * k for b in base)

SPEED = {"S1": 60, "S2": 70, "S3": 80, "S4": 90}
SPINV = {"Flat": "0", "Top 1": "+500", "Top 2": "+1000", "Top 3": "+2000", "Slice 1": "-500"}
rows = []
for i, (name, shots, speed, spin, interval, target, rule) in enumerate(L, 1):
    world = (i - 1) // 10 + 1
    balls, (s1, s2, s3) = marks(i, target, rule)
    rows.append(dict(level=i, world=world, world_name=WORLDS[world][0], name=name,
                     proton_name=f"L{i:03d}", shots=shots, speed=speed, spin=spin,
                     interval=interval, speed_kmh=SPEED[speed], spin_rpm=SPINV.get(spin, 'per shot'), target=target, balls=balls,
                     star1=s1, star2=s2, star3=s3, notes=rule))

# LEVELS.md
md = ["# All 100 Levels", "",
      "How to read a row: see **HOW-TO-PLAY.md** for the court grid (shot spots), speed and spin values and target zones.", "",
      "- **Save as**: the name to give the sequence in the Proton Control app (L001 … L100).",
      "- **Shots**: `->` means play in this order and repeat. `RANDOM:` means put these shots in one sequence and turn randomize on.",
      "- **Card**: a picture of the level drawn like the Proton app, with every setting on it (all cards are also in `Level-Cards.pdf`).",
      "- **Stars**: balls that must land in the target for ★ / ★★ / ★★★.", ""]
for w in range(1, 11):
    wn, wd = WORLDS[w]
    md += [f"## World {w}: {wn}", "", f"_{wd}_", "",
           "| Lvl | Save as | Name | Shots | Speed | Spin | Delay (s) | Target | Balls | ★ / ★★ / ★★★ | Notes | Card |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows[(w - 1) * 10: w * 10]:
        md.append(f"| {r['level']} | {r['proton_name']} | {'**'+r['name']+'**' if r['level'] % 10 == 0 else r['name']} | {r['shots']} | {r['speed_kmh']} km/h | {r['spin']}{'' if r['spin_rpm']=='per shot' else ' (' + r['spin_rpm'] + ' rpm)'} | {r['interval']} | {r['target']} | {r['balls']} | {r['star1']} / {r['star2']} / {r['star3']} | {r['notes']} | [card](level-cards/{r['proton_name']}.png) |")
    md.append("")
open(os.path.join(OUT, "LEVELS.md"), "w").write("\n".join(md))

# levels.csv (setup sheet)
with open(os.path.join(OUT, "levels.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# progress-tracker.csv
with open(os.path.join(OUT, "progress-tracker.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Level", "World", "Name", "Target", "Balls", "1 star", "2 stars", "3 stars",
                "Best score", "Best stars", "Attempts", "Date first passed", "Saved in Proton app (Y/N)"])
    for r in rows:
        w.writerow([r["level"], r["world"], r["name"], r["target"], r["balls"], r["star1"], r["star2"], r["star3"],
                    "", "", "", "", ""])

# session-log.csv
with open(os.path.join(OUT, "session-log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date", "Level", "Attempt #", "Balls in target", "Balls played", "Stars", "Best streak", "Badges earned", "Notes (how it felt, what to fix)"])
    w.writerow(["2026-11-29", 1, 1, 8, 10, 2, 5, "First Star", "Example row - delete me"])
print("ok")
