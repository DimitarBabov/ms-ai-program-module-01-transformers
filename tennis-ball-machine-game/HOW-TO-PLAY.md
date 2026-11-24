# Ball Machine Quest: How to Play

A 100-level tennis game for the Proton ball machine. There are **10 worlds with 10 levels each**. Every 10th level is a **BOSS** level. You earn stars, unlock levels, collect badges, and work your way up to the **Champion** level (100).

Files in this folder:

| File | What it's for |
|---|---|
| `HOW-TO-PLAY.md` | This file: the rules, the court map and the scoring |
| `LEVELS.md` | All 100 levels with the machine settings for each one |
| `level-cards/` + `Level-Cards.pdf` | **A picture card for every level**, drawn like the Proton app, with every setting to enter |
| `levels.csv` | The same 100 levels as a spreadsheet (handy while programming the machine) |
| `progress-tracker.csv` | One row per level: best score, stars, date passed |
| `session-log.csv` | One row per attempt: the full practice diary |

The `.csv` files open in Excel, Google Sheets or Numbers. You can also print them.

---

## 1. The court map

The machine sits in the **center of the far baseline**. Every level tells you two things:

- **Shots:** where the machine sends the ball (on the player's side)
- **Target:** where the player has to hit it (on the machine's side)

### Shot spots (player's side): where the machine sends the ball

There are 15 spots: 3 rows × 5 columns, exactly like the 15 circles in the Proton app. The level cards show the court **the same way the app does**: machine at the bottom, player at the top.

- **D** = back row (top row of circles in the app)
- **M** = middle row (2nd row)
- **S** = front row (the row just over the net)
- Columns are counted **from the left of the app screen**: 1st = FW, 2nd = F, 3rd = C, 4th = B, 5th = BW. The app view is from behind the machine, so the **left of the screen is a right-hander's forehand side**.

```
                    ======== NET ========
           +-------+-------+-------+-------+-------+
  S short  | S-BW  |  S-B  |  S-C  |  S-F  | S-FW  |   around the service line
           +-------+-------+-------+-------+-------+
  M middle | M-BW  |  M-B  |  M-C  |  M-F  | M-FW  |   between service line and baseline
           +-------+-------+-------+-------+-------+
  D deep   | D-BW  |  D-B  |  D-C  |  D-F  | D-FW  |   close to the baseline
           +-------+-------+-------+-------+-------+
             wide    back-   center  fore-   wide
             back-   hand            hand    fore-
             hand                            hand
                  ====== BASELINE (player) ======
                              X  <- center mark
```

- **Lanes** (shown for a right-hander): BW = wide backhand, B = backhand, C = center/body, F = forehand, FW = wide forehand.
  **If he's left-handed, tap Mirror** in the app on each program.
- `->` in a level means the machine plays the shots in that order (**Order: in order**) and repeats.
- `RANDOM:` means you add those shots to one program and set **Order** to random.
- `(lob)` or `(high)` means put the **lob riser** on (Riser ON) for that shot.

### Target zones (machine's side): where his ball must land

```
                  ====== BASELINE (machine) ======
                              [M]  <- machine
           +-----------+-------------------+-----------+
           |  Box  E/F |                   |  Box  E/F |   corner boxes 3 m x 3 m
           |  (cones)  |     Box G (3 m)   |  (cones)  |   (mark with 4 cones each)
           +-----------+   middle deep box +-----------+
           |                                           |
           |        Zone B = everything from the       |
           |        service line to the baseline       |
           |                                           |
           +=====================+=====================+   <- service line
           |                     |                     |
           |                     |                     |
           |                     |                     |
                    ======== NET ========
```

| Target | Name | The ball counts if it lands... |
|---|---|---|
| **A** | Whole court | anywhere in the **singles court** |
| **B** | Deep zone | past the **service line** (between service line and baseline) |
| **C** | Crosscourt | in the **diagonal half** of the singles court (full length) |
| **D** | Down the line | in the **straight-ahead half** of the singles court (full length) |
| **E** | Crosscourt box | in the **diagonal** corner box (3 m × 3 m, marked with cones) |
| **F** | Down-the-line box | in the **straight-ahead** corner box |
| **G** | Middle box | in a 3 m wide box in the middle of zone B |

"Diagonal" and "straight-ahead" are from where he hits the ball. A forehand from the forehand side goes crosscourt into the box on the other side. Set up both corner boxes once, and they work for E and F.

Two targets separated by a slash (like `C / D`) means odd-numbered balls go to the first target and even-numbered balls to the second. The level's notes explain it.

You need about **8–12 cones** or throw-down lines. Tape or chalk works too.

---

## 2. Machine settings

The Proton app shows speed and spin in its own numbers. Do a **calibration once**, write the numbers into the table below, and then every level just says S1, Top 2, and so on.

### Speed steps

| Step | Proton app speed | Meaning |
|---|---|---|
| **S1** | **60 km/h** (lowest) | Easy and comfortable |
| **S2** | **70 km/h** | A bit quicker |
| **S3** | **80 km/h** | Normal rally pace |
| **S4** | **90 km/h** | Match pace for a good junior |

If a step is too hard or too easy for him, move all levels that use it by one slider notch, and write the change here so you stay consistent.

### Spin

| Name | Proton app spin | Meaning |
|---|---|---|
| **Flat** | 0 rpm | no spin |
| **Top 1** | +500 rpm topspin | light topspin |
| **Top 2** | +1000 rpm topspin | medium, bounces up a bit |
| **Top 3** | +2000 rpm topspin | heavy, high kick |
| **Slice 1** | −500 rpm backspin | stays low |

"Mixed (per shot)" means each shot has its own spin. The level card lists the spin for every shot.

### Delay

The **Delay** on each shot is the number of seconds before that ball, the same as the app's Delay slider.

### Programming the machine

Open the level's card (`level-cards/L007.png`, or that page of `Level-Cards.pdf`) and copy it into the app:

1. **Programs → Custom →** create a new program named like the card (`L007`).
2. For each shot on the card: **Add New Shot**, tap the circle shown on the card, then set **Spin**, **Speed** and **Delay** as listed. Turn **Riser** on if the card says so. Then tap **Save Shot**.
3. Set **Loops** and **Order** as shown on the card's dark bar.
4. Left-handed? Tap **Mirror**.

Proton location stays **baseline center**, Stand Off. The card's **Balls** number is how many balls count. If loops × shots goes over that, stop the machine after that many balls (the card says so).

You don't need to do all 100 at once. **Program one world (10 levels) at a time**, and do the next world when he gets there.

---

## 3. Playing a level

1. Put out the cones for that level's target.
2. Start the level's sequence. The ball count is **10 balls** for a normal level, **20** for a boss and **30** for the final boss. Load enough balls, or stop the machine after the count.
3. **The scorer** (a parent, or the player calling it honestly) counts balls that land **in the target**.
4. Look up the stars in the level's row and write the result in the log.

### What counts as "in"

- The ball has to land **in the target on the first bounce**. A ball touching the line counts as **in**.
- A net cord that rolls over and lands in the target is **in**.
- A miss, a frame shot that goes out, or a ball he didn't get to is **out**.
- On **volley** levels the ball must be hit **before it bounces**, or it's out.
- On levels with **"Recover to center mark"**, he must get back to the center mark (or close to it) before the next ball is hit. If he doesn't, the shot is out even if it lands in.

### Stars

| Level type | Balls | ★ | ★★ | ★★★ |
|---|---|---|---|---|
| Normal, target A (whole court) | 10 | 7 | 8 | 9+ |
| Normal, targets B, C, D (half court or deep) | 10 | 6 | 7 | 8+ |
| Normal, targets E, F, G (small boxes) | 10 | 4 | 5 | 6+ |
| BOSS | 20 | ×2 | ×2 | ×2 |
| FINAL BOSS (level 100) | 30 | ×3 | ×3 | ×3 |

The exact numbers for each level are also in `LEVELS.md`. The marks are set so a level feels **hard but doable**: he should pass roughly 6–8 balls out of 10. That's where kids learn fastest.

### Unlocking

- **1 star passes a level** and unlocks the next one.
- A **BOSS** level unlocks only when he has **15 or more stars** across levels 1–9 of that world (the maximum is 27). That means he has to go back and improve some levels, not just scrape through.
- **Beating the BOSS** (1 star or more) opens the next world.
- He can **replay any unlocked level** for more stars. Only his best score counts.

### Fair-play rules (no burnout)

- **3 tries max** per level per session. If he fails three times, play the previous level once as a "confidence round" and move on to something else. Try the hard one again next session.
- On a **new world**, it's fine to lower the speed one step for the first try. A pass at the lower speed doesn't count, though: it's just practice.
- **Technique beats score.** If he starts slapping or pushing the ball just to get it in, call a "redo" and play the level again.

---

## 4. Badges

Write the badges in the session log when he earns them. A sticker chart on the fridge works well too.

| Badge | How to earn it |
|---|---|
| ⭐ **First Star** | earn your first star |
| 💯 **Perfect** | every ball in on any level (10/10, 20/20 …) |
| 🔥 **Hot Streak** | 7 balls in a row in the target during one level |
| 💪 **Comeback Kid** | pass a level after failing it twice |
| 👑 **Boss Slayer** | beat a BOSS (one badge per world: 10 to collect) |
| 🌟 **Three-Star World** | 3 stars on every level of one world |
| 🦵 **Iron Legs** | beat World 4 |
| 🌀 **Spin Doctor** | beat World 5 |
| 🎯 **Sniper** | 3 stars on any corner-box level (E or F) |
| 🥷 **Net Ninja** | beat World 7 |
| ☁️ **Sky Walker** | beat World 8 |
| 🎲 **Chaos Master** | beat World 9 |
| 📅 **Iron Habit** | 3 sessions in one week |
| 🏆 **Champion** | beat level 100 |

### Real-world rewards (agree on these with him before you start)

There are **300 stars** in total. Some ideas:

| Total stars | Reward idea |
|---|---|
| 25 | choose dinner |
| 50 | new overgrip in his favourite colour |
| 100 | trip to watch a match, or a new tennis shirt |
| 150 | he picks the weekend activity |
| 200 | new tennis shoes, or a racket bag |
| 250 | a special day out |
| 300 / Champion | something big: a camp, or a new racket |

---

## 5. A good session (45–60 minutes)

1. **Warm-up (10 min):** jog, side shuffles, arm circles, shadow swings, then 20 easy balls at S1 with no scoring.
2. **Replay (10 min):** one old level to beat his best score. Starting with a win feels great.
3. **New levels (20–30 min):** 2–4 new levels, with a short water break between them.
4. **Fun finish (5 min):** his choice of any unlocked level, or a "beat dad/mom" round on the same level.
5. **Write it down:** fill in `session-log.csv` and update `progress-tracker.csv`.

Guidelines for a 10-year-old:

- About **150–250 balls** per session is plenty. Quality first.
- **Stop** if anything hurts (wrist, elbow, shoulder). Overhead levels (World 8) need a good shoulder warm-up.
- **2–3 sessions a week** are better than one very long one.
- Bring water, and take shade breaks when it's hot.

---

## 6. Recording progress

**`session-log.csv`** is the practice diary. Add one row per attempt:

| Date | Level | Attempt # | Balls in target | Balls played | Stars | Best streak | Badges earned | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-29 | 1 | 1 | 8 | 10 | 2 | 5 | First Star | backhand late on contact |

**`progress-tracker.csv`** has all 100 levels already listed. Fill in **Best score**, **Best stars**, **Attempts** and **Date first passed**, and put a **Y** in the last column once that level is saved in the Proton app.

Add up the **Best stars** column for his total. That's the number for the rewards table.

Delete the example row in `session-log.csv` before you start.

---

## 7. The 10 worlds at a glance

| World | Levels | Name | What he learns |
|---|---|---|---|
| 1 | 1–10 | Rally Rookie | Clean forehands and backhands, slow balls straight to him |
| 2 | 11–20 | Forehand & Backhand | More speed, less time, patterns, the ball at his body |
| 3 | 21–30 | Deep Zone | Depth: hitting past the service line |
| 4 | 31–40 | Fast Feet | Wide and short balls, recovering to the center |
| 5 | 41–50 | Spin Lab | Handling topspin and slice |
| 6 | 51–60 | Aim Master | Crosscourt, down the line, corner boxes |
| 7 | 61–70 | Net Attack | Volleys, approach shots, moving forward |
| 8 | 71–80 | Sky Balls | Overheads and high balls (lob riser) |
| 9 | 81–90 | Chaos Mode | Random balls: reading and reacting |
| 10 | 91–100 | Match Play | Match speed, small targets, building points |

Have fun, and good luck on the way to Champion! 🎾
