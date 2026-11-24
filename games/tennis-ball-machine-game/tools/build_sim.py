import sys, csv, json, os
GAME = sys.argv[1]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "render.py")).read()
src = src[:src.index("# ---------- card")]
sys.argv = ["x", GAME]
ns = {}; exec(src, ns)
HIT = {61:"VV",62:"V",63:"VV",64:"VV",67:"GV",68:"GV",69:"VV",70:"GVV",78:"VO",94:"GGV"}
out = []
for r in csv.DictReader(open(os.path.join(GAME, "levels.csv"))):
    n = int(r["level"]); rnd, shots = ns["parse"](r)
    hs = HIT.get(n, "")
    js = []
    for i, s in enumerate(shots):
        h = hs[i] if i < len(hs) else ("V" if n in (61,62,63,64,69) else "G")
        if s["riser"] and s["depth"] != "D" or (s["riser"] and "lob" in r["shots"] and s["lane"]=="C" and s["depth"]=="D"):
            h = "O"
        js.append(dict(d=s["depth"], l=s["lane"], kmh=s["speed"], rpm=ns["SPIN"][s["spin"]], delay=s["interval"], riser=s["riser"], hit=h))
    out.append(dict(n=n, world=int(r["world"]), name=r["name"], save=r["proton_name"], shots=js, random=rnd,
                    target=r["target"], balls=int(r["balls"]), stars=[int(r["star1"]), int(r["star2"]), int(r["star3"])], notes=r["notes"]))
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_template.html")).read()
open(os.path.join(GAME, "simulator.html"), "w").write(html.replace("/*__DATA__*/", json.dumps(out, separators=(",", ":"), ensure_ascii=False)))
print(len(out), [(o["n"], [s["hit"] for s in o["shots"]]) for o in out if o["n"] in (61,70,71,75,76,78,80,94)])
