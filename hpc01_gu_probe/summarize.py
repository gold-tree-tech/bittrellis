import json, glob, numpy as np
rows = {}
for f in sorted(glob.glob("log_L*.txt")):
    for line in open(f):
        if "proj" not in line: continue
        L, p, js = line.split(" ", 2); d = json.loads(js)
        for k, v in d.items():
            if k.startswith("_"): continue
            rows.setdefault(k, {})[f"L{L}.{p[:2]}"] = v["rel"]; rows[k].setdefault("_s", []).append(v["s"])
cols = sorted({c for r in rows.values() for c in r if c != "_s"}, key=lambda c: (int(c[1:].split(".")[0]), c))
print("variant".ljust(22), " ".join(c.rjust(7) for c in cols), "  mean  cpu_s")
for k, r in rows.items():
    v = [r[c] for c in cols if c in r]
    print(k.ljust(22), " ".join(f"{r.get(c, float('nan')):7.4f}" for c in cols), f"{np.mean(v):.4f} {np.mean(r['_s']):5.0f}")
