"""FG-2 for N3g/N3i with the bot's #146 row: ledger 16:00 + bot rows of #146/#147/#148/#151/#152."""
import json, sys
sys.path.insert(0, "/home/administrator/dev/gittensor/bittrellis-hpc02")
from bittrellis.frontier.pareto import Row, frontier_gain, pareto, dominates
F = json.load(open("ledger-hpc02-e1-frontier-20261009T1600.json")); box, fl = F["box"], F["epsilon_floor"]
qc = lambda a, b: -1 if a.rp_kl - b.rp_kl < -fl["rp_kl"] else (1 if a.rp_kl - b.rp_kl > fl["rp_kl"] else 0)
R = lambda n, rp, d, p, g, ds=0.001, ps=0.003: Row(id=n, name=n, kind="internal", rp_kl=rp, decode_tps=d, prefill_tps=p, peak_gpu_gib=g, decode_spread=ds, prefill_spread=ps)
pool = [R(r["name"], r["rp_kl"], r["decode_tps"], r["prefill_tps"], r["peak_gpu_gib"], r["decode_spread"], r["prefill_spread"]) for r in F["internal"] if r["valid"]]
pool += [R("146", .0547, 431.81, 39947, 23.729, .00028, .00065), R("147", .0855, 474.63, 39962.7, 23.401, .00085, .0037),
         R("148", .0598, 447.2, 40163, 23.419, .0006, .0029), R("151", .0512, 420.1, 38416, 26.653), R("152", .0304, 383.6, 36808, 29.466)]
def g(c):
    if c not in pareto(pool + [c], fl, qc): return "dom by " + ",".join(o.name for o in pool if dominates(o, c, fl, qc))
    return f"{frontier_gain(c, pool, box, fl)*100:.3f}%"
k = 0.0547 / 0.0526  # bot / our box on the same recipe C
for n, rp in (("N3g", .0951), ("N3i", .1030)):
    for d in (477.0, 479.4, 479.77, 481.4):
        print(n, f"rp {rp*k:.4f} dec {d}:", g(R(n, rp * k, d, 39836.0, 23.452, .0011, .00005)))
print("== map + GPTQ grids (bot scale)")
L = {r.name: r for r in pool}
for n, src in (("N2 #148map", "148"), ("N1 #147map", "147"), ("C2 #146map", "146"), ("#152map", "152"), ("#151map", "151")):
    s = L[src]; print(n, " | ".join(f"{rp:.4f}: {g(R(n, rp, s.decode_tps, s.prefill_tps, s.peak_gpu_gib, s.decode_spread, s.prefill_spread))}" for rp in [s.rp_kl - d for d in (0.0025, 0.004, 0.006, 0.009)]))
