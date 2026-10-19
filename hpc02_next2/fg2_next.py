"""FG-2 estimates for the next HPC-02 candidates against the hpc02-e1 ledger (08:45 snapshot) plus #147's posted row.

Same formats as a measured row -> same speed and memory; only RP-KL changes. Quality compare uses the RP-KL floor
(the evaluator's paired CI is not modelled).
"""
import json, sys
sys.path.insert(0, "/home/administrator/dev/gittensor/bittrellis-hpc02")
from bittrellis.frontier.pareto import Row, frontier_gain, pareto, dominates
F = json.load(open(__file__.rsplit("/", 1)[0] + "/ledger-hpc02-e1-frontier-20261009T0845.json"))
box, floors = F["box"], F["epsilon_floor"]
qc = lambda a, b: -1 if a.rp_kl - b.rp_kl < -floors["rp_kl"] else (1 if a.rp_kl - b.rp_kl > floors["rp_kl"] else 0)
def R(name, rp, dec, pre, peak, ds=0.001, ps=0.003):
    return Row(id=name, name=name, kind="internal", rp_kl=rp, decode_tps=dec, prefill_tps=pre, peak_gpu_gib=peak,
               decode_spread=ds, prefill_spread=ps)
base = [R(r["name"], r["rp_kl"], r["decode_tps"], r["prefill_tps"], r["peak_gpu_gib"], r["decode_spread"], r["prefill_spread"])
        for r in F["internal"] if r["valid"]]
led = {r.name: r for r in base}
def same(src, name, rp):
    s = led[src] if isinstance(src, str) else src
    return R(name, rp, s.decode_tps, s.prefill_tps, s.peak_gpu_gib, s.decode_spread, s.prefill_spread)
P145, P142, P129 = "pr130-attention-qo-11-39-q4k", "pr129-recurrent-zo-deep-downs-q4k", "ud-downs-q4k-gdn-qkv-q4k-lmhead-embed-q4k"
pr147 = R("147-pr145-kq_imat", 0.0855, 474.63, 39962.7, 23.40, 0.0008, 0.0037)        # posted 10:05, eval:M
C146 = same(P129, "146-C", 0.0526)                                                    # our measured C
# rivals still unmeasured: kq_imat ~ -16% RP-KL on the kq_rtn tensors (#145 -> #147)
pr148 = same(P142, "148-pr142-kq_imat~", 0.0713 * 0.84)
pr153 = R("153-pr142+qo-kq_imat~", (0.0713 + 0.0051) * 0.84, 461.0, 39650, 23.39)    # speed: halfway #142 -> #145
def gain(c, pool):
    rows = pool + [c]
    if c not in pareto(rows, floors, qc):
        return "dominated by " + ",".join(o.name for o in pool if dominates(o, c, floors, qc))
    return f"{frontier_gain(c, pool, box, floors) * 100:.3f}%"
scen = {"ledger+#147": base + [pr147],
        "+#146C": base + [pr147, C146],
        "+#146C+#148~+#153~": base + [pr147, C146, pr148, pr153]}
cands = {"N1 #145map+GPTQ": [same(P145, f"N1@{r}", r) for r in (0.070, 0.074, 0.078, 0.082)],
         "N2 #142map+GPTQ": [same(P142, f"N2@{r}", r) for r in (0.050, 0.054, 0.058)],
         "C (#146)": [same(P129, f"C@{r}", r) for r in (0.0526,)]}
for sn, pool in scen.items():
    print("==", sn)
    for k, lst in cands.items():
        print(" ", k.ljust(18), " | ".join(f"rp {c.rp_kl:.3f}: {gain(c, pool)}" for c in lst))
print("tiers FG-2 %: XL 0.50, L 0.25, M 0.12, S 0.035, XS 0.005")
print("#147 vs ledger:", gain(pr147, base), " C vs ledger+#147:", gain(C146, base + [pr147]))

# the fastest measured map: #144 (attention + recurrent on Q5_K, kq_rtn; RP-KL 0.1098, guard fail 558/784)
P144 = R("144-map", 0.1098, 479.77, 39836.0, 23.452, 0.0011, 0.00005)
print("== #144's map with the GPTQ encoder (needs the guard to pass)")
for sn, pool in (("ledger+#147", base + [pr147]), ("+#146C+#148~+#153~", base + [pr147, C146, pr148, pr153])):
    print(" ", sn.ljust(20), " | ".join(f"rp {r:.3f}: {gain(same(P144, f'N3@{r}', r), pool)}" for r in (0.060, 0.070, 0.080, 0.090)))

# CPU-calibrated estimates (see RESULTS.md 2026-10-09 ~11:00): N3g ~0.0875 (0.075-0.095), N3i ~0.091 (0.083-0.100),
# N1g ~0.0825; #129+kq_imat (anyone can open it) ~0.0533 vs C 0.0526
print("== estimates")
for sn, pool in (("ledger+#147", base + [pr147]), ("+#146C", base + [pr147, C146]),
                 ("+#146C+#148~+#153~", base + [pr147, C146, pr148, pr153])):
    print(" ", sn.ljust(20), " | ".join(f"{n} {r:.4f}: {gain(same(src, n, r), pool)}" for n, src, r in (
        ("N3g", P144, 0.0875), ("N3g-lo", P144, 0.075), ("N3g-hi", P144, 0.095), ("N3i", P144, 0.091), ("N1g", P145, 0.0825))))
x129 = same(P129, "129+kq_imat~", 0.0533)
print("C vs ledger+#147+#148~+#153~:", gain(C146, base + [pr147, pr148, pr153]),
      "| C after a #129+kq_imat row:", gain(C146, base + [pr147, x129]))
