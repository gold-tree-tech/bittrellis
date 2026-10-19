"""Estimate FG-2 of candidate HPC-02 rows against the ledger frontier (simplified RP-KL compare: |d|>floor)."""
import json, sys
sys.path.insert(0, "/home/administrator/dev/gittensor/bittrellis-hpc02")
from bittrellis.frontier.pareto import Row, rank, frontier_gain, pareto
L = "/tmp/claude-1000/-home-administrator-dev-gittensor/bb1bb412-e7c3-4471-a7fb-6da178b471d6/scratchpad/ledger/hpc02-e1/frontier.json"
F = json.load(open(L))
box, floors = F["box"], F["epsilon_floor"]
def qc(a, b):
    d = a.rp_kl - b.rp_kl
    return -1 if d < -floors["rp_kl"] else (1 if d > floors["rp_kl"] else 0)
def R(name, rp, dec, pre, peak, ds=0.001, ps=0.001):
    return Row(id=name, name=name, kind="internal", rp_kl=rp, decode_tps=dec, prefill_tps=pre, peak_gpu_gib=peak,
               decode_spread=ds, prefill_spread=ps)
base = [R(r["name"], r["rp_kl"], r["decode_tps"], r["prefill_tps"], r["peak_gpu_gib"], r["decode_spread"], r["prefill_spread"])
        for r in F["internal"] if r["valid"]]
led = {r.name: r for r in base}
def same(src, name, rp):  # same formats as a measured row -> same speed/memory
    s = led[src]; return R(name, rp, s.decode_tps, s.prefill_tps, s.peak_gpu_gib, s.decode_spread, s.prefill_spread)
vb = same("V2-all-q4k", "146-vb", 0.0857)
pr140 = R("140-kq_xtx", 0.0818, 469.0, 40900, 23.45)
cands = {
  "A #130map+imx": [("pr129-recurrent-zo-q4k", r) for r in (0.070, 0.075, 0.082)],
  "B #142map+imx": [("pr129-recurrent-zo-deep-downs-q4k", r) for r in (0.054, 0.058, 0.063)],
  "C #129map+imx": [("ud-downs-q4k-gdn-qkv-q4k-lmhead-embed-q4k", r) for r in (0.048, 0.051, 0.055)],
  "D #126map+imx": [("ud-downs-q4k-lmhead-embed-q4k", r) for r in (0.044, 0.046, 0.048)],
  "G #133map+imx": [("ud-exps-gate-up-q5k", r) for r in (0.031, 0.033, 0.035)],
}
for scen, pool in (("ledger+vb", base + [vb]), ("ledger+vb+#140", base + [vb, pr140])):
    print("==", scen)
    for k, lst in cands.items():
        out = []
        for src, rp in lst:
            c = same(src, k, rp)
            rows = pool + [c]
            front = pareto(rows, floors, qc)
            on = c in front
            g = frontier_gain(c, pool, box, floors) if on else 0.0
            dom = [o.name for o in pool if (qc(o, c) == -1 or True) and o in rows and o is not c and
                   ( [qc(o,c)] + [0] ) and o not in [] and __import__('bittrellis.frontier.pareto', fromlist=['x']).dominates(o, c, floors, qc)]
            out.append(f"rp {rp:.3f}: FG2 {g*100:.3f}% {'' if on else 'DOMINATED by '+','.join(dom)}")
        print(k.ljust(16), " | ".join(out))
    # vb itself
    p2 = [r for r in pool if r is not vb]
    print("vb (#146)".ljust(16), f"FG2 {frontier_gain(vb, p2, box, floors)*100:.3f}%")
print("tiers (FG-2 %): XL 0.50, L 0.25, M 0.12, S 0.035, XS 0.005")

print("== joint: B(0.059) and C(0.052) both, ledger+vb")
B = same("pr129-recurrent-zo-deep-downs-q4k", "B", 0.059); C = same("ud-downs-q4k-gdn-qkv-q4k-lmhead-embed-q4k", "C", 0.052)
for x, y in ((B, C), (C, B)):
    p = base + [vb, y]
    print(x.name, "after", y.name, f"{frontier_gain(x, p, box, floors)*100:.3f}%")
print("B alone", f"{frontier_gain(B, base+[vb], box, floors)*100:.3f}%", "C alone", f"{frontier_gain(C, base+[vb], box, floors)*100:.3f}%")
print("vb if B merged first", f"{frontier_gain(vb, base+[B], box, floors)*100:.3f}%")
