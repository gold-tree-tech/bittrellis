"""RP-KL (paired vs K147), task guard vs V0 and FG-2 vs the hpc02-e1 ledger + #147 for the next2 run."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/administrator/dev/gittensor/bittrellis-hpc02-next2")
from bittrellis.eval.logits import paired_delta
from bittrellis.frontier.pareto import Row, frontier_gain, pareto, dominates, task_guard
import yaml

ART = Path(__file__).parent / "art"
V0 = json.load(open("/home/administrator/dev/gittensor/bittrellis-hpc02-next2/results/hpc02/artifacts/V0-unsloth-ud/tasks.json"))
cfg = yaml.safe_load(open("/home/administrator/dev/gittensor/bittrellis-hpc02-next2/configs/hpc02.yaml"))
gcfg = cfg["gates"]["task_guard"] if "gates" in cfg else cfg["task_guard"]
F = json.load(open("/tmp/btledger/hpc02-e1/frontier.json"))
box, floors = F["box"], F["epsilon_floor"]
qc = lambda a, b: -1 if a.rp_kl - b.rp_kl < -floors["rp_kl"] else (1 if a.rp_kl - b.rp_kl > floors["rp_kl"] else 0)
def R(name, rp, dec, pre, peak, ds=0.001, ps=0.003):
    return Row(id=name, name=name, kind="internal", rp_kl=rp, decode_tps=dec, prefill_tps=pre, peak_gpu_gib=peak,
               decode_spread=ds, prefill_spread=ps)
base = [R(r["name"], r["rp_kl"], r["decode_tps"], r["prefill_tps"], r["peak_gpu_gib"], r["decode_spread"], r["prefill_spread"])
        for r in F["internal"] if r["valid"]]
led = {r.name: r for r in base}
def same(s, name, rp):
    s = led[s] if isinstance(s, str) else s
    return R(name, rp, s.decode_tps, s.prefill_tps, s.peak_gpu_gib, s.decode_spread, s.prefill_spread)
P145, P142, P129 = "pr130-attention-qo-11-39-q4k", "pr129-recurrent-zo-deep-downs-q4k", "ud-downs-q4k-gdn-qkv-q4k-lmhead-embed-q4k"
P144 = R("144-map", 0.1098, 479.77, 39836.0, 23.452, 0.0011, 0.00005)
pr147 = R("147", 0.0855, 474.63, 39962.7, 23.40, 0.0008, 0.0037)
C146 = same(P129, "146-C", 0.0526)
pr148 = same(P142, "148~", 0.0713 * 0.84)
pr153 = R("153~", (0.0713 + 0.0051) * 0.84, 461.0, 39650, 23.39)
def gain(c, pool):
    if c not in pareto(pool + [c], floors, qc):
        return "dominated by " + ",".join(o.name for o in pool if dominates(o, c, floors, qc))
    return f"{frontier_gain(c, pool, box, floors) * 100:.3f}%"

q = {v: json.load(open(ART / v / "quality.json")) for v in ("K147", "N3i", "N3g", "N1g") if (ART / v / "quality.json").exists()}
pos = {v: dict(np.load(ART / v / "kl_positions.npz")) for v in q if (ART / v / "kl_positions.npz").exists()}
k = q["K147"]["rp_kl"]
src = {"N3i": P144, "N3g": P144, "N1g": P145}
for v in q:
    t = json.load(open(ART / v / "tasks.json"))
    fails = task_guard(t, V0, gcfg)
    lost = sum(1 for s, it in V0["items"].items() for qq, ok in it.items() if ok and not t["items"][s][qq])
    gained = sum(1 for s, it in V0["items"].items() for qq, ok in it.items() if not ok and t["items"][s][qq])
    led_rp = q[v]["rp_kl"] * 0.0855 / k
    line = f"{v}: rp {q[v]['rp_kl']:.4f} -> ledger {led_rp:.4f} | tasks {t['passed']}/784 (-{lost}/+{gained}) guard {'PASS' if not fails else 'FAIL ' + '; '.join(fails)}"
    if v != "K147" and v in pos and "K147" in pos:
        d = paired_delta(pos["K147"], pos[v])
        line += f" | paired vs K147 {d['delta']:+.4f} CI [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]"
    p = json.load(open(ART / v / "performance.json"))
    line += f" | perf {json.dumps({kk: p.get(kk) for kk in ('decode_tps', 'prefill_tps', 'peak_gpu_gib') if kk in p})}"
    print(line)
    if v in src:
        c = same(src[v], v, led_rp)
        print("   FG-2 vs ledger+#147:", gain(c, base + [pr147]), "| +#146C:", gain(c, base + [pr147, C146]),
              "| +#146C+#148~+#153~:", gain(c, base + [pr147, C146, pr148, pr153]))

# speed as measured on this box, scaled by K147's box-to-ledger ratio (K147 = #147's posted speed)
pk = json.load(open(ART / "K147" / "performance.json"))
for v in [x for x in q if x != "K147"]:
    p = json.load(open(ART / v / "performance.json"))
    led_rp = q[v]["rp_kl"] * 0.0855 / k
    c = R(v + "@box", led_rp, p["decode_tps"] * 474.63 / pk["decode_tps"], p["prefill_tps"] * 39962.7 / pk["prefill_tps"],
          p["peak_gpu_gib"], p.get("decode_spread", 0.001), p.get("prefill_spread", 0.003))
    print(f"   {v} with box speed {c.decode_tps:.1f} / {c.prefill_tps:.0f}: FG-2 vs ledger+#147:", gain(c, base + [pr147]),
          "| +#146C+#148~+#153~:", gain(c, base + [pr147, C146, pr148, pr153]))
