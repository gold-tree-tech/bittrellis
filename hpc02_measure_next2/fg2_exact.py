import io, contextlib, runpy
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("analyze.py")
R, same, base, pr147, P144, gain = g["R"], g["same"], g["base"], g["pr147"], g["P144"], g["gain"]
from bittrellis.frontier.pareto import frontier_gain
box, floors = g["box"], g["floors"]
for v in ("N3i", "N3g"):
    rp = g["q"][v]["rp_kl"] * 0.0855 / g["k"]
    for dec in (481.4, 479.77, 474.63 * 1.01, 470.0, 463.4, 448.7):
        c = R(v, rp, dec, 39836.0, 23.452, 0.0011, 0.00005)
        print(v, f"rp {rp:.4f} decode {dec:.1f}:", gain(c, base + [pr147]))
    print(v, "exact at #144 ledger speed:", frontier_gain(same(P144, v, rp), base + [pr147], box, floors) * 100)
