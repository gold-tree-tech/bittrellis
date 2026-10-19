"""HPC-01 gate/up encoder variants vs nvfp4_gptq on pinned H; metric tr(E H E^T)."""
import sys, time, json, numpy as np
sys.path.insert(0, "/tmp/bt_up")
import bittrellis.quantizers.nvfp4_gptq as g
from bittrellis.quantizers import nvfp4_blockfit as bf
from bittrellis.quant.formats import e4m3_encode, FP4_E2M1_MAX
P, E4, LV, E2 = g._e2m1_index, g._E4M3_POS, g._LEVELS, g._E2M1
OFF = np.arange(-3, 8)

def scales_wt(w16, ws2, wt):
    """block scale code minimising sum wt*(w-q)^2 (wt [16])."""
    a = np.abs(w16)
    rtn = e4m3_encode((a.max(-1) / FP4_E2M1_MAX / ws2).astype(np.float32)).astype(np.int64)
    first = np.where(rtn & 0x80, 0, rtn).clip(1, 126)
    code = np.clip(first[:, None] + OFF[None, :], 1, 126)
    s = (E4[code] * ws2)[..., None]
    q = LV[P(np.broadcast_to(a[:, None, :], (a.shape[0], code.shape[1], 16)), s)] * s
    loss = (((a[:, None, :] - q) ** 2) * wt).sum(-1)
    return code[np.arange(code.shape[0]), np.argmin(loss, 1)]

def gptq(w, u, ws2, mode="base", d=None, fixed=None):
    w = np.array(w, np.float64); rows, cols = w.shape
    codes = np.empty((rows, cols), np.uint8); sc_all = np.empty((rows, cols // 16), np.int64)
    B = g.BLOCK
    for b0 in range(0, cols, B):
        b1 = min(b0 + B, cols); w1 = w[:, b0:b1].copy(); err = np.empty_like(w1); u1 = u[b0:b1, b0:b1]
        for gg in range(0, b1 - b0, 16):
            j0 = b0 + gg; wg = w1[:, gg:gg + 16]
            if fixed is not None: sc = fixed[:, j0 // 16]
            elif mode == "base": sc = g._block_scales(wg, ws2)
            elif mode == "diagH": sc = scales_wt(wg, ws2, d[j0:j0 + 16])
            elif mode == "uinv": sc = scales_wt(wg, ws2, 1.0 / np.diag(u)[j0:j0 + 16] ** 2)
            sc_all[:, j0 // 16] = sc; s = E4[sc] * ws2
            for j in range(16):
                col = wg[:, j]; lv = P(np.abs(col), s); c = np.where(col < 0, lv | 8, lv).astype(np.uint8)
                codes[:, j0 + j] = c; e = (col - E2[c] * s) / u1[gg + j, gg + j]; err[:, gg + j] = e
                if j < 15: wg[:, j + 1:] -= e[:, None] * u1[gg + j, gg + j + 1:gg + 16]
            if gg + 16 < b1 - b0: w1[:, gg + 16:] -= err[:, gg:gg + 16] @ u1[gg:gg + 16, gg + 16:]
        if b1 < cols: w[:, b1:] -= err @ u[b0:b1, b1:]
    return codes, sc_all

def deq(codes, sc, ws2): return E2[codes] * np.repeat(E4[sc] * ws2, 16, 1)

def refit(W, H, codes, sc, ws2, sweeps=1):
    """coordinate descent on each block scale code (offsets -3..3) under full H, codes fixed."""
    sc = sc.copy(); L = E2[codes]; E = W - deq(codes, sc, ws2); G = E @ H; r = np.arange(W.shape[0])
    for _ in range(sweeps):
        for b in range(sc.shape[1]):
            sl = slice(16 * b, 16 * b + 16); Lb = L[:, sl]; Hb = H[sl, sl]; s0 = E4[sc[:, b]] * ws2
            LHL = np.einsum("ri,ij,rj->r", Lb, Hb, Lb); LG = (Lb * G[:, sl]).sum(1)
            best = np.zeros(len(r)); bc = sc[:, b].copy()
            for o in (-3, -2, -1, 1, 2, 3):
                c = np.clip(sc[:, b] + o, 1, 126); ds = E4[c] * ws2 - s0
                dobj = -2 * ds * LG + ds * ds * LHL   # E changes by -ds*Lb
                m = dobj < best; best[m] = dobj[m]; bc[m] = c[m]
            ds = E4[bc] * ws2 - s0; ch = ds != 0
            if ch.any():
                dq = ds[ch, None] * Lb[ch]; E[ch, sl] -= dq; G[ch] -= dq @ H[sl, :]
            sc[:, b] = bc
    return sc

def err(W, Q, H):
    E = W - Q; return float(np.einsum("ij,ij->", E @ H, E))

def run(L):
    z = np.load(f"data/L{L}.npz"); H = z["H"].astype(np.float64); d = np.diag(H)
    us = {dm: g.inverse_hessian_factor(H, dm) for dm in (0.01, 0.001, 0.003, 0.03)}
    out = {}
    for p in ("gate_proj", "up_proj"):
        W = z[p].astype(np.float64); amax = np.abs(W).max(); ws2 = float(bf.tensor_scale(amax)); res = {}; tm = {}
        def rec(name, codes, sc, w2=ws2, t0=None):
            res[name] = err(W, deq(codes, sc, w2), H); tm[name] = round(time.time() - t0, 1)
        u = us[0.01]
        t = time.time(); c0, s0 = gptq(W, u, ws2); rec("0_base", c0, s0, t0=t)
        # sanity vs upstream encoder
        if L == 3 and p == "gate_proj":
            pk, ss, w2 = g._quantize_with(W[:256].astype(np.float32), u, amax)
            assert np.array_equal(ss, s0[:256].astype(np.uint8)), "mismatch vs upstream"
        t = time.time(); c, s = gptq(W, u, ws2, "diagH", d); rec("1_diagH_scale", c, s, t0=t)
        t = time.time(); c, s = gptq(W, u, ws2, "uinv"); rec("1b_uinv_scale", c, s, t0=t)
        t = time.time(); s1 = refit(W, H, c0, s0, ws2); rec("2a_refit_only", c0, s1, t0=t)
        c2, s2 = gptq(W, u, ws2, fixed=s1); rec("2_refit_gptq", c2, s2, t0=t)
        s3 = refit(W, H, c2, s2, ws2); rec("2c_refit_gptq_refit", c2, s3, t0=t)
        c4, s4 = gptq(W, u, ws2, fixed=s3); rec("2d_x2", c4, s4, t0=t)
        for dm in (0.001, 0.003, 0.03):
            t = time.time(); c, s = gptq(W, us[dm], ws2); rec(f"3_damp{dm}", c, s, t0=t)
        for f in (0.9, 0.95, 0.97, 1.03):
            w2 = float(bf.tensor_scale(amax * f)); t = time.time(); c, s = gptq(W, u, w2); rec(f"4_ws2x{f}", c, s, w2, t)
        base = res["0_base"]; out[p] = {k: {"rel": round(v / base, 4), "s": tm[k]} for k, v in res.items()}
        out[p]["_base_abs_rel_to_WHW"] = base / err(W, 0 * W, H)
        print(L, p, json.dumps(out[p]), flush=True)
    return out

if __name__ == "__main__":
    allr = {L: run(int(L)) for L in sys.argv[1:]}
    json.dump(allr, open(f"results_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
