"""Per-unit rounding error at Q4_K: kq_rtn vs merged kq_imat vs our kq_imatrix (diag only / GPTQ)."""
import json, sys, time, importlib.util
import numpy as np
sys.path.insert(0, "/home/administrator/dev/gittensor/bittrellis-hpc02")
from bittrellis import kquant
from bittrellis.hpc02_encoders import kq_imatrix as ours
spec = importlib.util.spec_from_file_location("kq_imat_merged", "/home/administrator/dev/gittensor/bittrellis/hpc02_next2/cpu/kq_imat_merged.py")
mi = importlib.util.module_from_spec(spec); spec.loader.exec_module(mi)
from safetensors_np import load
CAL = load("/home/administrator/dev/gittensor/bittrellis-hpc02/models/hpc02-calib-v1/model-00001-of-00001.safetensors")
fmt = sys.argv[3] if len(sys.argv) > 3 else "Q4_K"
path, kind = sys.argv[1], sys.argv[2]            # kind: dense:<layer>, down:<layer>, noh
W = np.load(path).astype(np.float32)
res = {"tensor": path.split("/")[-1], "fmt": fmt}
def deq(b, cols): return kquant.dequantize(fmt, b, cols).reshape(-1, cols).astype(np.float64)
def t(): return time.time()
if kind.startswith("dense"):
    L = int(kind.split(":")[1]); H = CAL[f"blk.{L}.attn_input.xtx"].astype(np.float64)
    rows, cols = W.shape
    def eH(b):
        d = deq(b, cols) - W
        return float(np.einsum("rc,cd,rd->", d, H, d, optimize=True))
    hd = np.maximum(np.diag(H), 0)
    t0 = t(); e_rtn = eH(kquant.RTN[fmt](W))
    e_imat = eH(mi._encode.__wrapped__(W) if False else mi.quantize(W.astype(np.float64).reshape(-1, 256), np.broadcast_to(mi._weights(W.reshape(-1,256).astype(np.float64), np.broadcast_to(np.maximum(hd, mi.FLOOR*hd.mean()), W.shape).reshape(-1,256)), (W.size//256, 256)), fmt)) 
    e_od = eH(ours.encode_weighted(W, hd, fmt))
    t1 = t(); e_og = eH(ours.encode_gptq(W, H, fmt)); res["gptq_s"] = round(t() - t1, 1)
    res.update(rtn=1.0, imat=e_imat / e_rtn, ours_diag=e_od / e_rtn, ours_gptq=e_og / e_rtn)
elif kind.startswith("down"):
    L = int(kind.split(":")[1]); E = W.shape[0]
    ss = CAL[f"blk.{L}.exps_down.sumsq"].astype(np.float64); cnt = CAL[f"blk.{L}.exps.count"].astype(np.float64)
    true = ss[:E] / np.maximum(cnt[:E], 1)[:, None]
    imp_i = mi.expert_importance(ss, cnt)[:E]
    tot = cnt.sum(); pooled = ss.sum(0) / tot
    imp_o = (ss + 64 * pooled[None]) / (cnt + 64)[:, None]; imp_o = imp_o[:E]
    acc = dict(rtn=0., imat=0., ours=0.)
    for e in range(E):
        x = W[e]; cols = x.shape[1]; w = true[e][None, :]
        err = lambda b: float((w * (deq(b, cols) - x) ** 2).sum()) * cnt[e]   # weight by tokens routed
        acc["rtn"] += err(kquant.RTN[fmt](x))
        xb = x.astype(np.float64).reshape(-1, 256)
        ib = np.broadcast_to(imp_i[e], x.shape).reshape(-1, 256)
        acc["imat"] += err(mi.quantize(xb, mi._weights(xb, ib), fmt))
        acc["ours"] += err(ours.encode_weighted(x, imp_o[e], fmt))
    res.update({k: v / acc["rtn"] for k, v in acc.items()}, experts=E, counts=cnt[:E].tolist()[:6])
else:  # no statistics: plain MSE, uniform
    cols = W.shape[1]
    m = lambda b: float(((deq(b, cols) - W) ** 2).sum())
    r = m(kquant.RTN[fmt](W)); xb = W.astype(np.float64).reshape(-1, 256)
    res.update(rtn=1.0, imat=m(mi.quantize(xb, mi._weights(xb, None), fmt)) / r,
               ours_uniform=m(ours.encode_weighted(W, np.ones(cols), fmt)) / r)
print(json.dumps(res), flush=True)
