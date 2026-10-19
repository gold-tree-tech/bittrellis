# build.py <manifest> <out.gguf> <jobs>: HPC-02 build; kq_imatrix (GPTQ) outputs cached per (tensor, format, params)
import os, sys, time, json, hashlib, dataclasses
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
from pathlib import Path
from bittrellis.track import load_track
from bittrellis import cli
t = load_track("HPC-02")
h = cli._hpc02(t)
CACHE = Path("/workspace/enc_cache"); CACHE.mkdir(exist_ok=True)
for name in ("kq_imatrix", "kq_imat"):
    enc = h.ENCODERS[name]
    def cached(ctx, fmt, _fn=enc.fn, _ref=enc.ref):
        key = hashlib.sha256(f"{_ref}|{ctx.unit.tensor}|{fmt}|{json.dumps(ctx.params, sort_keys=True)}".encode()).hexdigest()[:24]
        p = CACHE / f"{key}.bin"
        if p.exists():
            return p.read_bytes()
        b = _fn(ctx, fmt)
        tmp = CACHE / f"{key}.{os.getpid()}.tmp"; tmp.write_bytes(b); os.replace(tmp, p)
        return b
    h.ENCODERS[name] = dataclasses.replace(enc, fn=cached)
t0 = time.time()
rec = h.build(Path(sys.argv[1]), out=Path(sys.argv[2]), jobs=int(sys.argv[3]), **cli._hpc02_sources(t))
print(f"BUILD_DONE {sys.argv[2]} {rec['candidate_id']} {time.time() - t0:.0f}s", flush=True)
