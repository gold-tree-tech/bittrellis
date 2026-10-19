"""Range-fetch single BF16 tensors (or a leading slice) from the pinned Qwen3.6-35B-A3B safetensors -> .npy float32."""
import json, struct, sys, urllib.request
import numpy as np
REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
BASE = f"https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/{REV}/"
IDX = json.load(open(sys.argv[1]))["weight_map"]
OUT = sys.argv[2]
_hdr = {}
def rng(f, a, b):
    req = urllib.request.Request(BASE + f, headers={"Range": f"bytes={a}-{b-1}"})
    return urllib.request.urlopen(req).read()
def header(f):
    if f not in _hdr:
        n = struct.unpack("<Q", rng(f, 0, 8))[0]
        _hdr[f] = (8 + n, json.loads(rng(f, 8, 8 + n)))
    return _hdr[f]
def get(name, lead=None):
    f = IDX[name]; off, h = header(f); m = h[name]
    a, b = m["data_offsets"]; shape = m["shape"]
    assert m["dtype"] == "BF16", m["dtype"]
    if lead is not None:
        per = (b - a) // shape[0]; b = a + per * lead; shape = [lead] + shape[1:]
    raw = np.frombuffer(rng(f, off + a, off + b), np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
for spec in sys.argv[3:]:
    name, _, lead = spec.partition("@")
    x = get(name, int(lead) if lead else None)
    np.save(f"{OUT}/{name.replace('model.language_model.layers.', 'L')}.npy", x)
    print(name, x.shape, flush=True)
