import json, struct, sys, urllib.request, numpy as np, os
REV="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CAL="https://github.com/coderbench/bittrellis/releases/download/calib-hpc01-v1/{}"
HF=f"https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{REV}/{{}}"
def get(url, rng=None):
    req=urllib.request.Request(url, headers={"Range":f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    return urllib.request.urlopen(req, timeout=600).read()
def header(url):
    n=struct.unpack("<Q", get(url,(0,7)))[0]
    return 8+n, json.loads(get(url,(8,8+n-1)))
def tensor(url, name, hdr_cache={}, rows=None):
    if url not in hdr_cache: hdr_cache[url]=header(url)
    base,h=hdr_cache[url]; m=h[name]; a,b=m["data_offsets"]; shp=m["shape"]
    dt={"F32":(np.float32,4),"BF16":(np.uint16,2),"F64":(np.float64,8)}[m["dtype"]]
    if rows is not None:
        rowb=int(np.prod(shp[1:]))*dt[1]; b=a+rows*rowb; shp=[rows]+shp[1:]
    raw=np.frombuffer(get(url,(base+a,base+b-1)),dt[0]).reshape(shp)
    if m["dtype"]=="BF16": raw=(raw.astype(np.uint32)<<16).view(np.float32)
    return raw
if __name__=="__main__":
    layers=[int(x) for x in sys.argv[1:]]
    ci=json.loads(get(CAL.format("model.safetensors.index.json")))["weight_map"]
    hi=json.loads(get(HF.format("model.safetensors.index.json")))["weight_map"]
    for L in layers:
        out=f"data/L{L}.npz"
        if os.path.exists(out): continue
        k=f"model.language_model.layers.{L}.mlp.input_xtx"
        H=tensor(CAL.format(ci[k]),k)
        d={"H":H}
        for p in ("gate_proj","up_proj"):
            wk=f"model.language_model.layers.{L}.mlp.{p}.weight"
            d[p]=tensor(HF.format(hi[wk]),wk,rows=1024)
        np.savez(out,**d); print(L,H.shape,H.dtype,d["gate_proj"].shape,flush=True)
