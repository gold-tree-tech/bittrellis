import json, struct, numpy as np
def load(p):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
    mm = np.memmap(p, np.uint8, "r")
    dt = {"F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32}
    return {k: np.frombuffer(mm[base + v["data_offsets"][0]: base + v["data_offsets"][1]], dt[v["dtype"]]).reshape(v["shape"])
            for k, v in h.items() if k != "__metadata__"}
