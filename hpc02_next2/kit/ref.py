import time
from pathlib import Path
from bittrellis.cli import DEFAULTS
from bittrellis.eval.corpus import load_corpus
from bittrellis.eval.reference import build_reference
t0 = time.time()
build_reference(Path("models/Qwen3.6-35B-A3B"), load_corpus(Path(DEFAULTS["corpus"])), Path("data/reference/hpc02-public-v2-k256"),
                topk=256, gpu_gib=20, cpu_gib=80, offload=None)
print(f"REF_DONE {time.time() - t0:.0f}s", flush=True)
