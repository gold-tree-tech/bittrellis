#144's map (every attention and Gated DeltaNet projection on Q5_K, routed experts on Q4_K except the L34/38/39 downs, lm_head and embeddings on Q4_K; decode 479.8, 23.45 GiB) with the kq_imatrix encoder from #146 on attention q/k/v and recurrent qkv/z, and kq_imat everywhere else. #144 failed the task guard on kq_rtn (558/784); this keeps its formats and only changes the rounding.

Recipe `manifests/hpc02/pr144-kq-imatrix-gptq-attn-gdn-q5k.yaml`, id `0bb8615daa65c1a2` (`bittrellis --track HPC-02 manifest ... --against manifests/` passes). Needs #146 merged; the branch sits on it.

Measured on an RTX 5090 with the repo's tools, paired with #147 built on the same box (#147 read 0.0856 here vs 0.0855 posted, decode 474.8 vs 474.6):

- RP-KL 0.0951 vs 0.0856, paired +0.0096 [+0.0021, +0.0177]
- decode 480.6 / 480.7 tok/s vs 474.8
- tasks 563/784 vs V0 570 (lost 22, gained 15), PASS

Our box read #146 at 0.0526 where the evaluator got 0.0547, so expect about 0.099 here. The row's gain is its decode over #147.

Tests: pytest 215 pass (one pr_bot test needs a plain clone, not a worktree), ruff clean.
