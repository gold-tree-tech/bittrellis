HPC-02 next2 measurement kit (not run). Host: 1x RTX 5090, >=48 cores, >=250 GB disk, inet_down_cost <= $0.01/GB.
Upload bt.tar.gz (upstream/main c8af001 + kq_imatrix as hpc02_encoders/kq_imatrix_gptq.py), setup*.sh, ref.py, build.py,
run.sh and var/ to /workspace. Then: bash setup1.sh (pip + ~160 GB downloads), bash run.sh (setup2 = sparkinfer build).
Variants: K147 = #147's recipe (anchor, ledger RP-KL 0.0855) | N3i = #144 map + kq_imat | N3g = #144 map + GPTQ on
attn q/k/v + gdn qkv/z | N1g = #145 map + GPTQ. Ledger scale = our RP-KL x 0.0855 / K147. Guard: tasks vs V0 570.
