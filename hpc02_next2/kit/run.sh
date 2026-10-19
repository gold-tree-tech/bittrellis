#!/usr/bin/env bash
# orchestrates everything after setup1 (pip + downloads) is done
cd /workspace/bt && . /workspace/venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_HOME=/usr/local/cuda-13.0
L=/workspace/run.log; mkdir -p /workspace/cand /workspace/art
NC=$(env -u OMP_NUM_THREADS nproc); J=$(( NC > 8 ? NC - 6 : 4 )); [ $J -gt 20 ] && J=20
dl() { until grep -q "rc=" /workspace/dl_$1.log 2>/dev/null; do sleep 15; done; }
echo "START $(date -u +%T) nproc=$NC jobs=$J" >> $L
( dl gguf; dl calib; python /workspace/build.py /workspace/var/K147.yaml /workspace/cand/K147.gguf 4 > /workspace/build_K147.log 2>&1
  python /workspace/build.py /workspace/var/N3i.yaml /workspace/cand/N3i.gguf 4 > /workspace/build_N3i.log 2>&1 ) &
( dl gguf; dl calib; python /workspace/build.py /workspace/var/N3g.yaml /workspace/cand/N3g.gguf $J > /workspace/build_N3g.log 2>&1
  echo "N3g built $(date -u +%T)" >> $L
  python /workspace/build.py /workspace/var/N1g.yaml /workspace/cand/N1g.gguf $J > /workspace/build_N1g.log 2>&1
  echo "N1g built $(date -u +%T)" >> $L ) &
# GPU side: sparkinfer + reference while CPU builds
( grep -q SI_DONE /workspace/setup2.log 2>/dev/null || bash /workspace/setup2.sh > /workspace/setup2.log 2>&1; echo "SI $(date -u +%T)" >> $L
  dl bf16; python /workspace/ref.py > /workspace/ref.log 2>&1; r=$?; echo "REF $(date -u +%T) rc=$r" >> $L ) &
wait_for() { until grep -q "BUILD_DONE\|Traceback" /workspace/build_$1.log 2>/dev/null; do sleep 15; done; }
until grep -q "^REF" $L; do sleep 15; done; grep -q "^REF.*rc=0" $L || { echo "REF FAILED" >> $L; exit 1; }
for c in K147 N3i N3g N1g; do
  wait_for $c
  t0=$(date +%s)
  bittrellis --track HPC-02 evaluate /workspace/cand/$c.gguf --out /workspace/art/$c > /workspace/eval_$c.log 2>&1; r=$?
  echo "EVAL_DONE $c rc=$r $(( $(date +%s) - t0 ))s $(date -u +%T)" >> $L
done
wait
echo ALL_DONE $(date -u +%T) >> $L
