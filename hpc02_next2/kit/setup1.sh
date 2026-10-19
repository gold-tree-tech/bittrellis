#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/bt && tar -xzf /workspace/bt.tar.gz -C /workspace/bt && cd /workspace/bt
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev build-essential cmake ninja-build curl ca-certificates git pkg-config tmux >/dev/null
python3 -m venv /workspace/venv
. /workspace/venv/bin/activate
pip install -q -U pip wheel
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -e ".[dev,eval,corpus,reference]"
pip install -q "huggingface_hub[cli]>=0.34"
echo PIP_DONE
export HF_HUB_DISABLE_XET=1
# downloads in parallel
(scripts/setup_models.sh qwen36_gguf > /workspace/dl_gguf.log 2>&1 || true; echo "rc=done" >> /workspace/dl_gguf.log) &
(hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir models/Qwen3.6-35B-A3B --max-workers 16 > /workspace/dl_bf16.log 2>&1; echo "rc=$?" >> /workspace/dl_bf16.log) &
(bittrellis --track HPC-02 calibration fetch > /workspace/dl_calib.log 2>&1; echo "rc=$?" >> /workspace/dl_calib.log) &
wait
echo DL_DONE
