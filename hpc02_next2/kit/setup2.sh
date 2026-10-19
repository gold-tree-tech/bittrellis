#!/usr/bin/env bash
set -euo pipefail
export RUSTUP_HOME=/workspace/rust/rustup CARGO_HOME=/workspace/rust/cargo
mkdir -p $RUSTUP_HOME $CARGO_HOME
[ -x "$CARGO_HOME/bin/rustc" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q --default-toolchain stable --no-modify-path
export PATH="$CARGO_HOME/bin:/workspace/venv/bin:$PATH"
cd /workspace/bt
PYTHON=/workspace/venv/bin/python scripts/setup_sparkinfer.sh HPC-02
echo SI_DONE
