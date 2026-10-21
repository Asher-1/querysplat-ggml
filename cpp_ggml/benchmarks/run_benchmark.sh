#!/usr/bin/env bash
# One-click benchmark collection: parity gate + quant x backend matrix + figures.
#
# Usage (from cpp_ggml/ or anywhere):
#   bash benchmarks/run_benchmark.sh [--S 2] [--backend cpu]
#        [--spy-device cpu] [--work /tmp/qs_bench]
#
# Produces:
#   results/querysplat/gate_s<S>_<backend>.log            seam cmp table
#   results/querysplat/timing_{cpp,py}_s4_*.json, quant_tiers.json
#   charts/querysplat/alignment_matrix.{md,json}, alignment_cpp_vs_py_matrix.png,
#                     latency_quant_backend.png, e2e_latency_bar.png,
#                     parity_scatter.png, quant_pareto.png
set -euo pipefail

QS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=2
BACKEND=cpu
SPY_DEVICE=cpu
WORK=/tmp/qs_bench
while [ $# -gt 0 ]; do
    case "$1" in
        --S) S="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --spy-device) SPY_DEVICE="$2"; shift 2 ;;
        --work) WORK="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

RES="$QS/benchmarks/results/querysplat"
CHARTS="$QS/benchmarks/charts/querysplat"
PY="${PY:-/home/asher/anaconda3/envs/python3.12/bin/python}"
# the official infer needs gsplat (CUDA rasterizer) — a separate env
PY_INFER="${PY_INFER:-/home/asher/anaconda3/envs/python3.13/bin/python}"
mkdir -p "$RES" "$CHARTS"

echo "== [1/3] parity gate (S=$S cli=$BACKEND spy=$SPY_DEVICE) =="
QS_BUILD_DIR="$QS/build-$BACKEND" S="$S" DEVICE="$BACKEND" SPY_DEVICE="$SPY_DEVICE" \
    OUT="$WORK/gate" PY="$PY" \
    bash "$QS/scripts/e2e_querysplat.sh" 2>&1 | tee "$RES/gate_s${S}_${BACKEND}.log"

echo "== [2/3] quant x backend matrix (official infer + cpp, S=4, 3 scenes) =="
PY="$PY_INFER" "$PY_INFER" "$QS/benchmarks/alignment_matrix.py" --work "$WORK/matrix"

echo "== [3/3] figures =="
"$PY_INFER" "$QS/benchmarks/plot_benchmarks.py" --work "$WORK/gate" --out "$CHARTS"
echo "benchmarks refreshed under:"
echo "  $RES"
echo "  $CHARTS"
