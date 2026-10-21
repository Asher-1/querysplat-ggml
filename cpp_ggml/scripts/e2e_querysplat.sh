#!/usr/bin/env bash
# One-click QuerySplat parity gate (M1): convert -> spy(torch) -> cli -> cmp.
#
# Env overrides:
#   CKPT / CFG      QuerySplat safetensors + matching yaml
#   VGGT_CKPT       frozen VGGT-Omega torch checkpoint (spy needs it)
#   OMEGA_GGUF      encoder GGUF (defaults to models/gguf/vggt-omega-1b-512-f32.gguf)
#   S               number of views (default 2; 4 = canonical)
#   DEVICE          cli backend (cpu/cuda)
#   SPY_DEVICE      torch device for the spy (default cuda)
#   OUT             work dir (default /tmp/qs_gate)
#   PY              python interpreter with torch+safetensors+yaml
#                   (default: the querysplat conda env python3.12)
set -euo pipefail

QS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$QS/.." && pwd)"

CKPT="${CKPT:-$ROOT/checkpoints/querysplat_vggto_1B_512_paper.safetensors}"
CFG="${CFG:-$ROOT/checkpoints/querysplat_vggto_1B_512_paper.yaml}"
VGGT_CKPT="${VGGT_CKPT:-$ROOT/checkpoints/vggt_omega_1b_512.pt}"
if [ ! -f "$VGGT_CKPT" ]; then
    # fallback: the embedded upstream keeps the official torch ckpt
    VGGT_CKPT="$QS/third_party/map-anything-ggml/cpp_ggml/models/pytorch/vggt-omega/vggt_omega_1b_512.pt"
fi
OMEGA_GGUF="${OMEGA_GGUF:-$QS/models/gguf/vggt-omega-1b-512-f32.gguf}"
DEC_GGUF="${DEC_GGUF:-$QS/models/gguf/querysplat-paper-f32.gguf}"
S="${S:-2}"
DEVICE="${DEVICE:-cpu}"
SPY_DEVICE="${SPY_DEVICE:-cuda}"
OUT="${OUT:-/tmp/qs_gate}"
PY="${PY:-/home/asher/anaconda3/envs/python3.12/bin/python}"

SH_DEGREE=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['gaussian_sh_degree'])")

mkdir -p "$OUT"
echo "== [1/5] frames =="
"$PY" "$QS/scripts/make_test_frames.py" --out "$OUT/frames.bin" --S "$S"

echo "== [2/5] converter =="
if [ ! -f "$DEC_GGUF" ]; then
    "$PY" "$QS/scripts/convert_querysplat_to_gguf.py" "$CKPT" "$CFG" "$DEC_GGUF" --outtype f32
else
    echo "decoder GGUF exists: $DEC_GGUF"
fi

echo "== [3/5] spy (torch reference) =="
"$PY" "$QS/scripts/spy_torch_querysplat.py" --config "$CFG" --checkpoint "$CKPT" \
    --vggt-checkpoint "$VGGT_CKPT" --bin "$OUT/frames.bin" --S "$S" \
    --out-dir "$OUT/spy" --device "$SPY_DEVICE" --timing

echo "== [4/5] querysplat-cli =="
mkdir -p "$OUT/cpp"
BUILD_DIR="${QS_BUILD_DIR:-$QS/build-cpu}"
"$BUILD_DIR/bin/querysplat-cli" \
    --encoder "$OMEGA_GGUF" --decoder "$DEC_GGUF" \
    --bin "$OUT/frames.bin" --S "$S" --out-prefix "$OUT/cpp" \
    --device "$DEVICE" --dump-stage "$OUT/cpp" --timing

echo "== [5/5] compare =="
PROFILE=$([[ "$CFG" == *paper* ]] && echo paper || echo 8192)
"$PY" "$QS/scripts/cmp_querysplat.py" "$OUT/cpp" "$OUT/spy" \
    --S "$S" --H 512 --W 512 --sh-degree "$SH_DEGREE" --profile "$PROFILE"
