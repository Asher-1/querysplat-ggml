#!/usr/bin/env bash
# QuerySplat one-click runner — official python chain and/or the pure-C++
# cpp_ggml chain, from the repository root.
#
# Quickstart (no arguments needed):
#   ./run_querysplat.sh quick
#
# All subcommands:
#   ./run_querysplat.sh quick [opts]              # zero-setup demo (cpp chain)
#   ./run_querysplat.sh python --input DIR --out DIR [--tto] [--extra "..."]
#   ./run_querysplat.sh cpp    --input DIR --out DIR [--backend cpu|cuda|vulkan|auto]
#                              [--quant f32|f16|q8_0] [--views N] [--frames FILE]
#                              [--no-render] [--device-threads N]
#   ./run_querysplat.sh compare --input DIR --out DIR [...]   # both + PSNR
#   ./run_querysplat.sh eval   --out DIR [--dataset-root DIR] [--split JSON]
#                              [--limit N | --case-indices "I J .."] [--save-images]
#                              # DL3DV protocol, C++ (querysplat-eval) -> DIR/cpp
#   ./run_querysplat.sh eval-compare --out DIR [same]          # + official -> DIR/python,
#                              # then DIR/protocol_compare.md (eval_protocol_cpp.py)
#   ./run_querysplat.sh help
#
# Common options: --input DIR (image folder, png/jpg/jpeg/bmp/tga), --out DIR,
# --views N, --quant TIER, --backend B. Missing checkpoints / models produce
# copy-pasteable download or conversion commands instead of raw failures.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QS="$ROOT/cpp_ggml"

# ---- defaults ---------------------------------------------------------------
SUBCMD="${1:-help}"; [ $# -gt 0 ] && shift
# QUANT=f32 is the default: it is the only tier that fully aligns with the
# official torch chain on CUDA (f16-weight CUDA GEMM kernels quantize
# activations to f16 internally; measured 2026-10-17, see cpp_ggml/README.md).
# --quant f16 is the fast/compact tier (renders within ~0.1-0.4 dB of f32 on
# most scenes).
INPUT="" OUT="" FRAMES="" VIEWS="" QUANT=f32 BACKEND=auto RENDER=1
DEVICE_THREADS="" EXTRA="" DO_TTO=0 TTO_N_STEPS=20 TTO_LR=5e-3 TTO_LPIPS=0.05
DATASET_ROOT="" SPLITS="" LIMIT="" CASES="" SAVE_IMAGES=0

usage() { sed -n '2,24p' "$0"; }

die()  { echo "error: $*" >&2; exit 1; }
info() { echo "== $*"; }

# ---- python interpreter (official chain + dev tooling) ----------------------
# The first candidate that can import the official chain's runtime (torch +
# gsplat + lpips) wins; otherwise the first one that exists (enough for the
# numpy-only dev tooling). PY=/path/to/python overrides.
resolve_py() {
    if [ -n "${PY:-}" ]; then return; fi
    local c first=""
    for c in "$HOME"/anaconda3/envs/querysplat/bin/python \
             "$HOME"/miniconda3/envs/querysplat/bin/python \
             "$HOME"/anaconda3/envs/python3.1[0-9]/bin/python \
             "$HOME"/miniconda3/envs/python3.1[0-9]/bin/python \
             "$(command -v python3 || true)"; do
        [ -x "$c" ] || continue
        [ -n "$first" ] || first="$c"
        if "$c" -c "import torch, gsplat, lpips" >/dev/null 2>&1; then PY="$c"; return; fi
    done
    PY="$first"
    [ -n "$PY" ] || die "no python3 found; set PY=/path/to/python"
}

# Run an official module unchanged. scripts.models imports fused_ssim_cuda at
# module load, but only TTO calls it: without TTO and without the extension a
# stub module is registered first, so inference/evaluation still run as-is.
py_official() {  # $1 module, rest = its argv
    local mod="$1"; shift
    if [ "$DO_TTO" = 1 ] || "$PY" -c "import fused_ssim_cuda" >/dev/null 2>&1; then
        (cd "$ROOT" && "$PY" -m "$mod" "$@")
    else
        (cd "$ROOT" && "$PY" -c "import sys, types, runpy
m = types.ModuleType('fused_ssim_cuda'); m.fusedssim = m.fusedssim_backward = None
sys.modules['fused_ssim_cuda'] = m
runpy.run_module('$mod', run_name='__main__', alter_sys=True)" "$@")
    fi
}

# ---- checkpoint / model preflight (prints fixes, never fails silently) ------
# The paper checkpoint is the shipped decoder family: in a controlled
# comparison (CUDA + Vulkan x every quant tier x 3 scenes, 2026-10-17) it beat
# the 8192 checkpoint in every cell by +0.7 to +1.9 dB cpp-vs-GT; the current
# per-tier numbers are in cpp_ggml/benchmarks/charts/querysplat/alignment_matrix.md.
CKPT="$ROOT/checkpoints/querysplat_vggto_1B_512_paper.safetensors"
CFG="$ROOT/checkpoints/querysplat_vggto_1B_512_paper.yaml"
VGGT_CKPT="$ROOT/checkpoints/vggt_omega_1b_512.pt"
if [ ! -f "$VGGT_CKPT" ] && [ -f "$QS/third_party/map-anything-ggml/cpp_ggml/models/pytorch/vggt-omega/vggt_omega_1b_512.pt" ]; then
    VGGT_CKPT="$QS/third_party/map-anything-ggml/cpp_ggml/models/pytorch/vggt-omega/vggt_omega_1b_512.pt"
fi

preflight_python() {
    [ -f "$CKPT" ] || die "missing $CKPT
  fix: hf download inspatio/querysplat querysplat_vggto_1B_512_paper.safetensors --local-dir checkpoints
       (then verify: sha256sum -c SHA256SUMS)"
    [ -f "$VGGT_CKPT" ] || die "missing $VGGT_CKPT
  fix: hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints"
    [ -f "$CFG" ] || die "missing config $CFG (it ships with the repo)"
    resolve_py
    "$PY" -c "import torch, gsplat, lpips" 2>/dev/null || die "torch/gsplat/lpips not importable with $PY
  fix: see the README Installation section (conda env + torch cu128), or set PY="
}

gguf_paths() {
    OMEGA_GGUF="$QS/models/gguf/vggt-omega-1b-512-$QUANT.gguf"
    DEC_GGUF="$QS/models/gguf/querysplat-paper-$QUANT.gguf"
}

preflight_cpp() {
    gguf_paths
    [ -f "$OMEGA_GGUF" ] || die "missing encoder $OMEGA_GGUF
  fix: see cpp_ggml/README.md (convert_vggt_omega_to_gguf.py --outtype $QUANT
       from the pinned upstream, checkpoint vggt_omega_1b_512.pt)"
    if [ ! -f "$DEC_GGUF" ]; then
        die "missing decoder $DEC_GGUF
  fix: $PY $QS/scripts/convert_querysplat_to_gguf.py $CKPT $CFG $DEC_GGUF --outtype $QUANT"
    fi
}

# ---- cpp build on demand -----------------------------------------------------
build_dir_for() {  # backend -> build dir
    case "$1" in
        cuda) echo "$QS/build-cuda" ;;
        vulkan) echo "$QS/build-vulkan" ;;
        *) echo "$QS/build-cpu" ;;
    esac
}

build_flag_for() {
    case "$1" in
        cuda) echo "-DMAP_GGML_CUDA=ON" ;;
        vulkan) echo "-DMAP_GGML_VULKAN=ON" ;;
        *) echo "" ;;
    esac
}

ensure_build() {
    local backend="$1" bdir flag
    bdir="$(build_dir_for "$backend")"
    flag="$(build_flag_for "$backend")"
    if [ -x "$bdir/bin/querysplat-cli" ] && [ -x "$bdir/bin/querysplat-eval" ]; then
        echo "$bdir"; return
    fi
    info "building cpp_ggml ($backend) — first time only"
    # shellcheck disable=SC2086
    cmake -B "$bdir" -S "$QS" $flag -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$bdir" -j "$(nproc)" >/dev/null
    echo "$bdir"
}

vram_free_mb() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | sort -rn | head -1 || echo 0
}

# ---- option parsing ----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        --views) VIEWS="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --device-threads) DEVICE_THREADS="$2"; shift 2 ;;
        --render) RENDER=1; shift ;;
        --no-render) RENDER=0; shift ;;
        --tto) DO_TTO=1; shift ;;
        --tto-n-steps) TTO_N_STEPS="$2"; shift 2 ;;
        --tto-lr) TTO_LR="$2"; shift 2 ;;
        --tto-lpips-weight) TTO_LPIPS="$2"; shift 2 ;;
        --extra) EXTRA="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --split) SPLITS="$SPLITS $2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --case-indices) CASES="$2"; shift 2 ;;
        --save-images) SAVE_IMAGES=1; shift ;;
        *) die "unknown option: $1 (see ./run_querysplat.sh help)" ;;
    esac
done

pick_backend() {
    case "$BACKEND" in
        auto)
            # querysplat needs ~7 GB of VRAM at S=2 (weights + workspace);
            # 8 GB free is a safe bar, otherwise stay on CPU.
            if [ "$(vram_free_mb)" -ge 8000 ] && [ -x "$(build_dir_for cuda)/bin/querysplat-cli" ]; then
                BACKEND=cuda
            else
                BACKEND=cpu
            fi ;;
        cpu|cuda|vulkan) ;;
        *) die "--backend must be cpu|cuda|vulkan|auto" ;;
    esac
}

out_root() {
    [ -n "$OUT" ] && { mkdir -p "$OUT"; return; }
    if [ -n "$INPUT" ]; then
        OUT="$ROOT/outputs/$(basename "$INPUT")"
    else
        OUT="$ROOT/outputs/demo"
    fi
    mkdir -p "$OUT"
}

run_cpp() {
    # $1 = frames.bin path (empty => use INPUT images)
    local bdir cli prefix
    pick_backend
    bdir="$(ensure_build "$BACKEND")"
    cli="$bdir/bin/querysplat-cli"
    out_root
    preflight_cpp
    prefix="$OUT/cpp/querysplat"
    mkdir -p "$OUT/cpp"
    local -a args=(--encoder "$OMEGA_GGUF" --decoder "$DEC_GGUF"
                   --out-prefix "$prefix" --device "$BACKEND" --timing)
    [ -n "$DEVICE_THREADS" ] && args+=(--threads "$DEVICE_THREADS")
    if [ -n "$1" ]; then
        local s="$VIEWS"
        if [ -z "$s" ]; then
            local hw=512 elems
            elems=$(( $(stat -c%s "$1") / 4 / 3 / hw / hw ))
            s=$elems
        fi
        args+=(--bin "$1" --S "$s")
    else
        [ -n "$INPUT" ] || die "cpp: needs --input DIR or --frames FILE"
        args+=(--images "$INPUT")
    fi
    [ "$RENDER" = 1 ] && args+=(--render)
    info "cpp chain: backend=$BACKEND quant=$QUANT out=$OUT"
    "$cli" "${args[@]}"
    echo
    echo "cpp outputs:"
    ls -1 "$OUT/cpp/" 2>/dev/null | while read -r f; do echo "  $OUT/cpp/$f"; done
}

run_python_official() {
    out_root
    preflight_python
    [ -n "$INPUT" ] || die "python: needs --input DIR (image folder)"
    local -a pargs=(--config "$CFG" --checkpoint "$CKPT"
                    --input_folder "$INPUT" --output_dir "$OUT/python")
    [ "$DO_TTO" = 1 ] && pargs+=(--use_tto)
    if [ -n "$EXTRA" ]; then read -r -a pargs_extra <<< "$EXTRA"; pargs+=("${pargs_extra[@]}"); fi
    info "python official chain (requires CUDA) -> $OUT/python"
    py_official scripts.infer "${pargs[@]}"
}

# ---- DL3DV evaluation protocol (evaluations/evaluate_jsons.py) ---------------
# Both sides get the same selection flags; output trees share one layout
# (<split>/<group>/case_*/{metrics,timing}.json, cameras.npz, summary.json).
eval_selection() {  # -> EVAL_ARGS
    EVAL_ARGS=(--dataset-root "${DATASET_ROOT:-$ROOT/data/DL3DV-Evaluation}")
    # shellcheck disable=SC2206
    [ -n "$SPLITS" ] && EVAL_ARGS+=(--eval-jsons $SPLITS)
    [ -n "$LIMIT" ] && EVAL_ARGS+=(--limit "$LIMIT")
    # shellcheck disable=SC2206
    [ -n "$CASES" ] && EVAL_ARGS+=(--case-indices $CASES)
    [ "$SAVE_IMAGES" = 1 ] && EVAL_ARGS+=(--save-images)
    return 0
}

run_eval_cpp() {
    local bdir lpips
    pick_backend
    bdir="$(ensure_build "$BACKEND")"
    [ -n "$OUT" ] || OUT="$ROOT/outputs/eval_protocol"
    mkdir -p "$OUT"
    preflight_cpp
    lpips="$QS/models/gguf/lpips-vgg-f32.gguf"
    if [ ! -f "$lpips" ]; then
        resolve_py
        die "missing LPIPS metric weights $lpips
  fix: $PY $QS/scripts/convert_lpips_to_gguf.py $lpips"
    fi
    eval_selection
    local tto_args=()
    [ "$DO_TTO" = 1 ] && tto_args=(--use-tto --tto-n-steps "$TTO_N_STEPS"
                                   --tto-lr "$TTO_LR"
                                   --tto-lpips-weight "$TTO_LPIPS")
    info "querysplat-eval: backend=$BACKEND quant=$QUANT -> $OUT/cpp"
    (cd "$ROOT" && "$bdir/bin/querysplat-eval" --encoder "$OMEGA_GGUF" --decoder "$DEC_GGUF" \
        --lpips "$lpips" --device "$BACKEND" --output-dir "$OUT/cpp" \
        ${DEVICE_THREADS:+--threads "$DEVICE_THREADS"} "${tto_args[@]}" "${EVAL_ARGS[@]}")
}

run_eval_python() {
    [ -n "$OUT" ] || OUT="$ROOT/outputs/eval_protocol"
    mkdir -p "$OUT"
    preflight_python
    eval_selection
    local -a targs=()
    [ "$DO_TTO" = 1 ] && targs=(--use-tto --tto-n-steps "$TTO_N_STEPS"
                                --tto-lr "$TTO_LR"
                                --tto-lpips-weight "$TTO_LPIPS")
    info "official evaluate_jsons (requires CUDA) -> $OUT/python"
    py_official evaluations.evaluate_jsons --config "$CFG" --checkpoint "$CKPT" \
        --output-dir "$OUT/python" "${EVAL_ARGS[@]}" "${targs[@]}"
}

psnr_table() {  # $1 cpp png prefix dir/prefix, $2 python rendered dir, $3 view count
    resolve_py
    "$PY" - "$1" "$2" "$3" <<'PYEOF'
import glob, math, os, sys
import numpy as np
from PIL import Image
cpp_prefix, py_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for i in range(n):
    a = glob.glob(f"{cpp_prefix}_render_view{i}.png")
    b = os.path.join(py_dir, f"render_view{i}.png")
    if not a or not os.path.isfile(b):
        continue
    x = np.asarray(Image.open(a[0]), dtype=np.float64)
    y = np.asarray(Image.open(b), dtype=np.float64)
    if x.shape != y.shape:
        rows.append((i, None)); continue
    mse = ((x - y) ** 2).mean()
    rows.append((i, 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)))
if rows:
    print(f"{'view':>5} {'PSNR(cpp vs official py)':>26}")
    for i, v in rows:
        print(f"{i:>5} {'n/a (shape)' if v is None else f'{v:10.2f} dB'}")
    vals = [v for _, v in rows if v is not None]
    if vals:
        print(f"{'mean':>5} {sum(vals)/len(vals):10.2f} dB")
else:
    print("no comparable rendered views found")
PYEOF
}

# ---- subcommands --------------------------------------------------------------
case "$SUBCMD" in
    help|-h|--help)
        usage ;;
    quick)
        resolve_py
        mkdir -p /tmp/qs_quick
        info "generating deterministic demo frames (S=${VIEWS:-2})"
        "$PY" "$QS/scripts/make_test_frames.py" --out /tmp/qs_quick/frames.bin \
            --S "${VIEWS:-2}" --H 512 --W 512
        run_cpp /tmp/qs_quick/frames.bin
        echo
        echo "next steps:"
        echo "  ./run_querysplat.sh cpp --input path/to/your/images --out outputs/my_scene"
        echo "  ./run_querysplat.sh compare --input path/to/your/images   # vs official python"
        ;;
    cpp)
        run_cpp "$FRAMES"
        ;;
    python)
        run_python_official
        ;;
    eval)
        run_eval_cpp
        ;;
    eval-compare)
        run_eval_cpp
        run_eval_python
        resolve_py
        "$PY" "$QS/benchmarks/eval_protocol_cpp.py" --official "$OUT/python" \
            --cpp "$OUT/cpp" --out "$OUT/protocol_compare.md"
        ;;
    compare)
        [ -n "$INPUT" ] || die "compare: needs --input DIR"
        out_root
        preflight_cpp
        run_cpp ""
        if command -v nvidia-smi >/dev/null 2>&1 && [ "$(vram_free_mb)" -ge 8000 ]; then
            run_python_official
            N_PY=$(ls "$OUT/python/rendered" 2>/dev/null | wc -l || echo 0)
            N_CPP=$(ls "$OUT"/cpp/*_render_view*.png 2>/dev/null | wc -l || echo 0)
            if [ "${N_CPP:-0}" -gt 0 ] && [ "${N_PY:-0}" -gt 0 ]; then
                info "rendered-view agreement (cpp CPU rasterizer vs official gsplat)"
                psnr_table "$OUT/cpp/querysplat" "$OUT/python/rendered" "$N_PY"
            fi
            for ply in "$OUT"/cpp/querysplat_gaussians.ply "$OUT"/python/gaussians.ply; do
                [ -f "$ply" ] && grep -m1 "element vertex" "$ply" | sed "s|^|$(basename "$(dirname "$ply")")/$(basename "$ply"): |"
            done
        else
            echo "compare: official python chain skipped — needs a CUDA GPU with >= 8 GB free"
            echo "(cpp chain results are complete above; re-run with a free GPU for the PSNR table)"
        fi
        ;;
    *)
        die "unknown subcommand: $SUBCMD (see ./run_querysplat.sh help)" ;;
esac
