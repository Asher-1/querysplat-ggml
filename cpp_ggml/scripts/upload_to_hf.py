#!/usr/bin/env python3
"""Upload the QuerySplat GGUF family to Hugging Face (Asher-1/querysplat_vggto).

One-shot publisher: creates the repo if missing, uploads the four GGUFs
(sha256-printed before each transfer) plus the model card, then lists the
result. Requires an HF token with write access (hf auth login / HF_TOKEN).
The encoder GGUFs are NOT uploaded here: vggt-omega-1b-512-{f32,f16,q8_0}.gguf
are already published at Asher-1/map_anything_vggt (digests pinned in
ACloudViewer AICore asset_digests.h).

Usage:
  /home/asher/anaconda3/envs/python3.13/bin/python \\
      cpp_ggml/scripts/upload_to_hf.py [--dry-run]
"""
import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Asher-1/querysplat_vggto"
# (local path relative to cpp_ggml, repo filename) — decoder family + the
# LPIPS metric weights (f32-only by design: the metric is the quantization
# judge; see models/MODEL_CARDS.md)
FILES = [
    ("models/gguf/querysplat-paper-f32.gguf", "querysplat-paper-f32.gguf"),
    ("models/gguf/querysplat-paper-f16.gguf", "querysplat-paper-f16.gguf"),
    ("models/gguf/querysplat-paper-q8_0.gguf", "querysplat-paper-q8_0.gguf"),
    ("models/gguf/lpips-vgg-f32.gguf", "lpips-vgg-f32.gguf"),
]
CARD = "models/hf/README.md"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-ggml",
                    default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dry-run", action="store_true",
                    help="hash + list only; no repo writes")
    args = ap.parse_args()
    root = Path(args.cpp_ggml)
    api = HfApi()
    print(f"hf user: {api.whoami()['name']}")
    if not args.dry_run:
        api.create_repo(REPO, exist_ok=True)

    for local, name in FILES:
        p = root / local
        if not p.is_file():
            sys.exit(f"missing {p}")
        digest = sha256(p)
        print(f"{name}: {p.stat().st_size / 1e9:.2f} GB  "
              f"sha256={digest}", flush=True)
        if args.dry_run:
            continue
        api.upload_file(path_or_fileobj=str(p), path_in_repo=name,
                        repo_id=REPO, repo_type="model")
        print(f"  uploaded {name}", flush=True)

    card = root / CARD
    if card.is_file():
        print(f"README.md (model card): {card.stat().st_size} bytes")
        if not args.dry_run:
            api.upload_file(path_or_fileobj=str(card),
                            path_in_repo="README.md", repo_id=REPO,
                            repo_type="model")
    if not args.dry_run:
        print("repo contents:")
        for f in api.list_repo_files(REPO):
            print(f"  {f}")
    print("done" if not args.dry_run else "dry-run done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
