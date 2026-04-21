"""Upload the cmbemu dataset files to the Hugging Face Hub.

Reads HF_TOKEN from the process environment (or from a .env file in the
project root if unset). Idempotent: re-running overwrites existing files
in the repo.

    python scripts/upload_dataset.py --dry-run
    python scripts/upload_dataset.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "borisbolliet/cmbemu-competition-v1"

FILES = [
    # (local path under data/, path in repo)
    ("data/test_small.npz",  "test_small.npz"),
    ("data/train_small.npz", "train_small.npz"),
    ("data/test.npz",        "test.npz"),
    ("data/train.npz",       "train.npz"),
]
README_SRC = ROOT / "scripts" / "hf_dataset_readme.md"
README_DST = "README.md"


def _load_env_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == "HF_TOKEN":
            v = val.strip().strip("'").strip('"')
            return v or None
    return None


def _size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List planned actions without uploading")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    token = _load_env_token()
    if not token:
        print("HF_TOKEN not found (env or .env). Aborting.", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # Sanity: does the target repo exist?
    try:
        info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
        print(f"repo OK: {info.id}  (last_modified={info.last_modified})")
    except Exception as e:
        print(f"repo_info failed for {args.repo_id}: {e}", file=sys.stderr)
        print("If the repo does not exist, create it at "
              "https://huggingface.co/new-dataset then rerun.", file=sys.stderr)
        return 2

    plan = [(README_SRC, README_DST)]
    for local_rel, remote in FILES:
        local = ROOT / local_rel
        if not local.exists():
            print(f"missing {local} — skip")
            continue
        plan.append((local, remote))

    print("\n=== upload plan ===")
    total_mb = 0.0
    for local, remote in plan:
        mb = _size_mb(Path(local))
        total_mb += mb
        print(f"  {mb:>8.1f} MB  {local}  ->  {remote}")
    print(f"  {'-'*8}")
    print(f"  {total_mb:>8.1f} MB  total")

    if args.dry_run:
        print("\n[dry-run] not uploading.")
        return 0

    for local, remote in plan:
        mb = _size_mb(Path(local))
        print(f"\n[upload] {local} ({mb:.1f} MB) -> {args.repo_id}:{remote}")
        t0 = time.perf_counter()
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"upload {remote}",
        )
        dt = time.perf_counter() - t0
        mb_per_s = mb / dt if dt > 0 else float("inf")
        print(f"[upload] done in {dt:.1f}s  ({mb_per_s:.1f} MB/s)")

    print(f"\nall uploaded. See https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
