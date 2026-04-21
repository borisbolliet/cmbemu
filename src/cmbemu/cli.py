"""cmbemu command-line tool.

Usage:
    cmbemu download              # fetch pre-generated train+test from HF Hub
    cmbemu generate --n 10000 --seed 42 --out extra.npz
    cmbemu score-baseline        # run the ConstantPlanck baseline end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_download(args) -> int:
    from .data import download_data
    paths = download_data(size=args.size)
    for tag, p in paths.items():
        print(f"{tag}: {p}")
    return 0


def _cmd_generate(args) -> int:
    from .api import generate_data
    out = generate_data(n=args.n, seed=args.seed,
                        save_to=Path(args.out) if args.out else None)
    print(f"generated {out['params'].shape[0]} spectra "
          f"(seed={out['seed']}, cosmo_model={out['cosmo_model']})")
    if args.out:
        print(f"saved to {args.out}")
    return 0


def _cmd_score_baseline(args) -> int:
    from .api import get_score
    from .baselines import ConstantPlanck
    from .data import load_test

    test = load_test(size=args.size, local_dir=args.local_dir)
    emu = ConstantPlanck(lmax_cmb=test["lmax_cmb"], lmax_pp=test["lmax_pp"])
    result = get_score(emu, test=test,
                       measure_timing=not args.skip_timing,
                       n_timing_calls=args.n_timing)
    print(json.dumps(result, indent=2, default=float))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cmbemu")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("download", help="download pre-generated data")
    pd.add_argument("--size", default="full", choices=["full", "small"])
    pd.set_defaults(func=_cmd_download)

    pg = sub.add_parser("generate", help="generate more spectra via classy_szfast")
    pg.add_argument("--n", type=int, required=True)
    pg.add_argument("--seed", type=int, required=True)
    pg.add_argument("--out", default=None)
    pg.set_defaults(func=_cmd_generate)

    ps = sub.add_parser("score-baseline", help="run ConstantPlanck end-to-end")
    ps.add_argument("--size", default="small", choices=["full", "small"])
    ps.add_argument("--local-dir", default=None,
                    help="directory with local .npz files (skips HF fetch)")
    ps.add_argument("--n-timing", type=int, default=200)
    ps.add_argument("--skip-timing", action="store_true")
    ps.set_defaults(func=_cmd_score_baseline)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
