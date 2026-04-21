"""Dataset I/O: save and load competition train/test bundles as .npz."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .box import COSMO_MODEL, LMAX_CMB, LMAX_PP, PARAM_NAMES, box_bounds


def save_dataset(
    path: str | Path,
    params: np.ndarray,
    cls: dict[str, np.ndarray],
    lmax_cmb: int = LMAX_CMB,
    lmax_pp: int = LMAX_PP,
    metadata: dict | None = None,
) -> None:
    """Save an emulator-competition dataset to a compressed .npz.

    Layout:
      params        float32 (N, 6)
      tt, te, ee    float32 (N, lmax_cmb + 1)
      pp            float32 (N, lmax_pp  + 1)
      param_names   str (6,)
      box_lo, box_hi float32 (6,)
      lmax_cmb, lmax_pp  scalar int
      classy_sz_version  str (if provided in metadata)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lo, hi = box_bounds()
    meta = metadata or {}

    np.savez_compressed(
        path,
        params=params.astype(np.float32),
        tt=cls["tt"].astype(np.float32),
        te=cls["te"].astype(np.float32),
        ee=cls["ee"].astype(np.float32),
        pp=cls["pp"].astype(np.float32),
        param_names=np.array(PARAM_NAMES),
        box_lo=lo.astype(np.float32),
        box_hi=hi.astype(np.float32),
        lmax_cmb=np.int32(lmax_cmb),
        lmax_pp=np.int32(lmax_pp),
        classy_sz_version=np.array(str(meta.get("classy_sz_version", "unknown"))),
        cosmo_model=np.int32(meta.get("cosmo_model", COSMO_MODEL)),
        seed=np.int64(meta.get("seed", -1)),
    )


def load_dataset(path: str | Path) -> dict:
    """Inverse of save_dataset — returns a dict with typed numpy arrays."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        out = {
            "params": z["params"],
            "tt": z["tt"],
            "te": z["te"],
            "ee": z["ee"],
            "pp": z["pp"],
            "param_names": tuple(str(n) for n in z["param_names"]),
            "box_lo": z["box_lo"],
            "box_hi": z["box_hi"],
            "lmax_cmb": int(z["lmax_cmb"]),
            "lmax_pp": int(z["lmax_pp"]),
            "classy_sz_version": str(z["classy_sz_version"]),
            "cosmo_model": int(z["cosmo_model"]) if "cosmo_model" in z.files else -1,
            "seed": int(z["seed"]),
        }
    return out
