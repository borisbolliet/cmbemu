"""Competition parameter box. Single source of truth for ranges and defaults."""
from __future__ import annotations

import numpy as np

PARAM_NAMES: tuple[str, ...] = (
    "omega_b",
    "omega_cdm",
    "H0",
    "tau_reio",
    "ln10^{10}A_s",
    "n_s",
)

BOX: dict[str, tuple[float, float]] = {
    "omega_b":      (0.020, 0.025),
    "omega_cdm":    (0.09,  0.15),
    "H0":           (55.0,  85.0),
    "tau_reio":     (0.03,  0.10),
    "ln10^{10}A_s": (2.7,   3.3),
    "n_s":          (0.92,  1.02),
}

PLANCK_FIDUCIAL: dict[str, float] = {
    "omega_b":      0.02242,
    "omega_cdm":    0.11933,
    "H0":           67.66,
    "tau_reio":     0.0561,
    "ln10^{10}A_s": 3.047,
    "n_s":          0.9665,
}

LMAX_CMB = 6000
LMAX_PP = 3000
F_SKY = 1.0

# classy_szfast emulator backend. Index into cosmo_model_list in
# classy_szfast/emulators_meta_data.py:
#   0='lcdm', 1='mnu', 2='neff', 3='wcdm', 4='ede', 5='mnu-3states', 6='ede-v2'.
# We use 'ede-v2' because it is the newest emulator stack and has slightly
# broader n_s validity than the plain 'lcdm' one. The EDE-specific extra
# parameters (fEDE, log10z_c, thetai_scf, r, neutrino params) fall back to
# their 'default' entries in emulator_dict, producing near-LCDM spectra.
COSMO_MODEL = 6


def box_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) arrays in PARAM_NAMES order."""
    lo = np.array([BOX[n][0] for n in PARAM_NAMES], dtype=np.float64)
    hi = np.array([BOX[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lo, hi


def params_array_to_dict(params: np.ndarray) -> dict[str, float]:
    """Convert a 1-D array (shape (6,)) to a classy_sz-ready dict."""
    return {name: float(params[i]) for i, name in enumerate(PARAM_NAMES)}


def validate_in_box(params: np.ndarray) -> None:
    lo, hi = box_bounds()
    if params.shape[-1] != len(PARAM_NAMES):
        raise ValueError(f"expected last dim {len(PARAM_NAMES)}, got {params.shape}")
    bad = (params < lo) | (params > hi)
    if bad.any():
        idx = np.where(bad.any(axis=tuple(range(params.ndim - 1))))[0]
        raise ValueError(f"params out of box on axes {idx.tolist()}")
