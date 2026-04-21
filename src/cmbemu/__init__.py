"""cmbemu — CMB power-spectrum emulator competition.

Pre-generated training/test data, Wishart-likelihood scoring, and CPU
timing harness built on classy_szfast. Public API:

    import cmbemu as cec

    train = cec.load_train()
    test  = cec.load_test()
    extra = cec.generate_data(n=10_000, seed=42, save_to="extra.npz")

    class MyEmulator:
        def predict(self, params: dict) -> dict:
            return {"tt": ..., "te": ..., "ee": ..., "pp": ...}

    result = cec.get_score(MyEmulator())
"""
from __future__ import annotations

# Parameter box + cosmo_model setting
from .box import (
    BOX,
    COSMO_MODEL,
    F_SKY,
    LMAX_CMB,
    LMAX_PP,
    PARAM_NAMES,
    PLANCK_FIDUCIAL,
    box_bounds,
    params_array_to_dict,
    validate_in_box,
)

# LHC sampler
from .lhc import sample_lhc

# Data loading / regeneration (import api last — it pulls in generate + data)
from .data import download_data, load_test, load_train
from .api import generate_data, get_score

# Likelihood + scoring primitives (exposed for advanced users)
from .likelihood import chi2_diag, chi2_matrix
from .scoring import chi2_mae, combined_score

# Baselines
from .baselines import ConstantPlanck, NearestNeighbour

__version__ = "0.1.0a2"

__all__ = [
    "__version__",
    # box
    "BOX", "COSMO_MODEL", "F_SKY", "LMAX_CMB", "LMAX_PP",
    "PARAM_NAMES", "PLANCK_FIDUCIAL",
    "box_bounds", "params_array_to_dict", "validate_in_box",
    # sampling
    "sample_lhc",
    # data
    "download_data", "load_train", "load_test",
    "generate_data", "get_score",
    # scoring primitives
    "chi2_matrix", "chi2_diag", "chi2_mae", "combined_score",
    # baselines
    "ConstantPlanck", "NearestNeighbour",
]
