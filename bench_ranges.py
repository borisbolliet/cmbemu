"""Probe classy_szfast across the proposed competition box and make diagnostic plots.

Outputs to plots/:
  - sweeps.png      : single-axis sweeps for each parameter (rows=param, cols=spectrum)
  - corners.png     : all 2^6=64 box corners overlaid per spectrum
  - residuals.png   : fractional residuals vs Planck for corners
"""
import itertools
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from classy_sz import Class as Class_sz

BOX = {
    "omega_b": (0.020, 0.025),
    "omega_cdm": (0.09, 0.15),
    "H0": (55.0, 85.0),
    "tau_reio": (0.03, 0.10),
    "ln10^{10}A_s": (2.7, 3.3),
    "n_s": (0.92, 1.02),
}
PLANCK = {
    "omega_b": 0.02242, "omega_cdm": 0.11933, "H0": 67.66,
    "tau_reio": 0.0561, "ln10^{10}A_s": 3.047, "n_s": 0.9665,
}
LMAX_CMB = 6000
LMAX_PP = 3000
N_SWEEP = 11

os.makedirs("plots", exist_ok=True)

csz = Class_sz()
csz.initialize_classy_szfast()


def cls(params):
    out = csz.get_cmb_cls(params_values_dict=params)
    return out


def dl_factor(ell):
    return ell * (ell + 1.0) / (2.0 * np.pi)


def pp_factor(ell):
    return (ell * (ell + 1.0)) ** 2 / (2.0 * np.pi)


# Warm up
fid = cls(PLANCK)
ell = fid["ell"]
mask_cmb = (ell >= 2) & (ell <= LMAX_CMB)
mask_pp = (ell >= 2) & (ell <= LMAX_PP)
ell_cmb = ell[mask_cmb]
ell_pp = ell[mask_pp]


def plot_spec(ax, ell_loc, arr, key, style=None, **kw):
    style = style or {}
    if key == "pp":
        y = pp_factor(ell_loc) * arr
    elif key == "te":
        y = dl_factor(ell_loc) * arr  # can be negative; use symlog
    else:
        y = dl_factor(ell_loc) * arr
    ax.plot(ell_loc, y, **style, **kw)


def check_finite_positive(arr, positive):
    bad = ~np.isfinite(arr)
    if bad.any():
        return f"{bad.sum()} non-finite"
    if positive and (arr <= 0).any():
        return f"{(arr <= 0).sum()} non-positive (min={arr.min():.3e})"
    return None


# ---------- Figure 1: single-axis sweeps ----------
param_names = list(BOX.keys())
keys = ("tt", "te", "ee", "pp")
fig, axes = plt.subplots(len(param_names), len(keys), figsize=(18, 2.4 * len(param_names)),
                         sharex=False)
print("=== single-axis sweeps ===")
for i, pname in enumerate(param_names):
    lo, hi = BOX[pname]
    values = np.linspace(lo, hi, N_SWEEP)
    cmap = plt.get_cmap("viridis")
    for v_idx, v in enumerate(values):
        p = dict(PLANCK)
        p[pname] = float(v)
        try:
            out = cls(p)
        except Exception as e:
            print(f"  {pname}={v:.4g}: EXCEPTION {type(e).__name__}: {e}")
            continue
        color = cmap(v_idx / (N_SWEEP - 1))
        for j, key in enumerate(keys):
            mask = mask_pp if key == "pp" else mask_cmb
            ell_loc = ell_pp if key == "pp" else ell_cmb
            arr = out[key][mask]
            plot_spec(axes[i, j], ell_loc, arr, key, color=color,
                      lw=0.9, alpha=0.85)
    for j, key in enumerate(keys):
        ax = axes[i, j]
        ax.set_xscale("log")
        if key in ("tt", "ee", "pp"):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-16)
        ax.set_xlim(2, LMAX_PP if key == "pp" else LMAX_CMB)
        ax.grid(alpha=0.2, which="both", ls="--")
        if i == 0:
            ax.set_title(key.upper())
        if j == 0:
            ax.set_ylabel(pname, fontsize=9)
        if i == len(param_names) - 1:
            ax.set_xlabel(r"$\ell$")
    # norm bar
    print(f"  {pname}: swept {N_SWEEP} values in [{lo}, {hi}]")
fig.suptitle("Single-axis sweeps (other params at Planck); colour = low→high", y=1.0)
fig.tight_layout()
fig.savefig("plots/sweeps.png", dpi=110)
plt.close(fig)
print("  wrote plots/sweeps.png")

# ---------- Figure 2: all 64 corners overlaid ----------
print("\n=== 64 corners ===")
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
issue_count = 0
dts = []
rng = np.random.default_rng(0)
# Hash each corner tuple to a stable color
for bits in itertools.product((0, 1), repeat=len(param_names)):
    params = {n: BOX[n][b] for n, b in zip(param_names, bits)}
    t0 = time.perf_counter()
    try:
        out = cls(params)
    except Exception as e:
        print(f"  EXCEPTION at {bits}: {e}")
        issue_count += 1
        continue
    dts.append(time.perf_counter() - t0)
    color = plt.cm.tab20(sum(bits) / 6)
    for j, key in enumerate(keys):
        mask = mask_pp if key == "pp" else mask_cmb
        ell_loc = ell_pp if key == "pp" else ell_cmb
        arr = out[key][mask]
        msg = check_finite_positive(arr, positive=(key in ("tt", "ee", "pp")))
        if msg:
            print(f"  {bits} {key}: {msg}")
            issue_count += 1
        plot_spec(axes[j], ell_loc, arr, key, color=color, lw=0.5, alpha=0.4)

# overlay fiducial Planck in red
for j, key in enumerate(keys):
    mask = mask_pp if key == "pp" else mask_cmb
    ell_loc = ell_pp if key == "pp" else ell_cmb
    plot_spec(axes[j], ell_loc, fid[key][mask], key, color="red", lw=1.5,
              label="Planck fid")
    axes[j].set_xscale("log")
    if key in ("tt", "ee", "pp"):
        axes[j].set_yscale("log")
    else:
        axes[j].set_yscale("symlog", linthresh=1e-16)
    axes[j].set_xlim(2, LMAX_PP if key == "pp" else LMAX_CMB)
    axes[j].set_title(key.upper())
    axes[j].set_xlabel(r"$\ell$")
    axes[j].grid(alpha=0.2, which="both", ls="--")
    axes[j].legend(loc="best", fontsize=8)
fig.suptitle(f"All 64 corners of competition box (red = Planck)")
fig.tight_layout()
fig.savefig("plots/corners.png", dpi=110)
plt.close(fig)
print(f"  wrote plots/corners.png; {issue_count} issues; per-call {np.mean(dts)*1000:.1f} ms")

# ---------- Figure 3: fractional residual vs Planck, corners ----------
print("\n=== corner residuals vs Planck ===")
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for bits in itertools.product((0, 1), repeat=len(param_names)):
    params = {n: BOX[n][b] for n, b in zip(param_names, bits)}
    try:
        out = cls(params)
    except Exception:
        continue
    color = plt.cm.tab20(sum(bits) / 6)
    for j, key in enumerate(keys):
        mask = mask_pp if key == "pp" else mask_cmb
        ell_loc = ell_pp if key == "pp" else ell_cmb
        num = out[key][mask]
        den = fid[key][mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            res = num / den - 1.0
        axes[j].plot(ell_loc, res, color=color, lw=0.5, alpha=0.4)
for j, key in enumerate(keys):
    axes[j].axhline(0, color="k", lw=0.8)
    axes[j].set_xscale("log")
    axes[j].set_xlim(2, LMAX_PP if key == "pp" else LMAX_CMB)
    axes[j].set_title(f"{key.upper()} / Planck - 1")
    axes[j].set_xlabel(r"$\ell$")
    axes[j].grid(alpha=0.2, which="both", ls="--")
fig.tight_layout()
fig.savefig("plots/residuals.png", dpi=110)
plt.close(fig)
print("  wrote plots/residuals.png")

print("\nDone.")
