"""Validation plots for each stage of the pipeline. dpi=300 throughout."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .box import PARAM_NAMES, PLANCK_FIDUCIAL, box_bounds

DPI = 300
STYLE = {
    "label_size": 10,
    "title_size": 12,
    "tick_size": 8,
}


def _setup_axes(ax, xlabel=None, ylabel=None, title=None, logx=False, logy=False):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=STYLE["label_size"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=STYLE["label_size"])
    if title is not None:
        ax.set_title(title, fontsize=STYLE["title_size"])
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.tick_params(labelsize=STYLE["tick_size"])
    ax.grid(alpha=0.25, which="both", ls="--", lw=0.4)


def plot_lhc(params: np.ndarray, out_path: str | Path, title: str = "LHC") -> None:
    """Marginal histograms (diagonal) + pair scatter (lower triangle) of the LHC."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = box_bounds()
    d = len(PARAM_NAMES)
    fig, axes = plt.subplots(d, d, figsize=(2.3 * d, 2.3 * d))
    fid = np.array([PLANCK_FIDUCIAL[n] for n in PARAM_NAMES])
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(params[:, i], bins=40, color="steelblue", alpha=0.85,
                        edgecolor="none")
                ax.axvline(fid[i], color="crimson", lw=1.0, label="Planck fid")
                ax.axvline(lo[i], color="k", lw=0.7, ls="--")
                ax.axvline(hi[i], color="k", lw=0.7, ls="--")
                ax.set_yticks([])
            elif j < i:
                ax.scatter(params[:, j], params[:, i], s=1.5, alpha=0.35,
                           color="steelblue", edgecolors="none")
                ax.scatter([fid[j]], [fid[i]], s=30, color="crimson",
                           marker="x", lw=1.5, zorder=5)
                ax.set_xlim(lo[j], hi[j])
                ax.set_ylim(lo[i], hi[i])
            else:
                ax.axis("off")
            if j == 0 and i != 0:
                ax.set_ylabel(PARAM_NAMES[i], fontsize=STYLE["label_size"])
            if i == d - 1:
                ax.set_xlabel(PARAM_NAMES[j], fontsize=STYLE["label_size"])
            ax.tick_params(labelsize=STYLE["tick_size"] - 1)
    fig.suptitle(f"{title}   (n={len(params)})", y=0.995,
                 fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _dl_factor(ell):
    return ell * (ell + 1.0) / (2.0 * np.pi)


def _pp_factor(ell):
    return (ell * (ell + 1.0)) ** 2 / (2.0 * np.pi)


def _yvals(cls_arr, key, ell):
    if key == "pp":
        return _pp_factor(ell) * cls_arr
    return _dl_factor(ell) * cls_arr


def _yaxis_labels():
    return {
        "tt": r"$\ell(\ell+1) C_\ell^{TT}/2\pi$",
        "te": r"$\ell(\ell+1) C_\ell^{TE}/2\pi$",
        "ee": r"$\ell(\ell+1) C_\ell^{EE}/2\pi$",
        "pp": r"$[\ell(\ell+1)]^2 C_\ell^{\phi\phi}/2\pi$",
    }


def plot_spectra_random(
    cls: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Spectra",
    n_curves: int = 200,
    seed: int = 0,
) -> None:
    """Overlay a random subset of spectra on 4 panels (TT, TE, EE, PP)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = cls["tt"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_curves, N), replace=False)

    ell_cmb = np.arange(cls["tt"].shape[1])
    ell_pp = np.arange(cls["pp"].shape[1])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    cmap = plt.get_cmap("viridis")
    yl = _yaxis_labels()
    for k, i in enumerate(idx):
        color = cmap(k / max(1, len(idx) - 1))
        axes[0].plot(ell_cmb[2:], _yvals(cls["tt"][i, 2:], "tt", ell_cmb[2:]),
                     color=color, lw=0.4, alpha=0.5)
        axes[1].plot(ell_cmb[2:], _yvals(cls["te"][i, 2:], "te", ell_cmb[2:]),
                     color=color, lw=0.4, alpha=0.5)
        axes[2].plot(ell_cmb[2:], _yvals(cls["ee"][i, 2:], "ee", ell_cmb[2:]),
                     color=color, lw=0.4, alpha=0.5)
        axes[3].plot(ell_pp[2:], _yvals(cls["pp"][i, 2:], "pp", ell_pp[2:]),
                     color=color, lw=0.4, alpha=0.5)

    for ax, key in zip(axes, ("tt", "te", "ee", "pp")):
        _setup_axes(ax, xlabel=r"$\ell$", ylabel=yl[key],
                    title=key.upper(), logx=True,
                    logy=(key in ("tt", "ee", "pp")))
        if key == "te":
            ax.set_yscale("symlog", linthresh=1e-16)
    fig.suptitle(f"{title} — random subset ({len(idx)} of {N})",
                 fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spectra_envelope(
    cls: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Spectra envelope",
    fiducial_cls: dict[str, np.ndarray] | None = None,
) -> None:
    """Min/max envelope + 5-95 percentile band + median + optional fiducial.

    Gives a compact "what does the dataset look like" view at a glance.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ell_cmb = np.arange(cls["tt"].shape[1])
    ell_pp = np.arange(cls["pp"].shape[1])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    yl = _yaxis_labels()
    keys = ("tt", "te", "ee", "pp")
    for ax, key in zip(axes, keys):
        arr = cls[key].astype(np.float64)
        ell = ell_pp if key == "pp" else ell_cmb
        y_all = np.array([_yvals(arr[i], key, ell) for i in range(arr.shape[0])])
        # Full range shaded
        y_min = np.min(y_all, axis=0)
        y_max = np.max(y_all, axis=0)
        y_q05 = np.percentile(y_all, 5, axis=0)
        y_q95 = np.percentile(y_all, 95, axis=0)
        y_med = np.median(y_all, axis=0)

        ax.fill_between(ell[2:], y_min[2:], y_max[2:], color="steelblue",
                        alpha=0.18, label="min / max")
        ax.fill_between(ell[2:], y_q05[2:], y_q95[2:], color="steelblue",
                        alpha=0.38, label="5 / 95 percentile")
        ax.plot(ell[2:], y_med[2:], color="navy", lw=1.0, label="median")
        if fiducial_cls is not None:
            y_fid = _yvals(fiducial_cls[key][0], key, ell)
            ax.plot(ell[2:], y_fid[2:], color="crimson", lw=1.0, ls="--",
                    label="Planck fid")

        _setup_axes(ax, xlabel=r"$\ell$", ylabel=yl[key],
                    title=key.upper(), logx=True,
                    logy=(key in ("tt", "ee", "pp")))
        if key == "te":
            ax.set_yscale("symlog", linthresh=1e-16)
        ax.legend(loc="best", fontsize=STYLE["tick_size"], framealpha=0.85)

    fig.suptitle(f"{title} — N={cls['tt'].shape[0]}",
                 fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spectra_colored_by(
    cls: dict[str, np.ndarray],
    params: np.ndarray,
    color_param: str,
    out_path: str | Path,
    title: str | None = None,
    n_curves: int = 300,
    seed: int = 0,
    cmap_name: str = "plasma",
) -> None:
    """Overlay spectra colored by a specific cosmological parameter.

    Shows how that parameter modulates each spectrum — the emulator must
    capture this dependence.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if color_param not in PARAM_NAMES:
        raise ValueError(f"unknown param {color_param}")
    param_idx = PARAM_NAMES.index(color_param)
    pvals = params[:, param_idx]

    N = cls["tt"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_curves, N), replace=False)

    ell_cmb = np.arange(cls["tt"].shape[1])
    ell_pp = np.arange(cls["pp"].shape[1])

    vmin, vmax = pvals.min(), pvals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(18, 4.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.03])
    axes = [fig.add_subplot(gs[0, k]) for k in range(4)]
    cax = fig.add_subplot(gs[0, 4])

    yl = _yaxis_labels()
    keys = ("tt", "te", "ee", "pp")
    for k in idx:
        color = cmap(norm(pvals[k]))
        for ax, key in zip(axes, keys):
            arr = cls[key][k]
            ell = ell_pp if key == "pp" else ell_cmb
            ax.plot(ell[2:], _yvals(arr[2:], key, ell[2:]),
                    color=color, lw=0.5, alpha=0.65)

    for ax, key in zip(axes, keys):
        _setup_axes(ax, xlabel=r"$\ell$", ylabel=yl[key],
                    title=key.upper(), logx=True,
                    logy=(key in ("tt", "ee", "pp")))
        if key == "te":
            ax.set_yscale("symlog", linthresh=1e-16)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(color_param, fontsize=STYLE["label_size"])
    cbar.ax.tick_params(labelsize=STYLE["tick_size"])

    fig.suptitle(title or f"Spectra colored by {color_param} — {len(idx)} of {N}",
                 fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spectra_dispersion(
    cls: dict[str, np.ndarray],
    out_path: str | Path,
    title: str = "Fractional dispersion vs multipole",
    fiducial_cls: dict[str, np.ndarray] | None = None,
) -> None:
    """Std / |mean| across the dataset vs ell, per spectrum.

    Approximates "how much variation must the emulator capture as a function
    of ell". For TE the spectrum crosses zero so we use std / sqrt(<|D_l|>^2)
    to avoid divide-by-zero.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ell_cmb = np.arange(cls["tt"].shape[1])
    ell_pp = np.arange(cls["pp"].shape[1])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    keys = ("tt", "te", "ee", "pp")
    for ax, key in zip(axes, keys):
        arr = cls[key].astype(np.float64)
        ell = ell_pp if key == "pp" else ell_cmb
        y_all = np.array([_yvals(arr[i], key, ell) for i in range(arr.shape[0])])
        std = np.std(y_all, axis=0)
        if key == "te":
            # Normalize by RMS (TE crosses zero)
            rms = np.sqrt(np.mean(y_all ** 2, axis=0))
            disp = std / np.where(rms > 0, rms, 1.0)
            ylabel = r"$\sigma(D_\ell) / \mathrm{rms}(D_\ell)$"
        else:
            mu = np.mean(y_all, axis=0)
            disp = std / np.where(np.abs(mu) > 0, np.abs(mu), 1.0)
            ylabel = r"$\sigma(D_\ell) / |\langle D_\ell\rangle|$"
        ax.plot(ell[2:], disp[2:], color="darkblue", lw=0.9)
        _setup_axes(ax, xlabel=r"$\ell$", ylabel=ylabel,
                    title=key.upper(), logx=True, logy=False)
    fig.suptitle(f"{title} — N={cls['tt'].shape[0]}",
                 fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_chi2_distribution(
    chi2: np.ndarray,
    out_path: str | Path,
    title: str = "chi^2(i,j) distribution",
    cuts: tuple[float, ...] = (40.0, 1000.0, 1e5, 1e7),
) -> dict:
    """Histogram + CDF of off-diagonal chi^2 values, with cut thresholds."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = chi2.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = chi2[mask].astype(np.float64)
    finite = vals[np.isfinite(vals)]
    pos = finite[finite > 0]
    survivor_counts = {float(c): int((finite < c).sum()) for c in cuts}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: PDF
    bins = np.logspace(np.log10(max(pos.min(), 1e-3)),
                       np.log10(max(pos.max(), 10.0)), 80)
    axes[0].hist(pos, bins=bins, color="steelblue", alpha=0.85,
                 edgecolor="white", lw=0.2)
    _setup_axes(axes[0], xlabel=r"$\chi^2(i,j)$ off-diagonal",
                ylabel="count", title="PDF (histogram)",
                logx=True, logy=True)
    for c in cuts:
        axes[0].axvline(c, color="crimson", lw=0.8, ls="--", alpha=0.7)
        axes[0].text(c, axes[0].get_ylim()[1] * 0.4,
                     f" cut={c:g}\n n<cut={survivor_counts[c]}",
                     color="crimson", fontsize=7, rotation=90, va="top")

    # Panel 2: CDF
    sorted_vals = np.sort(pos)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf, color="steelblue", lw=1.0)
    _setup_axes(axes[1], xlabel=r"$\chi^2(i,j)$ off-diagonal",
                ylabel="CDF", title="CDF",
                logx=True, logy=False)
    axes[1].set_ylim(0, 1.02)
    for c in cuts:
        axes[1].axvline(c, color="crimson", lw=0.8, ls="--", alpha=0.7)
    # Mark quantiles
    for q in (0.5, 0.95):
        v = np.quantile(pos, q)
        axes[1].axhline(q, color="gray", lw=0.6, ls=":")
        axes[1].text(axes[1].get_xlim()[0] * 1.5, q + 0.01,
                     f"q{q:.2f} at χ²={v:.2e}", fontsize=7, color="gray")

    fig.suptitle(f"{title}   (N_off-diag = {len(finite)})",
                 fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return survivor_counts


def plot_score_diagnostic(
    chi2_true: np.ndarray,
    chi2_emu: np.ndarray,
    out_path: str | Path,
    title: str = "score diagnostic",
) -> None:
    """Scatter chi2_emu vs chi2_true (all pairs), with y=x reference, plus
    histogram of Delta chi^2.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = chi2_true.ravel().astype(np.float64)
    e = chi2_emu.ravel().astype(np.float64)
    ok = np.isfinite(t) & np.isfinite(e)
    t, e = t[ok], e[ok]
    delta = e - t

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(t, e, s=1.2, alpha=0.15, color="steelblue",
                    edgecolors="none")
    lim_lo = max(min(t.min(), e.min()), 1e-3)
    lim_hi = max(t.max(), e.max())
    axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="crimson", lw=1.0)
    _setup_axes(axes[0], xlabel=r"$\chi^2_{\rm true}(i,j)$",
                ylabel=r"$\chi^2_{\rm emu}(i,j)$",
                title="emu vs true", logx=True, logy=True)

    # Delta histogram
    absd = np.abs(delta)
    bins = np.logspace(np.log10(max(absd[absd > 0].min(), 1e-3)),
                       np.log10(max(absd.max(), 10.0)), 80)
    axes[1].hist(absd[absd > 0], bins=bins, color="darkorange", alpha=0.8,
                 edgecolor="white", lw=0.2)
    _setup_axes(axes[1], xlabel=r"$|\chi^2_{\rm emu} - \chi^2_{\rm true}|$",
                ylabel="count", title=r"$|\Delta \chi^2|$ distribution",
                logx=True, logy=True)
    axes[1].axvline(absd.mean(), color="crimson", lw=1.0, ls="--",
                    label=f"mean = {absd.mean():.2e}")
    axes[1].axvline(np.median(absd), color="navy", lw=1.0, ls=":",
                    label=f"median = {np.median(absd):.2e}")
    axes[1].legend(loc="best", fontsize=STYLE["tick_size"] + 1)

    fig.suptitle(title, fontsize=STYLE["title_size"] + 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
