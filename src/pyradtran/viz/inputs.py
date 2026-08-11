"""Input-diagnostic plots: size distributions, phase functions, Legendre decay.

Pure data-in -> fig-out. No Runner, no libRadtran. Phase functions use
monodisperse Mie at the characteristic radius (lognormal r_g / monodisperse
radius_um) — an approximation showing the characteristic angular structure.
For exact polydisperse moments, use plot_legendre_decay.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from numpy import trapezoid as _trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapz  # type: ignore[attr-defined]

from pyradtran.viz._style import get_palette, require_mpl, save, set_theme


def _ensure_axes(ax=None):
    require_mpl()
    import matplotlib.pyplot as plt

    set_theme()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _characteristic_radius_um(sd) -> float:
    params = sd.params
    if sd.kind == "lognormal":
        return float(params["r_g_um"])
    if sd.kind == "monodisperse":
        return float(params["radius_um"])
    if sd.kind == "modified_gamma":
        return float(params["r_c_um"])
    raise NotImplementedError(
        f"plot_phase_functions does not support kind={sd.kind!r} "
        "(no single characteristic radius)."
    )


def plot_size_distributions(
    blocks: dict,
    n_points: int = 200,
    r_min_um: float = 1e-3,
    r_max_um: float = 50.0,
    ax=None,
    save_path=None,
):
    """dn/dlog10(r) per block on a log-radius axis.

    Args:
        blocks: mapping ``{name: SizeDistribution}``.
    """
    fig, ax = _ensure_axes(ax)
    r = np.logspace(np.log10(r_min_um), np.log10(r_max_um), n_points)
    colors = get_palette(len(blocks))
    for color, (name, sd) in zip(colors, blocks.items(), strict=True):
        dn = sd.evaluate(r)
        dn_dlogr = dn * r * np.log(10)  # dn/dr -> dn/dlog10(r)
        ax.plot(r, dn_dlogr, color=color, label=name, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Radius (µm)")
    ax.set_ylabel("dn/dlog₁₀(r)  (per µm³)")
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax


def plot_phase_functions(
    blocks: dict,
    wavelength_um: float = 0.55,
    n_angles: int = 181,
    ax=None,
    save_path=None,
):
    """Normalized phase function P(θ) per block at the characteristic radius.

    Args:
        blocks: mapping ``{name: (RefractiveIndex, SizeDistribution)}``.
        wavelength_um: wavelength at which to evaluate refractive index + Mie.
    """
    from pyradtran.optics.mie import bhmie

    fig, ax = _ensure_axes(ax)
    angles = np.linspace(0.0, 180.0, n_angles)
    mu = np.cos(np.deg2rad(angles))
    colors = get_palette(len(blocks))
    for color, (name, (ri, sd)) in zip(colors, blocks.items(), strict=True):
        m = complex(ri.at(np.array([wavelength_um]))[0])
        r_um = _characteristic_radius_um(sd)
        x = 2.0 * np.pi * r_um / wavelength_um
        res = bhmie(x, m, n_angles=n_angles)
        intensity = np.abs(res["S1"]) ** 2 + np.abs(res["S2"]) ** 2
        # Normalize so integral over the sphere = 1: 2*pi * trapz(P, mu) = 1.
        norm = 2.0 * np.pi * _trapz(intensity, mu)
        P = intensity / norm if norm > 0 else intensity
        ax.semilogy(angles, P, color=color, label=name, linewidth=1.5)
    ax.set_xlabel("Scattering angle θ (°)")
    ax.set_ylabel("Phase function P(θ)  (sr⁻¹)")
    ax.set_xlim(0.0, 180.0)
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax


def plot_legendre_decay(
    comp,
    wavelength_um: float = 0.55,
    n_legendre: int = 32,
    altitude_grid_km=(10.0, 0.0),
    ax=None,
    save_path=None,
):
    """Column-tau-weighted Legendre moments β_l vs l at one wavelength.

    Shows the anisotropy / how many moments carry weight (relates to DISORT
    streams needed). Uses ``comp.evaluate`` -> ``LayerOptics.legendre_moments``.
    """
    wl = np.asarray([wavelength_um], dtype=float)
    z = np.asarray(altitude_grid_km, dtype=float)
    lo = comp.evaluate(wl_um=wl, z_km=z, n_legendre=n_legendre)
    moments = np.asarray(lo.legendre_moments)  # (wl, layer, n_moments) or (wl, n_moments)
    tau = np.asarray(lo.tau)  # (wl, layer)
    if moments.ndim == 3:  # (wl, layer, n_moments)
        per_layer = moments[0]  # (layer, n_moments)
        w = tau[0]  # (layer,)
        wsum = w.sum()
        beta = (per_layer * w[:, None]).sum(axis=0) / wsum if wsum > 0 else per_layer.mean(axis=0)
    elif moments.ndim == 2:  # (wl, n_moments) — already column
        beta = moments[0]
    else:
        beta = np.ravel(moments)[:n_legendre]
    ls = np.arange(beta.size)

    fig, ax = _ensure_axes(ax)
    ax.stem(ls, beta, basefmt=" ")
    ax.set_xlabel("Legendre moment index l")
    ax.set_ylabel("βₗ (tau-weighted column)")
    ax.set_title(f"Phase-function moments @ {wavelength_um} µm")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax
