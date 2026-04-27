"""External mixing rules for aerosol optical properties."""

import numpy as np


def _fill_hg_moments(g: np.ndarray, n_legendre: int) -> np.ndarray:
    """Fill Legendre moments with Henyey-Greenstein expansion.

    Args:
        g: Asymmetry parameter, shape (n_wl, n_layer).
        n_legendre: Number of moments.

    Returns:
        Array of shape (n_wl, n_layer, n_legendre).
    """
    n_wl, n_layer = g.shape
    moments = np.zeros((n_wl, n_layer, n_legendre))
    for l in range(n_legendre):
        moments[:, :, l] = (2 * l + 1) * g**l
    return moments


def _normalize_moments(
    moments: np.ndarray | None, g: np.ndarray, n_legendre: int
) -> np.ndarray:
    """Ensure moments have shape (n_wl, n_layer, n_legendre).

    - If None: H-G fill from g.
    - If shorter: zero-pad.
    - If longer: truncate.
    """
    n_wl, n_layer = g.shape
    result = np.zeros((n_wl, n_layer, n_legendre))

    if moments is None:
        return _fill_hg_moments(g, n_legendre)

    n_mom = moments.shape[2] if moments.ndim == 3 else 0
    if n_mom == 0:
        return _fill_hg_moments(g, n_legendre)

    n_copy = min(n_mom, n_legendre)
    result[:, :, :n_copy] = moments[:, :, :n_copy]

    return result


def combine_sources(
    sources: list,
    n_legendre: int,
) -> dict:
    """Externally mix N aerosol sources into a single LayerOptics-like dict.

    Args:
        sources: List of LayerOptics objects.
        n_legendre: Number of Legendre moments.

    Returns:
        Dict with keys ``tau``, ``ssa``, ``g``, ``legendre_moments``.
        All arrays have shape (n_wl, n_layer, ...).
    """
    if not sources:
        raise ValueError("At least one source required for mixing")

    n_wl, n_layer = sources[0].tau.shape

    tau_total = np.zeros((n_wl, n_layer))
    tw_total = np.zeros((n_wl, n_layer))
    g_num = np.zeros((n_wl, n_layer))
    kl_num = np.zeros((n_wl, n_layer, n_legendre))

    for src in sources:
        tau = src.tau
        ssa = src.ssa
        g = src.g
        moments = _normalize_moments(src.legendre_moments, g, n_legendre)

        tw = tau * ssa
        tau_total += tau
        tw_total += tw
        g_num += tw * g
        kl_num += tw[:, :, np.newaxis] * moments

    # Compute mixed properties
    with np.errstate(divide="ignore", invalid="ignore"):
        ssa_total = np.where(tau_total > 0, tw_total / tau_total, 0.0)
        g_total = np.where(tw_total > 0, g_num / tw_total, 0.0)
        kl_total = np.where(
            tw_total[:, :, np.newaxis] > 0,
            kl_num / tw_total[:, :, np.newaxis],
            0.0,
        )

    # Clamp ssa near 1.0 for solver stability
    ssa_total = np.minimum(ssa_total, 1.0 - 1e-9)

    # k_0 is always 1
    kl_total[:, :, 0] = 1.0

    return {
        "tau": tau_total,
        "ssa": ssa_total,
        "g": g_total,
        "legendre_moments": kl_total,
    }
