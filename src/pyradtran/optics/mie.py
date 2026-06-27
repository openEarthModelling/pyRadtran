"""Pure-NumPy Mie scattering (Bohren-Huffman algorithm).

No scipy dependency — uses only numpy for all computations.
"""

from dataclasses import dataclass

import numpy as np

# numpy >=2.0 removed np.trapz in favor of np.trapezoid; import whichever exists.
# The eager getattr(np, "trapezoid", np.trapz) default would crash on numpy 2.x.
try:
    from numpy import trapezoid as _trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapz  # type: ignore[attr-defined]

# Import here to avoid circular import: mie.py uses SizeDistribution
# from this module, and aerosol_composite.py imports integrate_size_distribution.
from pyradtran.models.aerosol_composite import (
    IntegrationConfig,
    SizeDistribution,
)


def bhmie(x: float, m: complex, n_angles: int = 0) -> dict:
    """Compute Mie scattering for a single homogeneous sphere.

    Args:
        x: Size parameter ``2*pi*r/lambda``.
        m: Complex refractive index relative to surrounding medium.
        n_angles: Number of equally-spaced scattering angles between 0 and 180°
            for phase-function output. 0 = skip angular computation.

    Returns:
        Dictionary with keys:
        - ``Qext``: Extinction efficiency.
        - ``Qsca``: Scattering efficiency.
        - ``Qback``: Backscatter efficiency.
        - ``g``: Asymmetry parameter ``<cos(theta)>``.
        - ``S1``, ``S2`` (if n_angles > 0): Complex amplitude functions.
        - ``angles_deg`` (if n_angles > 0): Angle grid in degrees.

    Reference:
        Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*,
        Wiley, 1983.  Fortran code bhmie.f translated to Python/NumPy.
    """
    if x <= 0:
        return {
            "Qext": 0.0,
            "Qsca": 0.0,
            "Qback": 0.0,
            "g": 0.0,
        }

    # Series termination criterion
    nstop = int(x + 4.0 * x**0.3333 + 2.0)
    nmx = int(max(nstop, abs(m) * x)) + 15

    # Downward recursion for Dn(z) where z = m*x
    z = m * x
    D = np.zeros(nmx, dtype=complex)
    for n in range(nmx - 1, 0, -1):
        denom = (n + 1) / z + D[n]
        if abs(denom) < 1e-30:
            denom = 1e-30
        D[n - 1] = (n + 1) / z - 1.0 / denom

    # Riccati-Bessel functions via upward recurrence
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 = np.cos(x)
    xi1 = complex(psi1, -chi1)

    Qsca = 0.0
    Qext = 0.0
    g_prev_a = 0.0
    g_prev_b = 0.0
    g_num = 0.0
    S_back = 0.0j

    for n in range(1, nstop + 1):
        dn = n / x
        psi = (2.0 * n - 1.0) / x * psi1 - psi0
        chi = (2.0 * n - 1.0) / x * chi1 - chi0
        xi = complex(psi, -chi)

        an_num = (D[n - 1] / m + dn) * psi - psi1
        an_den = (D[n - 1] / m + dn) * xi - xi1
        an = an_num / an_den

        bn_num = (m * D[n - 1] + dn) * psi - psi1
        bn_den = (m * D[n - 1] + dn) * xi - xi1
        bn = bn_num / bn_den

        Qext += (2.0 * n + 1.0) * (an.real + bn.real)
        an_abs2 = abs(an) ** 2
        bn_abs2 = abs(bn) ** 2
        Qsca += (2.0 * n + 1.0) * (an_abs2 + bn_abs2)

        # Backscatter amplitude accumulation (B&H Eq. 4.76)
        if n == 1:
            S_back = 0.0j
        S_back += (2.0 * n + 1.0) * ((-1.0) ** n) * (an - bn)

        # Asymmetry parameter recurrence (B&H Eq. 4.77)
        if n > 1:
            g_num += (
                (n - 1) * (n + 1) / n * (g_prev_a * an.conjugate() + g_prev_b * bn.conjugate()).real
            )
            g_num += (2.0 * n - 1) / (n * (n - 1)) * (g_prev_a * g_prev_b.conjugate()).real

        g_prev_a = an
        g_prev_b = bn

        # Shift for next iteration
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = xi

    factor = 2.0 / (x * x)
    Qext *= factor
    Qsca *= factor
    Qback = factor * abs(S_back) ** 2

    if Qsca > 0:
        g = (4.0 / (x * x * Qsca)) * g_num
        # Clamp to physical range (numerical noise)
        g = max(-1.0, min(1.0, g))
    else:
        g = 0.0

    result = {
        "Qext": Qext,
        "Qsca": Qsca,
        "Qback": Qback,
        "g": g,
    }

    if n_angles > 0:
        # Compute S1 and S2 on angular grid
        angles_deg = np.linspace(0.0, 180.0, n_angles)
        mu = np.cos(np.radians(angles_deg))
        S1 = np.zeros(n_angles, dtype=complex)
        S2 = np.zeros(n_angles, dtype=complex)

        psi0 = np.cos(x)
        psi1 = np.sin(x)
        chi0 = -np.sin(x)
        chi1 = np.cos(x)
        xi1 = complex(psi1, -chi1)

        pi_n = np.zeros(n_angles)
        tau_n = np.zeros(n_angles)
        pi_nm1 = np.zeros(n_angles)
        pi_nm2 = np.zeros(n_angles)

        for n in range(1, nstop + 1):
            dn = n / x
            psi = (2.0 * n - 1.0) / x * psi1 - psi0
            chi = (2.0 * n - 1.0) / x * chi1 - chi0
            xi = complex(psi, -chi)

            an_num = (D[n - 1] / m + dn) * psi - psi1
            an_den = (D[n - 1] / m + dn) * xi - xi1
            an = an_num / an_den

            bn_num = (m * D[n - 1] + dn) * psi - psi1
            bn_den = (m * D[n - 1] + dn) * xi - xi1
            bn = bn_num / bn_den

            if n == 1:
                pi_n = np.ones(n_angles)
                tau_n = mu * pi_n
            else:
                pi_n = ((2.0 * n - 1.0) / (n - 1.0)) * mu * pi_nm1 - (n / (n - 1.0)) * pi_nm2
                tau_n = n * mu * pi_n - (n + 1.0) * pi_nm1

            S1 += (2.0 * n + 1.0) / (n * (n + 1.0)) * (an * pi_n + bn * tau_n)
            S2 += (2.0 * n + 1.0) / (n * (n + 1.0)) * (an * tau_n + bn * pi_n)

            pi_nm2 = pi_nm1
            pi_nm1 = pi_n
            psi0, psi1 = psi1, psi
            chi0, chi1 = chi1, chi
            xi1 = xi

        result["S1"] = S1
        result["S2"] = S2
        result["angles_deg"] = angles_deg

    return result


def phase_function_to_legendre(
    s1: np.ndarray, s2: np.ndarray, angles_deg: np.ndarray, n_legendre: int
) -> np.ndarray:
    """Project the unpolarised Mie phase function onto Legendre polynomials.

    ``P(mu) ∝ |S1|^2 + |S2|^2``, normalised so ``(1/2) ∫_{-1}^{1} P(mu) dmu = 1``.
    Returns ``beta_l = (1/2) ∫_{-1}^{1} P(mu) P_l(mu) dmu`` for ``l = 0..n_legendre-1``
    — the PMOM / ``g_l`` form (``beta_0 = 1``, ``beta_1`` = asymmetry parameter).
    Pure numpy (no scipy): Legendre polynomials via the Bonnet recurrence.
    """
    mu = np.cos(np.radians(np.asarray(angles_deg, dtype=float)))
    raw = np.abs(s1) ** 2 + np.abs(s2) ** 2

    order = np.argsort(mu)  # integrate over ascending mu
    mu, raw = mu[order], raw[order]

    norm = 0.5 * _trapz(raw, mu)
    if norm <= 0.0:
        out = np.zeros(n_legendre)
        out[0] = 1.0
        return out
    p_norm = raw / norm

    out = np.zeros(n_legendre)
    p_lm1 = np.ones_like(mu)  # P_{l-1}, seeded for the recurrence
    p_lm2 = np.ones_like(mu)  # P_{l-2}
    for el in range(n_legendre):
        if el == 0:
            p_l = np.ones_like(mu)
        elif el == 1:
            p_l = mu.copy()
        else:
            p_l = ((2 * el - 1) * mu * p_lm1 - (el - 1) * p_lm2) / el
        out[el] = 0.5 * _trapz(p_norm * p_l, mu)
        p_lm2 = p_lm1
        p_lm1 = p_l
    return out


@dataclass
class _SpeciesOptics:
    """Internal dataclass for mass-normalized intensive properties."""

    beta_ext_per_mass: np.ndarray
    ssa: np.ndarray
    g: np.ndarray
    legendre_moments: np.ndarray | None = None


def _mass_per_particle_avg(r_grid_um: np.ndarray, dn_dr: np.ndarray, rho_kg_m3: float) -> float:
    """Average particle mass: ρ * ∫ (4/3)πr³ n(r) dr."""
    r_m = r_grid_um * 1e-6
    volume = (4.0 / 3.0) * np.pi * r_m**3
    return rho_kg_m3 * _trapz(volume * dn_dr, r_m)


def integrate_size_distribution(
    *,
    wavelength_um: list[float],
    radius_um: list[float],
    Qext: np.ndarray,
    Qsca: np.ndarray,
    g: np.ndarray,
    legendre_moments: np.ndarray | None,
    size_distribution: SizeDistribution,
    particle_density_kg_m3: float,
    config: IntegrationConfig,
    n_legendre: int = 32,
) -> _SpeciesOptics:
    """Integrate Q-factors over size distribution to get intensive species optics."""
    n_wl = len(wavelength_um)
    r_sparse = np.asarray(radius_um)

    r_dense = np.logspace(
        np.log10(max(config.radius_min_um, r_sparse[0] * 0.1)),
        np.log10(min(config.radius_max_um, r_sparse[-1] * 10.0)),
        config.n_radius_grid,
    )
    r_dense = np.clip(r_dense, config.radius_min_um, config.radius_max_um)

    dn_dr = size_distribution.evaluate(r_dense)

    if len(r_sparse) == 1:
        Qext_dense = np.full((n_wl, config.n_radius_grid), Qext[0, 0])
        Qsca_dense = np.full((n_wl, config.n_radius_grid), Qsca[0, 0])
        g_dense = np.full((n_wl, config.n_radius_grid), g[0, 0])
        if legendre_moments is not None:
            n_mom = legendre_moments.shape[2]
            kl_dense = np.full(
                (n_wl, config.n_radius_grid, n_mom),
                legendre_moments[0, 0, :],
            )
        else:
            kl_dense = None
    else:
        log_r_sparse = np.log(r_sparse)
        log_r_dense = np.log(r_dense)

        Qext_dense = np.zeros((n_wl, config.n_radius_grid))
        Qsca_dense = np.zeros((n_wl, config.n_radius_grid))
        g_dense = np.zeros((n_wl, config.n_radius_grid))

        for i_wl in range(n_wl):
            log_Qext = np.log(np.clip(Qext[i_wl, :], 1e-30, None))
            Qext_dense[i_wl, :] = np.exp(
                np.interp(log_r_dense, log_r_sparse, log_Qext, left=log_Qext[0], right=log_Qext[-1])
            )
            log_Qsca = np.log(np.clip(Qsca[i_wl, :], 1e-30, None))
            Qsca_dense[i_wl, :] = np.exp(
                np.interp(log_r_dense, log_r_sparse, log_Qsca, left=log_Qsca[0], right=log_Qsca[-1])
            )
            g_dense[i_wl, :] = np.interp(
                r_dense,
                r_sparse,
                g[i_wl, :],
                left=g[i_wl, 0],
                right=g[i_wl, -1],
            )

        if legendre_moments is not None:
            n_mom = legendre_moments.shape[2]
            kl_dense = np.zeros((n_wl, config.n_radius_grid, n_mom))
            for i_wl in range(n_wl):
                for l in range(n_mom):
                    kl_dense[i_wl, :, l] = np.interp(
                        r_dense,
                        r_sparse,
                        legendre_moments[i_wl, :, l],
                        left=legendre_moments[i_wl, 0, l],
                        right=legendre_moments[i_wl, -1, l],
                    )
        else:
            kl_dense = None

    r_m = r_dense * 1e-6
    area = np.pi * r_m**2

    beta_ext_per_mass = np.zeros(n_wl)
    ssa = np.zeros(n_wl)
    g = np.zeros(n_wl)

    m_particle_avg = _mass_per_particle_avg(r_dense, dn_dr, particle_density_kg_m3)

    for i_wl in range(n_wl):
        integrand_ext = Qext_dense[i_wl, :] * area * dn_dr
        integrand_sca = Qsca_dense[i_wl, :] * area * dn_dr
        integrand_g = g_dense[i_wl, :] * Qsca_dense[i_wl, :] * area * dn_dr

        Iext = _trapz(integrand_ext, r_m)
        Isca = _trapz(integrand_sca, r_m)
        Ig = _trapz(integrand_g, r_m)

        beta_ext_per_mass[i_wl] = Iext / m_particle_avg if m_particle_avg > 0 else 0.0
        ssa[i_wl] = Isca / Iext if Iext > 0 else 0.0
        g[i_wl] = Ig / Isca if Isca > 0 else 0.0

    if kl_dense is not None:
        n_mom = kl_dense.shape[2]
        legendre_moments = np.zeros((n_wl, n_mom))
        for i_wl in range(n_wl):
            # Recompute Isca for this wavelength
            integrand_sca = Qsca_dense[i_wl, :] * area * dn_dr
            Isca_wl = _trapz(integrand_sca, r_m)
            for l in range(n_mom):
                integrand_kl = kl_dense[i_wl, :, l] * Qsca_dense[i_wl, :] * area * dn_dr
                Ikl = _trapz(integrand_kl, r_m)
                legendre_moments[i_wl, l] = Ikl / Isca_wl if Isca_wl > 0 else 0.0
    else:
        # Compute Henyey-Greenstein Legendre moments from integrated g
        legendre_moments = np.zeros((n_wl, n_legendre))
        l_vals = np.arange(n_legendre)
        for i_wl in range(n_wl):
            legendre_moments[i_wl, :] = g[i_wl] ** l_vals

    return _SpeciesOptics(
        beta_ext_per_mass=beta_ext_per_mass,
        ssa=ssa,
        g=g,
        legendre_moments=legendre_moments,
    )
