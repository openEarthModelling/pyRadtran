"""Pure-NumPy Mie scattering (Bohren-Huffman algorithm).

No scipy dependency — uses only numpy for all computations.
"""

import numpy as np


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

        # Asymmetry parameter recurrence (B&H Eq. 4.77)
        if n > 1:
            g_num += (
                (n - 1)
                * (n + 1)
                / n
                * (g_prev_a * an.conjugate() + g_prev_b * bn.conjugate()).real
            )
            g_num += (
                (2.0 * n - 1)
                / (n * (n - 1))
                * (g_prev_a * g_prev_b.conjugate()).real
            )

        g_prev_a = an
        g_prev_b = bn

        # Shift for next iteration
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = xi

    factor = 2.0 / (x * x)
    Qext *= factor
    Qsca *= factor
    Qback = factor * (abs(g_prev_a) ** 2 + abs(g_prev_b) ** 2)

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


def integrate_size_distribution(*args, **kwargs):
    """Stub for size-distribution integration (implemented in a later task)."""
    raise NotImplementedError("integrate_size_distribution is not yet implemented.")
