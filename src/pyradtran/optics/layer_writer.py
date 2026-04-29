"""Writer for libRadtran ``aerosol_file explicit`` format.

Produces:
    - master file: maps altitudes to .LAYER filenames
    - per-layer .LAYER files: wavelength_nm beta_ext_per_km ssa k_0 k_1 ...
    - NULL.LAYER: zero-optical-depth placeholder
"""

import hashlib
from pathlib import Path

import numpy as np


def _content_hash(
    wavelength_grid_um: list[float],
    altitude_grid_km: list[float],
    n_legendre: int,
    source_signatures: list[str],
) -> str:
    """Deterministic content hash for cache key."""
    data = (
        str(wavelength_grid_um)
        + str(altitude_grid_km)
        + str(n_legendre)
        + str(sorted(source_signatures))
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def write_explicit_aerosol(
    *,
    tau: np.ndarray,
    ssa: np.ndarray,
    g: np.ndarray,
    legendre_moments: np.ndarray,
    wavelength_um: np.ndarray,
    altitude_km: np.ndarray,
    output_dir: Path,
    source_signatures: list[str],
) -> Path:
    """Write explicit aerosol files for libRadtran.

    Args:
        tau: Optical depth per layer, shape (n_wl, n_layer).
        ssa: Single-scattering albedo, shape (n_wl, n_layer).
        g: Asymmetry parameter, shape (n_wl, n_layer).
        legendre_moments: Legendre expansion coefficients, shape (n_wl, n_layer, n_legendre).
        wavelength_um: Wavelength grid in um, shape (n_wl,).
        altitude_km: Altitude boundaries in km, strictly descending, shape (n_layer+1,).
        output_dir: Directory to write files.
        source_signatures: Strings identifying sources (for hashing).

    Returns:
        Path to the master file.
    """
    n_wl, n_layer = tau.shape
    n_legendre = legendre_moments.shape[2]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Content hash for cache/filename
    content_hash = _content_hash(
        wavelength_um.tolist(),
        altitude_km.tolist(),
        n_legendre,
        source_signatures,
    )
    prefix = f"scene_{content_hash}_layer"

    master_path = output_dir / f"scene_{content_hash}.master"

    # Check cache
    if master_path.exists():
        return master_path

    # Write NULL.LAYER
    null_path = output_dir / "NULL.LAYER"
    if not null_path.exists():
        null_vals = [0.0, 0.0, 1.0, 0.0] + [0.0] * (n_legendre - 1)
        with open(null_path, "w") as f:
            f.write(" ".join(f"{v:.6e}" for v in null_vals) + "\n")

    # Write per-layer .LAYER files
    layer_names = []
    for i_layer in range(n_layer):
        layer_name = f"{prefix}_{i_layer:02d}.LAYER"
        layer_path = output_dir / layer_name
        layer_names.append(layer_name)

        with open(layer_path, "w") as f:
            for i_wl in range(n_wl):
                wl_nm = wavelength_um[i_wl] * 1000.0
                # beta_ext per km = tau / dz_km
                dz_km = altitude_km[i_layer] - altitude_km[i_layer + 1]
                beta_ext_per_km = tau[i_wl, i_layer] / dz_km if dz_km > 0 else 0.0
                ssa_val = ssa[i_wl, i_layer]
                # Write k_0, k_1, ..., k_{n_legendre-1}
                moments = legendre_moments[i_wl, i_layer, :]
                vals = [wl_nm, beta_ext_per_km, ssa_val] + moments.tolist()
                f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")

    # Write master file
    with open(master_path, "w") as f:
        for i_layer in range(n_layer):
            z_top = altitude_km[i_layer]
            f.write(f"{z_top:.6f}  {layer_names[i_layer]}\n")
        # Bottom boundary -> NULL.LAYER
        z_bottom = altitude_km[-1]
        f.write(f"{z_bottom:.6f}  NULL.LAYER\n")

    return master_path
