"""Readers for libRadtran OPAC ASCII ingredients.

OPAC ships only ingredients (no precomputed tables); libRadtran computes optics
at runtime via internal Mie. These parsers return pyRadtran data structures so
the existing MieSpecies path can fold OPAC species. Files live under
``<data_root>/aerosol/OPAC/``:

  - ``refractive_indices/<sp><rh>_refr.dat`` : 3-col (lambda_um n_real n_imag), 61 wl
  - ``size_distr.cfg``                       : 7-col (no rh Rmin Rmax Rmod Rho Sigma)
  - ``standard_aerosol_files/<preset>.dat``  : z(km) x per-species mass columns (g/m^3)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyradtran.models.aerosol_composite import RefractiveIndex, SizeDistribution

# size_distr.cfg "no" column -> species code (aerosol_species_names.dat order).
_OPAC_SPECIES_TO_INDEX = {
    "inso": 1,
    "waso": 2,
    "soot": 3,
    "ssam": 4,
    "sscm": 5,
    "minm": 6,
    "miam": 7,
    "micm": 8,
    "mitr": 9,
    "suso": 10,
}
_HYGROSCOPIC_SPECIES = frozenset({"waso", "ssam", "sscm", "suso"})
_RH_LEVELS = (0, 50, 70, 80, 90, 95, 98, 99)


def _opac_root(data_path: str | Path | None) -> Path:
    """Resolve the OPAC ingredient directory under the active data root."""
    if data_path is not None:
        return Path(data_path) / "aerosol" / "OPAC"
    from pyradtran.data import get_data_root

    return Path(get_data_root()) / "aerosol" / "OPAC"


def snap_rh(species: str, rh_pct: float) -> int:
    """Snap a requested RH to the nearest available OPAC level for ``species``.

    Non-hygroscopic species only have RH=0 (the ``00`` file); the request is
    ignored. Hygroscopic species snap to the nearest of the tabulated levels
    (ties round down to the lower level).
    """
    if species not in _OPAC_SPECIES_TO_INDEX:
        raise ValueError(f"Unknown OPAC species {species!r}")
    if species not in _HYGROSCOPIC_SPECIES:
        return 0
    return int(min(_RH_LEVELS, key=lambda lv: (abs(lv - rh_pct), lv)))


def read_opac_refractive_index(
    species: str, rh_pct: float = 50.0, *, data_path: str | Path | None = None
) -> RefractiveIndex:
    """Read ``<sp><rh>_refr.dat`` -> :class:`RefractiveIndex`.

    OPAC stores the imaginary part *negative* (its sign convention is
    ``m = n - i*kappa``); bhmie/Bohren-Huffman use ``Im(m) > 0`` for absorption,
    and :class:`RefractiveIndex` rejects ``k < 0``. We therefore return the
    absolute value, so ``RefractiveIndex.at()`` yields ``m = n + i*k`` (k>0).
    """
    rh = snap_rh(species, rh_pct)
    path = _opac_root(data_path) / "refractive_indices" / f"{species}{rh:02d}_refr.dat"
    arr = np.loadtxt(path, comments="#")
    return RefractiveIndex(
        wavelength_um=arr[:, 0].tolist(),
        n_real=arr[:, 1].tolist(),
        k_imag=np.abs(arr[:, 2]).tolist(),
    )


def read_opac_size_distribution(
    species: str, rh_pct: float = 50.0, *, data_path: str | Path | None = None
) -> tuple[SizeDistribution, float]:
    """Read the matching ``size_distr.cfg`` row.

    Columns: ``no rh Rmin Rmax Rmod[um] Rho Sigma``. ``Rho`` is treated as
    g/cm^3 (the values are physical densities; the cfg header label ``[g/m**3]``
    is misleading) and returned as kg/m^3. Returns
    ``(lognormal SizeDistribution, density_kg_m3)``.
    """
    rh = snap_rh(species, rh_pct)
    no = _OPAC_SPECIES_TO_INDEX[species]
    rows = np.loadtxt(_opac_root(data_path) / "size_distr.cfg", comments="#")
    match = rows[(rows[:, 0] == no) & (rows[:, 1] == rh)]
    if match.size == 0:
        raise ValueError(f"No size_distr.cfg row for {species!r} (no={no}) at rh={rh}")
    _no, _rh, _rmin, _rmax, rmod_um, rho_g_cm3, sigma = match[0]
    sd = SizeDistribution(
        kind="lognormal",
        params={"r_g_um": float(rmod_um), "sigma_g": float(sigma)},
    )
    return sd, float(rho_g_cm3) * 1000.0


def read_opac_profile_file(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse a standard aerosol profile file -> ``{species: (z_km, mass_g_m3)}``.

    The file is altitude (km) followed by one mass-concentration column (g/m^3)
    per OPAC species. A ``#`` header line lists the species codes (after a
    ``z(km)`` label); its species tokens map columns. If no header is found,
    the canonical 10-species order is assumed for the columns present.
    Altitude is returned in file order (standard presets are descending).
    """
    species_order = list(_OPAC_SPECIES_TO_INDEX)
    z_rows: list[float] = []
    col_rows: list[list[float]] = []
    header_species: list[str] | None = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if header_species is None:
                    tokens = [t for t in line.lstrip("#").split() if t in _OPAC_SPECIES_TO_INDEX]
                    if tokens:
                        header_species = tokens
                continue
            vals = [float(p) for p in line.split()]
            z_rows.append(vals[0])
            col_rows.append(vals[1:])
    z = np.asarray(z_rows, dtype=float)
    data = np.asarray(col_rows, dtype=float)
    names = header_species if header_species is not None else species_order[: data.shape[1]]
    return {sp: (z, data[:, j]) for j, sp in enumerate(names)}


def read_opac_preset_profile(
    name: str, *, data_path: str | Path | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read ``standard_aerosol_files/<name>.dat`` (wrapper over the file parser)."""
    return read_opac_profile_file(_opac_root(data_path) / "standard_aerosol_files" / f"{name}.dat")
