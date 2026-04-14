"""High-level convenience functions for common radiative transfer tasks."""

from __future__ import annotations

import math

import xarray as xr

from pyradtran.scene import Scene
from pyradtran.core.runner import Runner


def _airmass_to_sza(airmass: float) -> float:
    """Convert relative airmass to solar zenith angle (degrees).

    Uses the Kasten & Young (1989) formula inverted via bisection.
    """
    if airmass <= 1.0:
        return 0.0
    low, high = 0.0, 90.0
    for _ in range(50):
        mid = (low + high) / 2.0
        am = 1.0 / (math.cos(math.radians(mid)) + 0.50572 * (96.07995 - mid) ** (-1.6364))
        if am < airmass:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def run_solar_transmittance(
    airmass: float = 1.0,
    pwv: float = 5.0,
    ozone: float = 300.0,
    profile: str = "us",
    altitude: float | str = 0.0,
    wl_min: float = 250.0,
    wl_max: float = 1200.0,
    albedo: float = 0.0,
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Calculate solar spectral transmittance.

    Args:
        airmass: Relative airmass (1.0 = zenith).
        pwv: Precipitable water vapor in mm (mol_modify H2O).
        ozone: Ozone column in DU (mol_modify O3).
        profile: Atmospheric profile name (us, ms, mw, tp, ss, sw).
        altitude: Surface altitude in km, or a preset name ("LSST", "CTIO").
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        albedo: Surface albedo.
        streams: Number of DISORT streams.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with transmittance vs wavelength.
    """
    from pyradtran.presets import resolve_altitude

    sza = _airmass_to_sza(airmass)
    resolved_altitude = resolve_altitude(altitude)

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolved_altitude)
        .set_mol_modify("H2O", pwv, "MM")
        .set_mol_modify("O3", ozone, "DU")
        .set_source_solar(sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="disort", streams=streams)
        .set_output(quantity="transmittance", format="netcdf", quiet=True)
        .set_surface(albedo=albedo)
    )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_thermal_brightness(
    pwv: float = 10.0,
    profile: str = "ms",
    altitude: float = 0.0,
    wl_min: float = 2500.0,
    wl_max: float = 50000.0,
    sur_temperature: float | None = None,
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Calculate thermal infrared brightness temperature."""
    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=altitude)
        .set_mol_modify("H2O", pwv, "MM")
        .set_source_thermal()
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="disort", streams=streams)
        .set_output(quantity="brightness", format="netcdf", quiet=True)
    )

    if sur_temperature is not None:
        scene = scene.set_surface(sur_temperature=sur_temperature)

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)
