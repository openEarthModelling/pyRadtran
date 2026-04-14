"""High-level convenience functions for common radiative transfer tasks."""

from __future__ import annotations

import math

import xarray as xr

from pyradtran.core.runner import Runner
from pyradtran.scene import Scene


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


def run_solar_radiance(
    sza: float = 30.0,
    airmass: float | None = None,
    profile: str = "us",
    altitude: float | str = 0.0,
    pwv: float = 5.0,
    ozone: float = 300.0,
    wl_min: float = 250.0,
    wl_max: float = 2500.0,
    albedo: float = 0.2,
    aerosol_default: bool = False,
    aerosol_tau: float | None = None,
    aerosol_angstrom: float | None = None,
    streams: int = 16,
    solver: str = "disort",
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Calculate solar spectral radiance (direct + diffuse irradiance).

    Args:
        sza: Solar zenith angle in degrees. Ignored if airmass is set.
        airmass: Relative airmass. Overrides sza if provided.
        profile: Atmospheric profile name.
        altitude: Surface altitude in km, or a preset name.
        pwv: Precipitable water vapor in mm.
        ozone: Ozone column in DU.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        albedo: Surface albedo.
        aerosol_default: Enable default Shettle aerosol.
        aerosol_tau: AOD at 550 nm (sets tau via angstrom).
        aerosol_angstrom: Angstrom exponent (requires aerosol_tau).
        streams: Number of DISORT streams.
        solver: RTE solver name.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with irradiance vs wavelength at TOA and surface.
    """
    from pyradtran.presets import resolve_altitude

    if airmass is not None:
        sza = _airmass_to_sza(airmass)
    resolved_altitude = resolve_altitude(altitude)

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolved_altitude)
        .set_mol_modify("H2O", pwv, "MM")
        .set_mol_modify("O3", ozone, "DU")
        .set_source_solar(sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method=solver, streams=streams)
        .set_output(
            quantities=["lambda", "edir", "edn", "eup"],
            format="netcdf",
            quiet=True,
            zout=[0, 100],
        )
        .set_surface(albedo=albedo)
    )

    if aerosol_default or aerosol_tau is not None:
        aerosol_kwargs: dict = {"default": True}
        if aerosol_tau is not None and aerosol_angstrom is not None:
            beta = aerosol_tau * (0.55 ** (-aerosol_angstrom)) * 1e-3
            aerosol_kwargs["angstrom_alpha"] = aerosol_angstrom
            aerosol_kwargs["angstrom_beta"] = beta
        scene = scene.set_aerosol(**aerosol_kwargs)

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_with_aerosol(
    aerosol_file_type: str = "explicit",
    aerosol_file_path: str | None = None,
    sza: float = 30.0,
    profile: str = "us",
    altitude: float | str = 0.0,
    pwv: float = 5.0,
    ozone: float = 300.0,
    wl_min: float = 300.0,
    wl_max: float = 2500.0,
    albedo: float = 0.2,
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Run uvspec with external aerosol optical property file.

    Args:
        aerosol_file_type: Type of aerosol file (explicit, gg, ssa, tau, moments).
        aerosol_file_path: Path to the aerosol file.
        sza: Solar zenith angle in degrees.
        profile: Atmospheric profile name.
        altitude: Surface altitude in km.
        pwv: Precipitable water vapor in mm.
        ozone: Ozone column in DU.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        albedo: Surface albedo.
        streams: Number of DISORT streams.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with irradiance vs wavelength.
    """
    from pyradtran.presets import resolve_altitude

    resolved_altitude = resolve_altitude(altitude)

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolved_altitude)
        .set_mol_modify("H2O", pwv, "MM")
        .set_mol_modify("O3", ozone, "DU")
        .set_source_solar(sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="disort", streams=streams)
        .set_output(
            quantities=["lambda", "edir", "edn", "eup"],
            format="netcdf",
            quiet=True,
            zout=[0, 100],
        )
        .set_surface(albedo=albedo)
    )

    if aerosol_file_path is not None:
        scene = scene.set_aerosol(
            file=(aerosol_file_type, aerosol_file_path),
        )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_cloudy_scene(
    ic_properties: str = "fu",
    ic_tau: float | None = None,
    ic_habit: str | None = None,
    wc_properties: str | None = None,
    wc_tau: float | None = None,
    sza: float = 30.0,
    profile: str = "us",
    altitude: float | str = 0.0,
    pwv: float = 5.0,
    ozone: float = 300.0,
    wl_min: float = 300.0,
    wl_max: float = 2500.0,
    albedo: float = 0.2,
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Run uvspec with cloud layer.

    Args:
        ic_properties: Ice cloud parameterization (fu, yang, key, baum, etc.).
        ic_tau: Ice cloud optical thickness (set via ic_modify).
        ic_habit: Ice crystal habit type.
        wc_properties: Water cloud parameterization (hu, echam4).
        wc_tau: Water cloud optical thickness.
        sza: Solar zenith angle in degrees.
        profile: Atmospheric profile name.
        altitude: Surface altitude in km.
        pwv: Precipitable water vapor in mm.
        ozone: Ozone column in DU.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        albedo: Surface albedo.
        streams: Number of DISORT streams.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with irradiance vs wavelength.
    """
    from pyradtran.models.cloud import CloudModifyEntry
    from pyradtran.presets import resolve_altitude

    resolved_altitude = resolve_altitude(altitude)

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolved_altitude)
        .set_mol_modify("H2O", pwv, "MM")
        .set_mol_modify("O3", ozone, "DU")
        .set_source_solar(sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="disort", streams=streams)
        .set_output(
            quantities=["lambda", "edir", "edn", "eup"],
            format="netcdf",
            quiet=True,
            zout=[0, 100],
        )
        .set_surface(albedo=albedo)
    )

    cloud_kwargs: dict = {}
    if ic_properties:
        cloud_kwargs["ic_properties"] = ic_properties
    if ic_habit:
        cloud_kwargs["ic_habit"] = ic_habit
    if wc_properties:
        cloud_kwargs["wc_properties"] = wc_properties

    scene = scene.set_cloud(**cloud_kwargs)

    if ic_tau is not None and scene.cloud is not None:
        entry = CloudModifyEntry(variable="tau", action="set", value=ic_tau)
        ic_modify = list(scene.cloud.ic_modify) + [entry]
        scene = scene.set_cloud(**{**cloud_kwargs, "ic_modify": ic_modify})
    if wc_tau is not None and scene.cloud is not None:
        entry = CloudModifyEntry(variable="tau", action="set", value=wc_tau)
        wc_modify = list(scene.cloud.wc_modify) + [entry]
        scene = scene.set_cloud(**{**cloud_kwargs, "wc_modify": wc_modify})

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)
