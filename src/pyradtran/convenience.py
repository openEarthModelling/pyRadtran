"""High-level convenience functions for common radiative transfer tasks."""

from __future__ import annotations

import math

import xarray as xr

from pyradtran.core.runner import Runner
from pyradtran.models.aerosol import ExternalAerosol, OpacCustom, OpacPreset, OpacPresetName
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
            zout=[0, "toa"],
        )
        .set_surface(albedo=albedo)
    )

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
            zout=[0, "toa"],
        )
        .set_surface(albedo=albedo)
    )

    if aerosol_file_path is not None:
        scene = scene.set_aerosol(
            ExternalAerosol(files=[(aerosol_file_type, aerosol_file_path)]),
        )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_with_opac_preset(
    preset: str | OpacPresetName,
    library: str = "OPAC",
    species_names: list[str] | None = None,
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
    """Run uvspec with OPAC preset mixture profile.

    Args:
        preset: OPAC preset name (e.g. "continental_average") or OpacPresetName enum.
        library: OPAC library path or "OPAC" for default resolution.
        species_names: Optional species filter (e.g. ["inso", "soot"]).
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
    if isinstance(preset, str):
        preset = OpacPresetName(preset)

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
            zout=[0, "toa"],
        )
        .set_surface(albedo=albedo)
        .set_aerosol(OpacPreset(name=preset, library=library, species_names=species_names))
    )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_with_opac_custom(
    species_file: str,
    library: str = "OPAC",
    species_names: list[str] | None = None,
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
    """Run uvspec with OPAC custom species profile.

    Args:
        species_file: Path to an ASCII species mass concentration profile file.
        library: OPAC library path or "OPAC" for default resolution.
        species_names: Optional species filter.
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
            zout=[0, "toa"],
        )
        .set_surface(albedo=albedo)
        .set_aerosol(OpacCustom(
            species_file=species_file,
            library=library,
            species_names=species_names,
        ))
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
            zout=[0, "toa"],
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
    if ic_tau is not None:
        cloud_kwargs["ic_modify"] = [CloudModifyEntry(variable="tau", action="set", value=ic_tau)]
    if wc_tau is not None:
        cloud_kwargs["wc_modify"] = [CloudModifyEntry(variable="tau", action="set", value=wc_tau)]

    scene = scene.set_cloud(**cloud_kwargs)

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_lidar(
    area: float = 1.0,
    E0: float = 0.1,
    efficiency: float = 0.5,
    position: float = 0.0,
    range_bin: float = 0.1,
    n_ranges: int = 100,
    profile: str = "us",
    altitude: float = 0.0,
    wl_min: float = 300.0,
    wl_max: float = 1100.0,
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Run single scattering lidar simulation.

    Args:
        area: Telescope area in m^2.
        E0: Pulse energy in Joules.
        efficiency: Detector efficiency.
        position: Lidar position in km.
        range_bin: Range bin width in km.
        n_ranges: Number of range bins.
        profile: Atmospheric profile.
        altitude: Surface altitude in km.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        streams: Number of DISORT streams.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with lidar signal vs range.
    """
    from pyradtran.presets import resolve_altitude

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolve_altitude(altitude))
        .set_source_solar(sza=0.0)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="sslidar", streams=streams)
        .set_sslidar(
            area=area, E0=E0, efficiency=efficiency,
            position=position, range_bin=range_bin, n_ranges=n_ranges,
        )
        .set_output(format="netcdf", quiet=True)
    )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_polarized(
    profile: str = "us",
    altitude: float = 0.0,
    sza: float = 30.0,
    wl_min: float = 250.0,
    wl_max: float = 1200.0,
    photons: int = 100000,
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Run polarized Monte Carlo simulation.

    Args:
        profile: Atmospheric profile.
        altitude: Surface altitude in km.
        sza: Solar zenith angle in degrees.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        photons: Number of Monte Carlo photons.
        streams: Number of streams (for correlated-k).
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with polarized radiance.
    """
    from pyradtran.presets import resolve_altitude

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolve_altitude(altitude))
        .set_source_solar(sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="mystic", streams=streams)
        .set_aerosol(OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE))
        .set_mc(photons=photons, backward=True, polarisation=True)
        .set_output(quantities=["lambda", "uu_pol"], quiet=True, format="netcdf")
    )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_3d(
    atmosphere_file: str | None = None,
    profile: str = "us",
    altitude: float = 0.0,
    sza: float = 30.0,
    wl_min: float = 300.0,
    wl_max: float = 2500.0,
    photons: int = 100000,
    ipa: bool = False,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Run 3D Monte Carlo radiative transfer simulation.

    Args:
        atmosphere_file: Path to 3D atmospheric field file.
        profile: Standard atmosphere profile (for 1D fallback).
        altitude: Surface altitude in km.
        sza: Solar zenith angle in degrees.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        photons: Number of Monte Carlo photons.
        ipa: Use independent pixel approximation.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with 3D radiative transfer results.
    """
    from pyradtran.presets import resolve_altitude

    scene = (
        Scene()
        .set_atmosphere(profile=profile, altitude=resolve_altitude(altitude))
        .set_source_solar(sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method="mystic", streams=8)
        .set_mc(photons=photons, ipa=ipa)
        .set_output(format="netcdf", quiet=True)
    )

    three_d_kwargs = {}
    if atmosphere_file is not None:
        three_d_kwargs["atmosphere_file"] = atmosphere_file
    if three_d_kwargs:
        scene = scene.set_three_d(**three_d_kwargs)

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def run_satellite(
    geometry: str = "MPS",
    pixel: tuple[int, int] | None = None,
    profile: str = "us",
    sza: float = 60.0,
    wl_min: float = 300.0,
    wl_max: float = 2500.0,
    solver: str = "disort",
    streams: int = 16,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
) -> xr.Dataset:
    """Run satellite-viewing radiative transfer simulation.

    Args:
        geometry: Satellite geometry name (e.g., MPS, SENTINEL2A).
        pixel: Optional (x, y) pixel coordinates for satellite_geometry file.
        profile: Standard atmosphere profile.
        sza: Solar zenith angle in degrees.
        wl_min: Minimum wavelength in nm.
        wl_max: Maximum wavelength in nm.
        solver: RTE solver name.
        streams: Number of angular streams.
        uvspec_exe: Path to uvspec binary.
        data_path: Path to libRadtran data directory.

    Returns:
        xarray.Dataset with satellite-view radiance/irradiance.
    """
    scene = (
        Scene()
        .set_atmosphere(profile=profile)
        .set_satellite(geometry=geometry, pixel=pixel, source="solar", sza=sza)
        .set_wavelength(wl_min, wl_max)
        .set_solver(method=solver, streams=streams)
        .set_output(format="netcdf", quiet=True)
    )

    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)
