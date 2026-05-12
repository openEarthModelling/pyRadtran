Convenience Functions
=====================

The :mod:`~pyradtran.convenience` module provides high-level functions for common radiative transfer tasks.

Solar Transmittance
-------------------

Calculate solar spectral transmittance::

    from pyradtran import run_solar_transmittance

    result = run_solar_transmittance(
        airmass=2.0,
        pwv=10.0,
        ozone=300.0,
        profile="us",
        wl_min=300,
        wl_max=1200,
    )

Solar Radiance
--------------

Calculate solar spectral radiance (direct + diffuse irradiance)::

    from pyradtran import run_solar_radiance

    result = run_solar_radiance(
        sza=45.0,
        profile="ms",
        wl_min=400,
        wl_max=800,
    )

Thermal Brightness
------------------

Calculate thermal infrared brightness temperature::

    from pyradtran import run_thermal_brightness

    result = run_thermal_brightness(
        pwv=10.0,
        profile="ms",
        wl_min=2500,
        wl_max=50000,
    )

Aerosol Simulations
-------------------

With OPAC preset::

    from pyradtran import run_with_opac_preset

    result = run_with_opac_preset(
        preset="maritime_clean",
        sza=45.0,
    )

With custom OPAC::

    from pyradtran import run_with_opac_custom

    result = run_with_opac_custom(
        species={"WASO": 5000.0, "SSAM": 200.0},
        sza=45.0,
    )

Cloudy Scenes
-------------

Run with cloud configuration::

    from pyradtran import run_cloudy_scene

    result = run_cloudy_scene(
        water_file="wc.dat",
        ice_file="ic.dat",
        sza=30.0,
    )

Lidar
-----

Lidar/SSLidar simulation::

    from pyradtran import run_lidar

    result = run_lidar(
        lambda0=532.0,
        source="lidar",
    )

Polarized
---------

Polarized radiative transfer::

    from pyradtran import run_polarized

    result = run_polarized(
        sza=30.0,
        streams=16,
    )

3D Radiative Transfer
---------------------

3D scene simulation::

    from pyradtran import run_3d

    result = run_3d(
        sza=30.0,
        three_d_config={"nxp": 10, "nyp": 10},
    )

Satellite
---------

Satellite geometry simulation::

    from pyradtran import run_satellite

    result = run_satellite(
        geometry="nadir",
        sza=30.0,
    )
