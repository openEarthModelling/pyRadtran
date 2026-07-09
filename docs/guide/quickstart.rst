Quick Start
===========

Basic Transmittance Calculation
-------------------------------

The simplest way to use pyRadtran is through the :class:`~pyradtran.scene.Scene` builder API::

    from pyradtran import Scene, Runner

    scene = (
        Scene()
        .set_atmosphere(profile="us", altitude=2.663)
        .set_source_solar(sza=30.0)
        .set_wavelength(250.0, 1200.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], quantity="transmittance")
    )

    result = Runner.execute(scene, data_path="/usr/local/share/libRadtran/data")
    result.edir.plot()

.. note::
   Passing ``data_path=None`` (or omitting it) lets pyRadtran's
   :class:`~pyradtran.data.DataResolver` find the data automatically — it checks
   ``PYRADTRAN_DATA_PATH``, then ``LIBRADTRAN_DATA_FILES``, then
   ``LIBRADTRANDIR/data``, then the subset bundled in the wheel. A portable call
   is simply ``Runner.execute(scene, data_path=None)``. For strict offline runs,
   force the bundled subset with ``Runner.configure(RunnerConfig(bundled_only=True))``.

Global Configuration
--------------------

Avoid repeating ``data_path`` and ``uvspec_exe`` on every call by setting global defaults::

    from pyradtran import Runner

    Runner.configure(
        uvspec_exe="/usr/local/bin/uvspec",
        data_path="/usr/local/share/libRadtran/data",
    )

    # Now execute without repeating paths
    result = Runner.execute(scene)

See :class:`~pyradtran.core.runner.RunnerConfig` for all available configuration options.

Using Convenience Functions
---------------------------

For common tasks, use the high-level convenience API::

    from pyradtran import run_solar_transmittance, run_solar_radiance

    # Solar spectral transmittance
    transmittance = run_solar_transmittance(
        airmass=2.0,
        pwv=10.0,
        ozone=300.0,
        wl_min=300,
        wl_max=1200,
    )

    # Solar spectral radiance
    radiance = run_solar_radiance(
        sza=45.0,
        wl_min=400,
        wl_max=800,
    )

See :mod:`~pyradtran.convenience` for all available functions.
