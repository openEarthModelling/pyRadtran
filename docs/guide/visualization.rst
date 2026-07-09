Visualization
=============

``pyradtran.viz`` provides publication-style plots. matplotlib is imported
lazily and is an optional dependency — install it with ``pip install pyradtran[plot]``.

Theme and palette
-----------------

.. code-block:: python

    from pyradtran.viz import set_theme, get_palette
    set_theme("publication")
    print("palette:", get_palette(3))

RT result plots
---------------

These consume the ``xarray.Dataset`` returned by ``Runner.execute``. Build a
small composite scene, run it, and plot the spectral fluxes, a flux profile,
the T/R/A budget (via :func:`~pyradtran.core.postprocess.add_budget_vars`), and
a three-panel overview. ``plot_heating_rate`` is guarded because libRadtran
only emits a heating-rate column when heating-rate output is requested:

.. code-block:: python

    from pyradtran import Scene, Runner
    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
    from pyradtran.core.postprocess import add_budget_vars
    from pyradtran.core.output_parser import HEATING_RATE_COLUMN
    from pyradtran.viz import (
        plot_spectral, plot_flux_profile, plot_budget, plot_rt_overview, plot_heating_rate, save,
    )

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]
    ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[1.53] * 3, k_imag=[0.008] * 3)
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.50, "sigma_g": 2.2})
    dust = MieSpecies(refractive_index=ri, size_distribution=sd,
                      particle_density_kg_m3=2600.0, integration_config=IntegrationConfig(), name="dust")
    aerosol = CompositeAerosol(
        pieces=[PlacedBlock(block=dust, profile=od_to_mass_profile(
            dust, tau_ref=0.20, ref_nm=550.0, altitude_km=altitude_km, scale_height_km=3.0))],
        wavelength_grid_um=[0.50, 0.55, 0.60], altitude_grid_km=altitude_km, n_legendre=32, output_dir=".",
    )
    rt = Runner.execute(
        Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0)
        .set_wavelength(500.0, 600.0).set_solver(method="disort", streams=16, disort_intcor="moments")
        .set_output(quantities=["lambda", "edir", "edn", "eup"], format="ascii", zout=[0, 2, "toa"])
        .set_aerosol(aerosol),
        data_path=None,
    )

    save(plot_spectral(rt)[0], "spectral.png")
    save(plot_flux_profile(rt, variable="edir", wavelength_nm=550.0)[0], "flux_profile.png")
    save(plot_budget(add_budget_vars(rt))[0], "budget.png")
    save(plot_rt_overview(rt, wavelength_nm=550.0)[0], "overview.png")
    if HEATING_RATE_COLUMN in rt.data_vars:
        save(plot_heating_rate(rt, wavelength_nm=550.0)[0], "heating.png")

Composite and per-block diagnostics (no RT)
-------------------------------------------

These plot analytic mixing results from
:func:`~pyradtran.core.postprocess.evaluate_composite_on_grid` and
:func:`~pyradtran.core.postprocess.evaluate_blocks_on_grid`:

.. code-block:: python

    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
    from pyradtran.core.postprocess import evaluate_composite_on_grid, evaluate_blocks_on_grid
    from pyradtran.viz import plot_composite_optics, plot_block_profiles, save

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]
    ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[1.53] * 3, k_imag=[0.008] * 3)
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.50, "sigma_g": 2.2})
    dust = MieSpecies(refractive_index=ri, size_distribution=sd,
                      particle_density_kg_m3=2600.0, integration_config=IntegrationConfig(), name="dust")
    aerosol = CompositeAerosol(
        pieces=[PlacedBlock(block=dust, profile=od_to_mass_profile(
            dust, tau_ref=0.20, ref_nm=550.0, altitude_km=altitude_km, scale_height_km=3.0))],
        wavelength_grid_um=[0.50, 0.55, 0.60], altitude_grid_km=altitude_km, n_legendre=32, output_dir=".",
    )

    wl = [0.50, 0.55, 0.60]
    grid = evaluate_composite_on_grid(aerosol, wl, altitude_km, n_legendre=32)
    save(plot_composite_optics(grid, quantity="tau")[0], "composite_tau.png")

    blocks = evaluate_blocks_on_grid(aerosol, wl, altitude_km, n_legendre=32)
    save(plot_block_profiles(blocks, quantity="tau")[0], "block_tau.png")
