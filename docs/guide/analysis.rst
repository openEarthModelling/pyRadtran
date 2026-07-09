Postprocessing and Component Attribution
========================================

Two kinds of analysis sit on top of a raw RT result: spectral **budgets**
(transmittance / reflectance / absorptance) and **component attribution**
(per-block contributions via leave-one-out).

T/R/A budget
------------

:func:`~pyradtran.core.postprocess.add_budget_vars` annotates a dataset with
``transmittance`` (downwelling at the surface over incident at TOA),
``reflectance`` (upwelling at TOA over incident at TOA), and ``absorptance``
(``1 - T - R``). :func:`~pyradtran.core.postprocess.compute_budget` returns
the same numbers as a typed :class:`~pyradtran.core.postprocess.BudgetResult`:

.. code-block:: python

    from pyradtran import Scene, Runner
    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
    from pyradtran.core.postprocess import add_budget_vars, compute_budget

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]


    def _species(n, k, rg, sig, rho, name):
        ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[n] * 3, k_imag=[k] * 3)
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": rg, "sigma_g": sig})
        return MieSpecies(refractive_index=ri, size_distribution=sd,
                          particle_density_kg_m3=rho, integration_config=IntegrationConfig(), name=name)


    dust = _species(1.53, 0.008, 0.50, 2.2, 2600.0, "dust")
    soot = _species(1.75, 0.44, 0.05, 1.8, 1800.0, "soot")
    pieces = [
        PlacedBlock(block=dust, profile=od_to_mass_profile(dust, 0.20, 550.0, altitude_km, 3.0)),
        PlacedBlock(block=soot, profile=od_to_mass_profile(soot, 0.05, 550.0, altitude_km, 1.5)),
    ]
    aerosol = CompositeAerosol(pieces=pieces, wavelength_grid_um=[0.50, 0.55, 0.60],
                               altitude_grid_km=altitude_km, n_legendre=32, output_dir=".")


    def build_scene(comp):
        return (Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0)
                .set_wavelength(500.0, 600.0).set_solver(method="disort", streams=16, disort_intcor="moments")
                .set_output(quantities=["lambda", "edir", "edn", "eup"], format="ascii", zout=[0, 2, "toa"])
                .set_aerosol(comp))


    rt = Runner.execute(build_scene(aerosol), data_path=None)
    annotated = add_budget_vars(rt)
    print("T,R,A @550nm:", float(annotated.transmittance.isel(wavelength=1)),
          float(annotated.reflectance.isel(wavelength=1)), float(annotated.absorptance.isel(wavelength=1)))
    budget = compute_budget(rt)
    print("BudgetResult wavelength count:", budget.wavelength.size)

Analytic grid evaluation (no RT)
--------------------------------

:func:`~pyradtran.core.postprocess.evaluate_composite_on_grid` and
:func:`~pyradtran.core.postprocess.evaluate_blocks_on_grid` compute mixed
optics analytically — no RT run — on a (wavelength, layer) grid:

.. code-block:: python

    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
    from pyradtran.core.postprocess import evaluate_composite_on_grid, evaluate_blocks_on_grid

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]
    ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[1.53] * 3, k_imag=[0.008] * 3)
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.50, "sigma_g": 2.2})
    dust = MieSpecies(refractive_index=ri, size_distribution=sd,
                      particle_density_kg_m3=2600.0, integration_config=IntegrationConfig(), name="dust")
    aerosol = CompositeAerosol(pieces=[PlacedBlock(block=dust, profile=od_to_mass_profile(
        dust, 0.20, 550.0, altitude_km, 3.0))],
        wavelength_grid_um=[0.50, 0.55, 0.60], altitude_grid_km=altitude_km, n_legendre=32, output_dir=".")

    grid = evaluate_composite_on_grid(aerosol, [0.50, 0.55, 0.60], altitude_km, n_legendre=32)
    print("composite vars:", list(grid.data_vars))
    per_block = evaluate_blocks_on_grid(aerosol, [0.50, 0.55, 0.60], altitude_km, n_legendre=32)
    print("block names:", list(per_block))

Component attribution (leave-one-out)
-------------------------------------

:func:`~pyradtran.workflow.compute_component_attribution` runs the full
composite, then re-runs it with each block removed in turn, and subtracts. Its
third argument is *any* callable mapping a list of scenes to a list of parsed
datasets.

.. warning::
   Despite the parameter name ``execute_many`` and the function's docstring
   suggesting ``Runner.execute_many`` (parallel), that method **swallows
   exceptions** and **cannot pickle** ``CompositeAerosol`` scenes. Pass a
   **sequential** runner instead:

.. code-block:: python

    from pyradtran import Scene, Runner
    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
    from pyradtran.workflow import compute_component_attribution

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]


    def _species(n, k, rg, sig, rho, name):
        ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[n] * 3, k_imag=[k] * 3)
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": rg, "sigma_g": sig})
        return MieSpecies(refractive_index=ri, size_distribution=sd,
                          particle_density_kg_m3=rho, integration_config=IntegrationConfig(), name=name)


    dust = _species(1.53, 0.008, 0.50, 2.2, 2600.0, "dust")
    soot = _species(1.75, 0.44, 0.05, 1.8, 1800.0, "soot")
    pieces = [
        PlacedBlock(block=dust, profile=od_to_mass_profile(dust, 0.20, 550.0, altitude_km, 3.0)),
        PlacedBlock(block=soot, profile=od_to_mass_profile(soot, 0.05, 550.0, altitude_km, 1.5)),
    ]
    aerosol = CompositeAerosol(pieces=pieces, wavelength_grid_um=[0.50, 0.55, 0.60],
                               altitude_grid_km=altitude_km, n_legendre=32, output_dir=".")


    def build_scene(comp):
        return (Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0)
                .set_wavelength(500.0, 600.0).set_solver(method="disort", streams=16, disort_intcor="moments")
                .set_output(quantities=["lambda", "edir", "edn", "eup"], format="ascii", zout=[0, 2, "toa"])
                .set_aerosol(comp))


    def run_sequential(scenes):
        return [Runner.execute(s, data_path=None) for s in scenes]


    attribution = compute_component_attribution(build_scene, aerosol, run_sequential)
    print("contributors:", list(attribution.contributions))
    print("full edir @550nm:", float(attribution.full.edir.isel(wavelength=1).isel(zout=0)))
