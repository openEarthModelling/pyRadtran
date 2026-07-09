LEGO Blocks Aerosol API
=======================

pyRadtran models aerosols as composable **blocks**. A *species block* carries
mass-normalized optics (e.g. :class:`~pyradtran.models.aerosol_composite.MieSpecies`);
a *vertical profile* places mass in the column; a
:class:`~pyradtran.models.blocks.PlacedBlock` binds the two. Any number of
blocks are externally mixed into one
:class:`~pyradtran.models.aerosol_composite.CompositeAerosol`, written as a
single explicit ``.master`` / ``.LAYER`` file set.

Vertical profiles
-----------------

Three concrete profiles implement the
:class:`~pyradtran.models.blocks.VerticalProfile` protocol:

.. code-block:: python

    from pyradtran.models.blocks import MassProfile, ExponentialProfile, TabulatedProfile

    mass = MassProfile(kg_m3_per_layer=(1e-7, 2e-7, 1e-7, 1e-8))      # explicit per-layer kg/m^3
    expo = ExponentialProfile(rho0_kg_m3=1e-6, scale_height_km=2.0)   # rho0 * exp(-z/H)
    tab = TabulatedProfile(z_km=(0.0, 2.0, 4.0, 8.0), kg_m3=(1e-6, 5e-7, 1e-7, 0.0))
    print(type(mass).__name__, type(expo).__name__, type(tab).__name__)

Building and running a composite
--------------------------------

Build a Mie species, invert a target optical depth into a mass profile with
:func:`~pyradtran.models.blocks.od_to_mass_profile`, wrap it in a
``PlacedBlock``, assemble the composite, attach it, and run DISORT. Each
composite needs the wavelength grid (µm), the altitude grid (km, descending),
and a writable ``output_dir``:

.. code-block:: python

    from pyradtran import Scene, Runner
    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]
    ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[1.53, 1.53, 1.53], k_imag=[0.008, 0.008, 0.008])
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.50, "sigma_g": 2.2})
    dust = MieSpecies(refractive_index=ri, size_distribution=sd,
                      particle_density_kg_m3=2600.0, integration_config=IntegrationConfig(), name="dust")

    profile = od_to_mass_profile(dust, tau_ref=0.20, ref_nm=550.0,
                                 altitude_km=altitude_km, scale_height_km=3.0)
    aerosol = CompositeAerosol(
        pieces=[PlacedBlock(block=dust, profile=profile)],
        wavelength_grid_um=[0.50, 0.55, 0.60],
        altitude_grid_km=altitude_km,
        n_legendre=32,
        output_dir=".",
    )

    scene = (
        Scene().set_atmosphere(profile="us")
        .set_source_solar(sza=30.0)
        .set_wavelength(500.0, 600.0)
        .set_solver(method="disort", streams=16, disort_intcor="moments")
        .set_output(quantities=["lambda", "edir", "edn", "eup"], format="ascii", zout=[0, 2, "toa"])
        .set_aerosol(aerosol)
    )
    result = Runner.execute(scene, data_path=None)
    print("surface edir:", float(result.edir.isel(zout=0).mean()))

Mix several blocks by adding more ``PlacedBlock`` items to ``pieces``; they are
combined with scattering-optical-depth weighting.

Direct (pre-computed) layer files
---------------------------------

For pre-computed explicit aerosol files, skip the profile entirely and use
:class:`~pyradtran.models.blocks.DirectLayerOpticsBlock`, which parses a
``.master`` / ``.LAYER`` set directly (the requested wavelength grid must
match the file's grid)::

    from pyradtran.models.blocks import DirectLayerOpticsBlock
    # block = DirectLayerOpticsBlock(master_path="my_aerosol.master", name="explicit")
