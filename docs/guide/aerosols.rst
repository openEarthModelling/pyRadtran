Aerosol Models
==============

pyRadtran supports multiple aerosol model types through a unified interface.

OPAC Presets
------------

Use predefined OPAC aerosol types::

    from pyradtran import run_with_opac_preset, OpacPreset, OpacPresetName

    # Using convenience function
    result = run_with_opac_preset(
        preset="maritime_clean",
        sza=45.0,
        wl_min=400,
        wl_max=800,
    )

    # Using Scene builder
    from pyradtran import Scene

    scene = Scene().set_aerosol(
        OpacPreset(preset=OpacPresetName.MARITIME_CLEAN)
    )

Available presets include ``MARITIME_CLEAN``, ``MARITIME_POLLUTED``, ``MARITIME_TROPICAL``, ``URBAN``, ``DESERT``, etc.

Custom OPAC
-----------

Mix individual OPAC species with custom concentrations::

    from pyradtran import OpacCustom

    aerosol = OpacCustom(
        species={
            "INSO": 1000.0,   # Insoluble
            "WASO": 5000.0,   # Water soluble
            "SSAM": 200.0,    # Sea salt (accumulation)
        }
    )
    scene = Scene().set_aerosol(aerosol)

Pre-computed layer files (DirectLayerOpticsBlock)
-------------------------------------------------

For a pre-computed explicit aerosol file (a ``.master`` / ``.LAYER`` set),
wrap it in a :class:`~pyradtran.models.blocks.DirectLayerOpticsBlock` and mix
it into a :class:`~pyradtran.models.aerosol_composite.CompositeAerosol`::

    from pyradtran.models.aerosol_composite import CompositeAerosol
    from pyradtran.models.blocks import DirectLayerOpticsBlock

    aerosol = CompositeAerosol(
        pieces=[DirectLayerOpticsBlock(master_path="my_aerosol.master", name="ext")],
        wavelength_grid_um=[0.50, 0.55, 0.60],   # must match the file's grid
        altitude_grid_km=[8.0, 6.0, 4.0, 2.0, 0.0],
        n_legendre=32,
        output_dir=".",
    )
    scene = Scene().set_aerosol(aerosol)

.. note::
   ``DirectLayerOpticsBlock`` performs no wavelength resampling — the grids
   passed to ``CompositeAerosol`` must match the file's grid exactly.

Composite Aerosol with Mie Scattering
-------------------------------------

For advanced users, pyRadtran supports composite aerosols built from LEGO
"blocks". Each block is a mass-normalized species (here a Mie species); a
vertical profile places it in the column; ``PlacedBlock`` binds the two. Any
number of blocks are externally mixed into one ``CompositeAerosol``.

.. note::
   The composite needs the wavelength grid (µm), the altitude grid (km,
   strictly descending), and a writable ``output_dir`` for the explicit
   ``.master`` / ``.LAYER`` file set it writes.

.. code-block:: python

    from pyradtran import Scene, Runner
    from pyradtran.models.aerosol_composite import (
        CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
    )
    from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile

    altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]


    def _species(n_real, k_imag, r_g_um, sigma_g, density, name):
        ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[n_real] * 3, k_imag=[k_imag] * 3)
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": r_g_um, "sigma_g": sigma_g})
        return MieSpecies(
            refractive_index=ri, size_distribution=sd,
            particle_density_kg_m3=density, integration_config=IntegrationConfig(), name=name,
        )


    dust = _species(1.53, 0.008, 0.50, 2.2, 2600.0, "dust")
    soot = _species(1.75, 0.44, 0.05, 1.8, 1800.0, "soot")
    pieces = [
        PlacedBlock(block=dust, profile=od_to_mass_profile(
            dust, tau_ref=0.20, ref_nm=550.0, altitude_km=altitude_km, scale_height_km=3.0)),
        PlacedBlock(block=soot, profile=od_to_mass_profile(
            soot, tau_ref=0.05, ref_nm=550.0, altitude_km=altitude_km, scale_height_km=1.5)),
    ]
    aerosol = CompositeAerosol(
        pieces=pieces,
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
    print(result.edir.shape)

Aerosol Modification
--------------------

After setting an aerosol, scale or set a column property. ``variable`` is one
of ``tau``, ``ssa``, ``gg``, ``tau550``; ``action`` is ``scale`` or ``set``.
Multiple calls accumulate (each adds one directive)::

    scene = scene.set_aerosol_modify(variable="tau", action="scale", value=0.5)

Bulk Aerosol from Aerosol3D
---------------------------

To use pre-computed bulk optics (e.g. from Aerosol3D's
``BulkAerosolOpticsData``), wrap them in ``BulkSpecies`` (duck-typed; pyRadtran
does not import Aerosol3D itself). For a real Mie phase function instead of
Henyey-Greenstein, build a Mie species with ``phase_function="mie"``.

.. code-block:: python

    import numpy as np
    from pyradtran.models.aerosol_composite import BulkSpecies, CompositeAerosol
    from pyradtran.models.blocks import PlacedBlock, MassProfile

    # A tiny bulk object built inline so this snippet is self-contained. In practice
    # pass an Aerosol3D BulkAerosolOpticsData here.
    wl = np.array([0.50, 0.55, 0.60])


    class _TinyBulk:
        size_distribution = None
        effective_density_kg_m3 = 2600.0

        def to_dataset(self, n_legendre=32):
            import xarray as xr
            nlayer = 4
            mom = np.zeros((wl.size, nlayer, n_legendre))
            mom[:, :, 0] = 1.0
            return xr.Dataset(
                {
                    "beta_ext_per_mass": (("wavelength", "layer"), np.full((wl.size, nlayer), 0.5)),
                    "ssa": (("wavelength", "layer"), np.full((wl.size, nlayer), 0.9)),
                    "g": (("wavelength", "layer"), np.full((wl.size, nlayer), 0.3)),
                    "legendre_moments": (("wavelength", "layer", "n_legendre"), mom),
                },
                coords={"wavelength": wl},
            )


    block = BulkSpecies(bulk=_TinyBulk(), name="dust")
    piece = PlacedBlock(block=block, profile=MassProfile(kg_m3_per_layer=(1e-7,) * 4))
    aerosol = CompositeAerosol(
        pieces=[piece], wavelength_grid_um=list(wl),
        altitude_grid_km=[8.0, 6.0, 4.0, 2.0, 0.0], n_legendre=32, output_dir=".",
    )
    print("pieces:", [p.name for p in aerosol.pieces])

    # Real Mie phase function (instead of Henyey-Greenstein):
    # MieSpecies(..., phase_function="mie")
