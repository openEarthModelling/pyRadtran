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

External Aerosol Files
----------------------

Use pre-computed aerosol optical properties from external files::

    from pyradtran import ExternalAerosol, ExternalFile

    aerosol = ExternalAerosol(
        files=[
            ExternalFile(path="aerosol_explicit.dat", type="explicit"),
        ]
    )
    scene = Scene().set_aerosol(aerosol)

Supported file types: ``"explicit"``, ``"gg"``, ``"ssa"``, ``"tau"``, ``"moments"``.

Composite Aerosol with Mie Scattering
-------------------------------------

For advanced users, pyRadtran supports composite aerosols computed via Mie scattering::

    from pyradtran import CompositeAerosol, MieSpecies
    from pyradtran.models.aerosol_composite import (
        SizeDistribution, RefractiveIndex, IntegrationConfig
    )

    aerosol = CompositeAerosol(
        species=[
            MieSpecies(
                name="dust",
                size_distribution=SizeDistribution.log_normal(
                    r_median=0.5,  # um
                    sigma_g=2.0,
                ),
                refractive_index=RefractiveIndex.from_constant(n=1.53, k=0.008),
                density=2.6,  # g/cm3
            ),
            MieSpecies(
                name="soot",
                size_distribution=SizeDistribution.log_normal(
                    r_median=0.05,
                    sigma_g=1.8,
                ),
                refractive_index=RefractiveIndex.from_constant(n=1.75, k=0.44),
                density=1.8,
            ),
        ],
        integration=IntegrationConfig(n_legendre=32),
    )

    scene = Scene().set_aerosol(aerosol)

Aerosol Modification
--------------------

Modify aerosol properties after setting::

    scene = scene.set_aerosol_modify(
        quantity="tau",
        value=0.3,
    )

Available quantities: ``"tau"``, ``"ssa"``, ``"gg"``, ``"reff"``, etc.
