Data Layer and ``DataResolver``
================================

pyRadtran ships a curated subset of libRadtran data inside the wheel (~60 MB:
OPAC aerosol optics, AFGL atmospheres, reptran correlated-k, CRS
cross-sections, solar flux) and resolves data references through
:class:`~pyradtran.data.DataResolver`.

Resolution priority for the data root
-------------------------------------

:class:`~pyradtran.data.DataResolver` picks the data root in this order:

1. an explicit ``data_root`` passed to the constructor,
2. the ``LIBRADTRAN_DATA_FILES`` environment variable,
3. the ``LIBRADTRANDIR`` environment variable (its ``data/`` subdirectory),
4. the bundled subset packaged in the wheel.

Inspecting the resolver
-----------------------

.. code-block:: python

    from pyradtran.data import DataResolver

    r = DataResolver()
    print("data root:", r.data_root)
    print("bundled asset count:", len(r.list_bundled()))

Running with the resolver
-------------------------

When you call ``Runner.execute(scene, data_path=None)``, the Runner uses
:class:`~pyradtran.data.DataResolver`, so a scene runs with no explicit path:

.. code-block:: python

    from pyradtran import Scene, Runner
    from pyradtran.data import DataResolver

    scene = (
        Scene().set_atmosphere(profile="us")
        .set_source_solar(sza=30.0)
        .set_wavelength(500.0, 600.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii", zout=[0, "toa"])
    )
    result = Runner.execute(scene, data_path=None)
    print("edir shape:", result.edir.shape)

    issues = DataResolver().validate_scene(scene)
    print("validation issues:", [i.message for i in issues])

Strict offline mode
-------------------

For reproducible runs (or CI), force the bundled subset and ignore every
environment variable and explicit path:

.. code-block:: python

    from pyradtran.data import DataResolver
    from pyradtran.core.runner import Runner, RunnerConfig

    strict = DataResolver(bundled_only=True)
    print("forced bundled root:", strict.data_root)

    Runner.configure(RunnerConfig(bundled_only=True))
