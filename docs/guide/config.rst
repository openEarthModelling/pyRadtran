YAML Configuration
==================

A pyRadtran experiment can be written as a single YAML file — scene, aerosol
and analysis intents — and run end-to-end without writing Python. The
:mod:`pyradtran.config` package maps the file onto exactly the builders the
Python API uses, so YAML and API-built scenes are interchangeable: for
equivalent input they produce **byte-identical uvspec input and
byte-identical aerosol layer files** (see `Round-trip guarantee`_).

.. code-block:: python

   from pyradtran.config import load_config, export_config

   experiment = {
       "config_version": 1,
       "name": "demo",
       "scene": {
           "atmosphere": {"profile": "us"},
           "source": {"sza": 30.0},
           "wavelength": {"min_nm": 400.0, "max_nm": 700.0},
           "solver": {"method": "disort", "streams": 16, "disort_intcor": "moments"},
           "surface": {"albedo": 0.1},
       },
   }
   loaded = load_config(experiment)                  # validates + builds; no files, no uvspec
   path = export_config(loaded.config, "demo.yaml")  # canonical YAML
   assert load_config(path).config == loaded.config  # re-loads identically

:func:`~pyradtran.config.load_config` accepts a YAML file path, a raw ``dict``
or an already-built :class:`~pyradtran.config.PyRadtranConfig` and returns a
``LoadedConfig`` (validated config, built :class:`~pyradtran.scene.Scene`, and
the optional composite aerosol). The schema is strict: unknown keys, unknown
block kinds and typo'd fields raise ``pydantic.ValidationError`` /
``ValueError`` instead of being silently ignored.

File layout
-----------

The schema version is pinned by ``config_version: 1``::

   config_version: 1
   name: my_experiment        # free-form label (default: "unnamed")
   scene: { ... }             # required — radiation scene
   aerosol: { ... }           # optional — LEGO composite aerosol
   analysis: { ... }          # optional — post-run intents

Scene
-----

Every sub-dict of ``scene`` is forwarded as keyword arguments to the matching
:class:`~pyradtran.scene.Scene` builder method:

.. list-table::
   :header-rows: 1

   * - Key
     - Builder method
     - Notes
   * - ``atmosphere``
     - ``set_atmosphere``
     - required; e.g. ``profile: us`` (US standard atmosphere)
   * - ``source``
     - ``set_source_solar`` / ``set_source_thermal``
     - inner key ``source: solar`` (default; needs ``sza``) or ``source: thermal``
   * - ``wavelength``
     - ``set_wavelength``
     - ``min_nm`` required, ``max_nm`` optional
   * - ``solver``
     - ``set_solver``
     - optional; defaults ``disort``, 16 streams, ``disort_intcor: null``, no pseudospherical
   * - ``surface``
     - ``set_surface``
     - optional; e.g. ``albedo: 0.1``
   * - ``output``
     - ``set_output``
     - optional; e.g. ``quantities``, ``zout``, ``format``

Aerosol
-------

The ``aerosol`` section builds one
:class:`~pyradtran.models.aerosol_composite.CompositeAerosol` from LEGO blocks
(see :doc:`aerosols`). Shared keys:

- ``wavelength_grid_um`` — required, µm; must match the block data
- ``altitude_grid_km`` — required, strictly descending, at least two levels
- ``n_legendre`` — Legendre moments per layer (default 32)
- ``output_dir`` — where the explicit ``.master`` / ``.LAYER`` files are
  written at run time. A relative path resolves against the *working*
  directory, not the config file; loading and validating never write files.
- ``blocks`` — list of blocks, discriminated by ``kind``:

.. list-table::
   :header-rows: 1

   * - ``kind``
     - Builds
     - Specific fields
   * - ``mie``
     - ``MieSpecies``
     - ``refractive_index`` (``wavelength_um`` / ``n_real`` / ``k_imag``), ``size_distribution``, ``particle_density_kg_m3``, ``phase_function`` (``hg`` default or ``mie``), ``integration``
   * - ``bulk``
     - ``BulkSpecies``
     - ``file`` — path to a NetCDF ``BulkAerosolOpticsData`` (Aerosol3D)
   * - ``opac_preset``
     - OPAC preset block
     - ``preset`` (e.g. ``maritime_clean``), ``rh_pct`` (50), ``species_names``, ``data_path``, ``n_legendre`` (32)
   * - ``explicit_layer``
     - ``DirectLayerOpticsBlock``
     - ``master_path`` — pre-computed explicit file set

``size_distribution`` carries ``kind`` (``lognormal``, ``modified_gamma``,
``discrete``, ``monodisperse``), a free-form ``params`` dict (for lognormal:
``r_g_um``, ``sigma_g``) and ``number_density_per_m3`` (default 1.0 — the
placement, not the distribution, sets the column loading).

Placements
----------

Each block carries exactly one ``placement``, discriminated by ``kind``:

.. list-table::
   :header-rows: 1

   * - ``kind``
     - Meaning
     - Fields
   * - ``od_inversion``
     - exponential mass profile scaled so the block's optical depth at ``ref_nm`` equals ``tau_ref``
     - ``tau_ref``, ``ref_nm`` (550.0), ``scale_height_km``
   * - ``exponential``
     - exponential profile by density
     - ``rho0_kg_m3``, ``scale_height_km``
   * - ``mass``
     - per-layer mass concentrations on the altitude grid
     - ``kg_m3_per_layer``
   * - ``tabulated``
     - arbitrary tabulated profile
     - ``z_km``, ``kg_m3`` (equal-length lists)

Analysis intents
----------------

The optional ``analysis`` section drives what
:func:`~pyradtran.config.run_config` does beyond the main flux run:

.. list-table::
   :header-rows: 1

   * - Key
     - Effect
   * - ``energy_conservation``
     - compute the column energy budget and assert ``F_inc = eup_TOA + (1-a)(edir+edn)_surf + F_abs_atm`` within ``tol`` (default 0.05); optional surface ``albedo`` override
   * - ``heating``
     - second uvspec run in heating mode (libRadtran's heating mode replaces flux output); rates merged into ``RunResult.rt``
   * - ``drf``
     - additional no-aerosol baseline run; ``RunResult.drf`` holds the spectral direct radiative forcing (negative = cooling)
   * - ``attribution``
     - N+1 leave-one-out runs; ``RunResult.attribution`` (see :doc:`analysis`)
   * - ``plots``
     - plot names resolved against a fixed registry (below)
   * - ``save_netcdf``
     - path; the main run is persisted to NetCDF (``RunResult.netcdf_path``)

Plot registry — valid ``analysis.plots`` names:

- ``rt_spectral``, ``rt_overview``, ``rt_budget`` — main run
- ``rt_flux_profile_edir`` / ``rt_flux_profile_edn`` / ``rt_flux_profile_eup``
- ``rt_heating_rate`` (requires ``heating: true``)
- ``drf_spectral`` (requires ``drf: true``)
- ``attribution_edir``, ``attribution_spectral`` (require ``attribution: true``)

Command line
------------

The ``pyradtran`` console script (also runnable as ``python -m pyradtran``)
wraps the front-end:

.. code-block:: console

   $ pyradtran validate canonical.yaml               # schema check; no uvspec, no files
   $ pyradtran run canonical.yaml                    # main run + analysis intents
   $ pyradtran run canonical.yaml --plot-dir plots   # redirect plots
   $ pyradtran export-config canonical.yaml -o copy.yaml

``run`` additionally accepts ``--uvspec`` (executable path) and ``--data-path``
(libRadtran data directory), both forwarded to the ``Runner``.

Canonical example walkthrough
-----------------------------

``examples/multicomponent_viz/canonical.yaml`` is the YAML twin of the
canonical multicomponent scene used by the example gallery and the regression
tests. It is **generated** from ``canonical.py`` by ``make_yaml.py`` — do not
edit it by hand. Trimmed to its skeleton:

.. code-block:: yaml

   config_version: 1
   name: multicomponent_viz_canonical
   scene:
     atmosphere: {profile: us}
     source: {sza: 30.0}
     wavelength: {min_nm: 401.0, max_nm: 699.0}
     solver: {method: disort, streams: 16, disort_intcor: moments, pseudospherical: true}
     surface: {albedo: 0.1}
     output:
       quantities: [lambda, edir, edn, eup]
       format: ascii
       zout: [0, 1, 2, 4, 6, 8, 10, toa]
       heating_rate: local
   aerosol:
     wavelength_grid_um: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
     altitude_grid_km: [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0]
     n_legendre: 32
     output_dir: output
     blocks:
       - kind: mie
         name: black_carbon
         refractive_index: {wavelength_um: [0.3, 0.4, 0.55, 0.7, 1.0],
                            n_real: [1.95, ...], k_imag: [0.79, ...]}
         size_distribution: {kind: lognormal, params: {r_g_um: 0.1, sigma_g: 2.0}}
         particle_density_kg_m3: 1800.0
         phase_function: hg
         placement: {kind: od_inversion, tau_ref: 0.15, ref_nm: 550.0, scale_height_km: 1.5}
       # ... two more mie blocks on the same grids:
       #   sulfate       tau_ref 0.15, scale_height_km 2.0
       #   mineral_dust  tau_ref 0.20, scale_height_km 3.0
   analysis: null

Three externally-mixed Mie blocks share the 7-wavelength / 7-altitude grids;
their ``od_inversion`` placements scale the exponential profiles so the column
holds τ550 = 0.15 + 0.15 + 0.20 = 0.5.

Because ``output_dir: output`` is relative, run the file from the example
directory so the generated ``.master`` / ``.LAYER`` files land next to it:

.. code-block:: console

   $ cd examples/multicomponent_viz
   $ pyradtran run canonical.yaml

(or copy the file and point ``aerosol.output_dir`` / ``--plot-dir`` elsewhere).

Round-trip guarantee
--------------------

The YAML front-end is not an approximation of the Python API — it is the same
code path:

1. A YAML config and its API-built twin produce **identical uvspec input
   text** (``test_yaml_matches_api_uvspec_text``).
2. The explicit aerosol ``.master`` layer files they write are
   **byte-identical** (``test_yaml_master_file_bytes_identical``).
3. :func:`~pyradtran.config.export_config` serializes a config as canonical
   YAML that re-loads into an equal config:
   ``load_config(export_config(cfg, p)).config == cfg``.
4. ``canonical.yaml`` is regenerated from ``canonical.py`` by ``make_yaml.py``
   and ``test_generated_canonical_yaml_matches_api`` fails if the two drift.

Guarantees 1, 2 and 4 are enforced by ``tests/test_config_roundtrip.py``;
the export round trip (3) by ``tests/test_config_schema.py``
(``test_export_config_round_trip``).
