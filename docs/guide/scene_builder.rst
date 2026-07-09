Scene Builder
=============

The :class:`~pyradtran.scene.Scene` class is the central configuration object in pyRadtran. It uses an **immutable, chainable API** — each ``set_*()`` method returns a new :class:`~pyradtran.scene.Scene` via ``copy.deepcopy()`` to avoid mutability traps.

Core Concepts
-------------

Immutable Builder Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^

Every ``set_*()`` call creates a new scene. This makes it safe to reuse base configurations::

    from pyradtran import Scene

    base = (
        Scene()
        .set_atmosphere(profile="us")
        .set_source_solar(sza=30.0)
        .set_wavelength(250.0, 1200.0)
    )

    # Each of these is independent
    scene_a = base.set_solver(method="disort", streams=16)
    scene_b = base.set_solver(method="twostream")

Explicit Cloning
^^^^^^^^^^^^^^^^

Use :meth:`~pyradtran.scene.Scene.clone` for an explicit deep copy::

    scene_copy = scene.clone()

Configuration Categories
------------------------

The scene builder supports all major uvspec configuration categories:

Atmosphere
^^^^^^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (see text above for imports)
    scene = Scene().set_atmosphere(profile="us", altitude=2.663)

Available profiles: ``"us"``, ``"ms"``, ``"mw"``, ``"tp"``, ``"ss"``, ``"sw"``.

Molecular Modifications
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (assumes `scene` from above)
    scene = scene.set_mol_modify("H2O", 10.0, "MM")  # mm precipitable water
    scene = scene.set_mol_modify("O3", 300.0, "DU")  # Dobson units

Source
^^^^^^

Solar source::

    scene = Scene().set_source_solar(sza=30.0)

Thermal source::

    scene = Scene().set_source_thermal()

Wavelength
^^^^^^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (see text above for imports)
    scene = Scene().set_wavelength(250.0, 1200.0)

Solver
^^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (see text above for imports)
    scene = Scene().set_solver(method="disort", streams=16)

Available solvers: ``"disort"``, ``"twostream"``, ``"rodents"``, ``"mystic"``, etc.

Output
^^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (see text above for imports)
    scene = Scene().set_output(
        quantities=["lambda", "edir", "edn"],
        quantity="transmittance",
        format="ascii",
    )

Surface
^^^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (see text above for imports)
    scene = Scene().set_surface(albedo=0.2)

Aerosol
^^^^^^^

See :doc:`aerosols` for detailed aerosol configuration.

Cloud
^^^^^

.. code-block:: python

    # skip-doc-check: progressive tutorial fragment (needs cloud data files
    scene = Scene().set_cloud(
        water_file="wc.dat",
        ice_file="ic.dat",
    )

Advanced Options
^^^^^^^^^^^^^^^^

For options not yet covered by the Python API, use raw keywords::

    scene = scene.add_raw_keyword("sza", "30.0")

Building Input
--------------

Generate the uvspec input text directly::

    input_text = scene.build_input(data_files_path="/path/to/data")
    print(input_text)

This is useful for debugging or when you need to run uvspec manually.
