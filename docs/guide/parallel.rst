Parallel Execution
==================

Running multiple simulations in parallel is a common need for parameter sweeps, sensitivity studies, or ensemble runs.

Batch Execution with ``execute_many``
-------------------------------------

The :meth:`~pyradtran.core.runner.Runner.execute_many` method runs multiple scenes in parallel using a process pool::

    from pyradtran import Scene, Runner

    scenes = [
        Scene().set_source_solar(sza=sza).set_wavelength(300, 800)
        for sza in [10, 20, 30, 40, 50, 60]
    ]

    results = Runner.execute_many(scenes, max_workers=4)

    for sza, result in zip([10, 20, 30, 40, 50, 60], results):
        print(f"SZA={sza}: mean edir = {result.edir.mean().values}")

Configuration
-------------

Control parallelism via :class:`~pyradtran.core.runner.RunnerConfig`::

    from pyradtran import Runner, RunnerConfig

    Runner.configure(
        config=RunnerConfig(
            uvspec_exe="/usr/local/bin/uvspec",
            data_path="/usr/local/share/libRadtran/data",
            max_workers=8,
        )
    )

    results = Runner.execute_many(scenes)  # Uses configured max_workers

Or override per-call::

    results = Runner.execute_many(scenes, max_workers=2)

Error Handling
--------------

When a scene fails, the error is collected and raised as an exception after all other scenes complete::

    try:
        results = Runner.execute_many(scenes)
    except Exception as e:
        print(f"Batch failed: {e}")

Performance Considerations
--------------------------

- ``max_workers`` defaults to 4. For CPU-bound tasks, set to the number of physical CPU cores.
- Each worker spawns a separate process, so memory usage scales with ``max_workers``.
- Use ``RunnerConfig.timeout`` to prevent hung simulations from blocking the batch.
