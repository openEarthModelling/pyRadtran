pyRadtran
=========

A complete Python wrapper for `libRadtran <https://www.libradtran.org>`_ radiative transfer simulations.

Prerequisites
-------------

This library requires libRadtran to be installed separately. See the `README <https://github.com/openEarthModelling/pyRadtran/blob/main/README.md#prerequisites>`_ for installation instructions.

Quick Start
-----------

.. code-block:: python

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

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
